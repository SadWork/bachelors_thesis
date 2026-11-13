#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int width = 160;
constexpr int depth = 4;
constexpr int classes = 10;

inline float birelu(float x) {
    return x > 0.0f ? 2.0f * x : 0.5f * x;
}

struct Prefix {
    std::vector<float> levels;
    std::vector<int> counts;
    std::vector<double> sums;

    void sorted(const float* x, int n) {
        levels.assign(x, x + n);
        std::sort(levels.begin(), levels.end());
        counts.resize(n + 1);
        sums.resize(n + 1);
        counts[0] = 0;
        sums[0] = 0;
        for (int i = 0; i < n; ++i) {
            counts[i + 1] = i + 1;
            sums[i + 1] = sums[i] + levels[i];
        }
    }

    void histogram(const float* x, int n, float mu, float sd, int stride = 0) {
        std::array<int, 256> histogram{};
        int rows = stride == 0 ? 1 : n;
        int pitch = stride == 0 ? n : stride;
        for (int y = 0; y < rows; ++y) {
            for (int i = 0; i < n; ++i) {
                int index = static_cast<int>((x[y * pitch + i] * sd + mu) * 255.0f + 0.5f);
                ++histogram[std::clamp(index, 0, 255)];
            }
        }
        levels.resize(256);
        counts.resize(257);
        sums.resize(257);
        counts[0] = 0;
        sums[0] = 0;
        for (int i = 0; i < 256; ++i) {
            levels[i] = (static_cast<float>(i) / 255.0f - mu) / sd;
            counts[i + 1] = counts[i] + histogram[i];
            sums[i + 1] = sums[i] + levels[i] * static_cast<double>(histogram[i]);
        }
    }

    double total(float w, float b) const {
        if (w == 0.0f) return counts.back() * static_cast<double>(birelu(b));
        float threshold = -b / w;
        auto cut = std::lower_bound(levels.begin(), levels.end(), threshold) - levels.begin();
        return total_at(w, b, static_cast<int>(cut));
    }

    double total_at(float w, float b, int cut) const {
        if (w == 0.0f) return counts.back() * static_cast<double>(birelu(b));
        double all = w * sums.back() + b * static_cast<double>(counts.back());
        double active = w * sums[cut] + b * static_cast<double>(counts[cut]);
        if (w > 0.0f) active = all - active;
        return 0.5 * all + 1.5 * std::max(active, 0.0);
    }
};
} // namespace

extern "C" void route_infer(const float* x, int batch, int inputs, const int* keep,
                            const float* values, const std::uint16_t* indices,
                            const float* uniform, const float* weights, const float* biases,
                            const float* head, const float* head_bias, int channels,
                            const float* mean, const float* sd, int tile,
                            const float* tile_uniform, const int* tile_cuts, float* result) {
    std::array<Prefix, 3> prefix;
    int tiles_side = tile ? 32 / tile : 0;
    int per_channel = tiles_side * tiles_side;
    int blocks = tile ? channels * per_channel : 0;
    std::vector<Prefix> tiles(blocks);
    std::array<std::array<float, width>, 2> work{};
    for (int sample = 0; sample < batch; ++sample) {
        const float* current = x + sample * inputs;
        int offset = 0;
        int index_offset = 0;
        for (int layer = 0; layer < depth; ++layer) {
            int n = layer == 0 ? inputs : width;
            int k = keep[layer];
            int parts = 0;
            bool tiled = layer == 0 && blocks > 0;
            if (k < n) {
                if (tiled) {
                    for (int c = 0; c < channels; ++c) {
                        for (int r = 0; r < tiles_side; ++r) {
                            for (int j = 0; j < tiles_side; ++j) {
                                int index = (c * tiles_side + r) * tiles_side + j;
                                const float* patch = current + c * 1024 + r * tile * 32 + j * tile;
                                tiles[index].histogram(patch, tile, mean[c], sd[c], 32);
                            }
                        }
                    }
                } else if (layer == 0 && channels > 0) {
                    parts = channels;
                    for (int c = 0; c < channels; ++c) {
                        prefix[c].histogram(current + c * (n / channels), n / channels, mean[c], sd[c]);
                    }
                } else {
                    parts = 1;
                    prefix[0].sorted(current, n);
                }
            }
            auto& next = work[layer % 2];
            for (int row = 0; row < width; ++row) {
                float w = weights[layer * width + row];
                float b = biases[layer * width + row];
                const float* p = values + offset + row * k;
                float sum = 0.0f;
                if (k == n) {
                    for (int i = 0; i < n; ++i) sum += p[i] * birelu(w * current[i] + b);
                } else {
                    const std::uint16_t* ids = indices + index_offset + row * k;
                    for (int i = 0; i < k; ++i) sum += p[i] * birelu(w * current[ids[i]] + b);
                    double tail = 0.0;
                    if (tiled) {
                        for (int c = 0; c < channels; ++c) {
                            int cut = tile_cuts[c * width + row];
                            for (int j = 0; j < per_channel; ++j) {
                                int block = c * per_channel + j;
                                tail += tile_uniform[row * blocks + block] * tiles[block].total_at(w, b, cut);
                            }
                        }
                        sum += tail;
                    } else {
                        for (int j = 0; j < parts; ++j) tail += prefix[j].total(w, b);
                        sum += uniform[layer * width + row] * tail;
                    }
                }
                next[row] = sum;
            }
            offset += k * width;
            if (k < n) index_offset += k * width;
            current = next.data();
        }
        for (int c = 0; c < classes; ++c) {
            float sum = head_bias[c];
            for (int i = 0; i < width; ++i) sum += head[c * width + i] * current[i];
            result[sample * classes + c] = sum;
        }
    }
}

extern "C" void dense_infer(const float* x, int batch, int inputs, const float* weights,
                            const float* biases, const float* head, const float* head_bias, float* result) {
    std::array<std::vector<float>, 2> work{
        std::vector<float>(batch * width), std::vector<float>(batch * width)};
    const float* current = x;
    int offset = 0;
    for (int layer = 0; layer < depth; ++layer) {
        int n = layer == 0 ? inputs : width;
        auto& next = work[layer % 2];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, width, n, 1.0f,
                    current, n, weights + offset, n, 0.0f, next.data(), width);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < width; ++j) {
                next[b * width + j] = birelu(next[b * width + j] + biases[layer * width + j]);
            }
        }
        current = next.data();
        offset += n * width;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, classes, width, 1.0f,
                current, width, head, width, 0.0f, result, classes);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < classes; ++c) result[b * classes + c] += head_bias[c];
    }
}
