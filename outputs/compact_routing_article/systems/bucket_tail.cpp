#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <arm_neon.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {
constexpr int width = 160;
constexpr int depth = 4;
constexpr int classes = 10;
constexpr int channels = 3;
constexpr int pixels = 1024;

inline float birelu(float x) {
    return x > 0.f ? 2.f * x : 0.5f * x;
}

void finish_routing(const float* first, const float* probabilities, const float* weights,
                    const float* biases, const float* head, const float* head_bias,
                    float* work, float* result) {
    const float* current = first;
    for (int layer = 1; layer < depth; ++layer) {
        float* next = work + width * (layer % 2);
        for (int row = 0; row < width; ++row) {
            float w = weights[layer * width + row], b = biases[layer * width + row];
            const float* p = probabilities + ((layer - 1) * width + row) * width;
            float sum = 0.f;
            for (int i = 0; i < width; ++i) sum += p[i] * birelu(w * current[i] + b);
            next[row] = sum;
        }
        current = next;
    }
    for (int c = 0; c < classes; ++c) {
        float sum = head_bias[c];
        for (int i = 0; i < width; ++i) sum += head[c * width + i] * current[i];
        result[c] = sum;
    }
}

template<int Tiles, class Probability = float, bool RemoveUniform = false>
void bucket_first(const float* x, const std::uint8_t* raw, const int* bucket_counts,
                  const std::uint8_t* lookup, const std::uint8_t* cuts,
                  const float* tail, const float* norm, const float* level,
                  const float* weights, const float* biases, int keep,
                  const Probability* head_prob, const std::uint16_t* indices,
                  float* moments, float* output) {
    int offsets[channels + 1] = {};
    for (int c = 0; c < channels; ++c) offsets[c + 1] = offsets[c] + (bucket_counts[c] + 1) * Tiles;
    float* masses = moments;
    float* sums = moments + offsets[channels];
    std::fill(moments, moments + 2 * offsets[channels], 0.f);

    for (int c = 0; c < channels; ++c) {
        float* m = masses + offsets[c], *s = sums + offsets[c];
        const float* xx = x ? x + c * pixels : nullptr;
        const std::uint8_t* bytes = raw ? raw + c * pixels : nullptr;
        const std::uint8_t* lut = lookup + c * 256;
        const float* levels = level + c * 256;
        float scale = norm[c], shift = norm[channels + c];
        for (int i = 0; i < pixels; ++i) {
            int v = bytes ? bytes[i] : std::clamp(static_cast<int>(xx[i] * scale + shift), 0, 255);
            int patch = Tiles == 16 ? ((i >> 8) * 4 + ((i >> 3) & 3)) : 0;
            int index = (lut[v] + 1) * Tiles + patch;
            m[index] += 1.f;
            s[index] += xx ? xx[i] : levels[v];
        }
        for (int bin = 1; bin <= bucket_counts[c]; ++bin) {
            for (int p = 0; p < Tiles; ++p) {
                m[bin * Tiles + p] += m[(bin - 1) * Tiles + p];
                s[bin * Tiles + p] += s[(bin - 1) * Tiles + p];
            }
        }
    }

    for (int row = 0; row < width; ++row) {
        float w = weights[row], b = biases[row];
        float base = (w > 0.f || (w == 0.f && b > 0.f)) ? 2.f : .5f;
        float low = w == 0.f ? 0.f : (w > 0.f ? -1.5f : 1.5f);
        float sum = 0.f;
        for (int c = 0; c < channels; ++c) {
            int cut = cuts[c * width + row];
            const float* m = masses + offsets[c] + cut * Tiles;
            const float* s = sums + offsets[c] + cut * Tiles;
            const float* total = sums + offsets[c] + bucket_counts[c] * Tiles;
            const float* q = RemoveUniform ? tail + row : tail + (row * channels + c) * Tiles;
            constexpr float n = pixels / Tiles;
            for (int p = 0; p < Tiles; ++p) {
                sum += q[p] * (base * (w * total[p] + b * n) + low * (w * s[p] + b * m[p]));
            }
        }
        const Probability* p = head_prob + row * keep;
        const std::uint16_t* ii = indices + row * keep;
        float correction = RemoveUniform ? tail[row] : 0.f;
        if (x) {
            for (int k = 0; k < keep; ++k) {
                sum += (static_cast<float>(p[k]) - correction) * birelu(w * x[ii[k]] + b);
            }
        } else {
            for (int k = 0; k < keep; ++k) {
                int index = ii[k];
                sum += (static_cast<float>(p[k]) - correction) *
                       birelu(w * level[(index >> 10) * 256 + raw[index]] + b);
            }
        }
        output[row] = sum;
    }
}

template<class Probability>
void refit_layer(const float* current, int keep, const Probability* p, const std::uint16_t* indices,
                 const float* q, const float* weights, const float* biases, const float* thresholds,
                 const std::uint8_t* order, const std::uint8_t* lookup, float* moment_work, float* next) {
    float* below_sum = moment_work + 2 * (width+1);
    float* below_count = below_sum + width;
    float total = 0.f;
    if (lookup) {
        float* mass = moment_work;
        float* sums = mass + width+1;
        std::fill(moment_work, below_sum, 0.f);
        for (int i = 0; i < width; ++i) {
            float value = current[i];
            std::uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            auto key = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
            int cut = lookup[key >> 20];
            while (cut < width && thresholds[cut] <= value) ++cut;
            mass[cut] += 1.f;
            sums[cut] += value;
        }
        float count = 0.f;
        for (int i = 0; i < width; ++i) {
            count += mass[i];
            total += sums[i];
            int row = order[i];
            below_sum[row] = total;
            below_count[row] = count;
        }
        total += sums[width];
    } else {
        float* sorted = moment_work;
        std::copy_n(current, width, sorted);
        std::sort(sorted, sorted+width);
        for (int i = 0; i < width; ++i) total += sorted[i];
        int j = 0;
        float below = 0.f;
        for (int i = 0; i < width; ++i) {
            while (j < width && sorted[j] < thresholds[i]) below += sorted[j++];
            int row = order[i];
            below_sum[row] = below;
            below_count[row] = static_cast<float>(j);
        }
    }
    for (int row = 0; row < width; ++row) {
        float w = weights[row], b = biases[row];
        float base = (w > 0.f || (w == 0.f && b > 0.f)) ? 2.f : .5f;
        float low = w == 0.f ? 0.f : (w > 0.f ? -1.5f : 1.5f);
        float all = w * total + b * width;
        float active = w * below_sum[row] + b * below_count[row];
        float uniform = q[row];
        float sum = uniform * (base * all + low * active);
        const auto* pp = p + row * keep;
        const auto* ii = indices + row * keep;
        for (int k = 0; k < keep; ++k) {
            sum += (static_cast<float>(pp[k]) - uniform) * birelu(w * current[ii[k]] + b);
        }
        next[row] = sum;
    }
}

template<class Probability>
void refit_route(const float* x, const std::uint8_t* raw, int batch, const int* keep,
                 const int* bucket_counts, const std::uint8_t* lookup, const std::uint8_t* cuts,
                 const float* uniform, const float* norm, const float* level, const Probability* p,
                 const std::uint16_t* indices, const float* weights, const float* biases,
                 const float* thresholds, const std::uint8_t* order, const std::uint8_t* hidden_lookup,
                 const float* head, const float* head_bias, float* moments, float* work, float* result) {
    for (int sample = 0; sample < batch; ++sample) {
        const float* xx = x ? x + sample * channels * pixels : nullptr;
        const std::uint8_t* bytes = raw ? raw + sample * channels * pixels : nullptr;
        bucket_first<1, Probability, true>(xx, bytes, bucket_counts, lookup, cuts, uniform, norm, level,
                                           weights, biases, keep[0], p, indices, moments, work);
        const float* current = work;
        int offset = keep[0] * width;
        for (int layer = 1; layer < depth; ++layer) {
            float* next = work + width * (layer % 2);
            refit_layer(current, keep[layer], p+offset, indices+offset, uniform+layer*width,
                        weights+layer*width, biases+layer*width, thresholds+(layer-1)*width,
                        order+(layer-1)*width,
                        hidden_lookup ? hidden_lookup+(layer-1)*4096 : nullptr, moments, next);
            offset += keep[layer] * width;
            current = next;
        }
        for (int c = 0; c < classes; ++c) {
            float sum = head_bias[c];
            for (int i = 0; i < width; ++i) sum += head[c * width + i] * current[i];
            result[sample * classes + c] = sum;
        }
    }
}

void dense_fp32(const float* input, int batch, const float* weights, const float* biases,
                const float* head, const float* head_bias, float* work, float* result) {
    const float* current = input;
    int offset = 0;
    for (int layer = 0; layer < depth; ++layer) {
        int n = layer == 0 ? channels * pixels : width;
        float* next = work + (layer % 2) * batch * width;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, width, n, 1.f,
                    current, n, weights + offset, n, 0.f, next, width);
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < width; ++j) {
                next[i * width + j] = birelu(next[i * width + j] + biases[layer * width + j]);
            }
        }
        current = next;
        offset += n * width;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, classes, width, 1.f,
                current, width, head, width, 0.f, result, classes);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < classes; ++j) result[i * classes + j] += head_bias[j];
    }
}

float dot_fp16(const float* x, const __fp16* w, int n) {
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    for (int i = 0; i < n; i += 16) {
        float16x8_t u = vld1q_f16(w+i), v = vld1q_f16(w+i+8);
        a = vfmaq_f32(a, vld1q_f32(x+i), vcvt_f32_f16(vget_low_f16(u)));
        b = vfmaq_f32(b, vld1q_f32(x+i+4), vcvt_f32_f16(vget_high_f16(u)));
        c = vfmaq_f32(c, vld1q_f32(x+i+8), vcvt_f32_f16(vget_low_f16(v)));
        d = vfmaq_f32(d, vld1q_f32(x+i+12), vcvt_f32_f16(vget_high_f16(v)));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a,b), vaddq_f32(c,d)));
}

void dense_fp16(const float* input, int batch, const __fp16* weights, const float* biases,
                const __fp16* head, const float* head_bias, float* work, float* result) {
    for (int sample = 0; sample < batch; ++sample) {
        const float* current = input + sample * channels * pixels;
        int offset = 0;
        for (int layer = 0; layer < depth; ++layer) {
            int n = layer == 0 ? channels * pixels : width;
            float* next = work + (layer % 2) * width;
            for (int row = 0; row < width; ++row) {
                next[row] = birelu(dot_fp16(current, weights + offset + row*n, n) + biases[layer*width+row]);
            }
            current = next;
            offset += n * width;
        }
        for (int c = 0; c < classes; ++c) result[sample*classes+c] = dot_fp16(current, head+c*width, width) + head_bias[c];
    }
}

const float* normalize(const float* x, const std::uint8_t* raw, const float* level, int batch, float* work) {
    if (x) return x;
    for (int sample = 0; sample < batch; ++sample) {
        for (int c = 0; c < channels; ++c) {
            int offset = (sample*channels+c)*pixels;
            for (int i = 0; i < pixels; ++i) work[offset+i] = level[c*256+raw[offset+i]];
        }
    }
    return work;
}
} // namespace

extern "C" void optimized_route(const float* x, const std::uint8_t* raw, int batch, int tiles, int keep,
                    const int* bucket_counts, const std::uint8_t* lookup, const std::uint8_t* cuts,
                    const float* tail, const float* norm, const float* level,
                    const float* head_prob, const std::uint16_t* indices, const float* probabilities,
                    const float* weights, const float* biases, const float* head, const float* head_bias,
                    float* moments, float* work, float* result) {
    for (int sample = 0; sample < batch; ++sample) {
        const float* xx = x ? x + sample * channels * pixels : nullptr;
        const std::uint8_t* bytes = raw ? raw + sample * channels * pixels : nullptr;
        if (tiles == 16)
            bucket_first<16>(xx, bytes, bucket_counts, lookup, cuts, tail, norm, level,
                             weights, biases, keep, head_prob, indices, moments, work);
        else
            bucket_first<1>(xx, bytes, bucket_counts, lookup, cuts, tail, norm, level,
                            weights, biases, keep, head_prob, indices, moments, work);
        finish_routing(work, probabilities, weights, biases, head, head_bias, work, result + sample * classes);
    }
}

extern "C" void optimized_dense(const float* x, const std::uint8_t* raw, int batch, const float* level,
                       const float* weights, const float* biases, const float* head,
                       const float* head_bias, float* normalization_work, float* work, float* result) {
    dense_fp32(normalize(x, raw, level, batch, normalization_work), batch,
               weights, biases, head, head_bias, work, result);
}

extern "C" void optimized_dense_fp16(const float* x, const std::uint8_t* raw, int batch, const float* level,
                        const __fp16* weights, const float* biases, const __fp16* head,
                        const float* head_bias, float* normalization_work, float* work, float* result) {
    dense_fp16(normalize(x, raw, level, batch, normalization_work), batch,
               weights, biases, head, head_bias, work, result);
}

extern "C" void optimized_refit_route(const float* x, const std::uint8_t* raw, int batch,
                 int half, const int* keep,
                 const int* bucket_counts, const std::uint8_t* lookup, const std::uint8_t* cuts,
                 const float* uniform, const float* norm, const float* level, const void* p,
                 const std::uint16_t* indices, const float* weights, const float* biases,
                 const float* thresholds, const std::uint8_t* order, const std::uint8_t* hidden_lookup,
                 const float* head, const float* head_bias, float* moments, float* work, float* result) {
    if (half) refit_route(x, raw, batch, keep, bucket_counts, lookup, cuts, uniform, norm, level,
                          static_cast<const __fp16*>(p), indices, weights, biases,
                          thresholds, order, hidden_lookup, head, head_bias, moments, work, result);
    else refit_route(x, raw, batch, keep, bucket_counts, lookup, cuts, uniform, norm, level,
                     static_cast<const float*>(p), indices, weights, biases,
                     thresholds, order, hidden_lookup, head, head_bias, moments, work, result);
}
