#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <vector>

inline double activate(double z) { return z > 0.0 ? 2.0*z : 0.5*z; }

extern "C" void first_capped(
    const float *x, const float *weights, const float *bias,
    const int *group_indices, const int *keep_indices, const float *keep_values,
    const std::uint8_t *keep_mask, const double *tail_mass, const double *mass_cap,
    const double *residual_l1, const double *residual_l2,
    int examples, int inputs, int neurons, int groups, int group_size, int k,
    double *out) {
    std::vector<int> sorted(groups*group_size);
    for (int sample = 0; sample < examples; ++sample) {
        const float *row = x + sample*inputs;
        std::copy(group_indices, group_indices + groups*group_size, sorted.begin());
        for (int g = 0; g < groups; ++g) {
            auto beg = sorted.begin() + g*group_size;
            std::sort(beg, beg+group_size, [row](int a, int b) { return row[a] < row[b]; });
        }
        for (int m = 0; m < neurons; ++m) {
            auto f = [row, m, weights, bias](int index) {
                return activate(static_cast<double>(weights[m])*row[index] + bias[m]);
            };
            const std::uint8_t *mask = keep_mask + m*inputs;
            double head = 0.0;
            for (int j = 0; j < k; ++j) head += keep_values[m*k+j]*f(keep_indices[m*k+j]);
            double midpoint = head, lower = head, upper = head, symmetric_radius = 0.0;
            for (int g = 0; g < groups; ++g) {
                double tau = tail_mass[m*groups + g];
                if (tau <= 0.0) continue;
                const int *block = sorted.data() + g*group_size;
                double cap = mass_cap[m];
                double low_mass = tau, high_mass = tau;
                double sum = 0.0, squares = 0.0, min_f = 1e300, max_f = -1e300;
                int count = 0;
                for (int j = 0; j < group_size; ++j) {
                    int lo = block[weights[m] >= 0 ? j : group_size-j-1];
                    int hi = block[weights[m] >= 0 ? group_size-j-1 : j];
                    if (!mask[lo]) {
                        double value = f(lo);
                        double used = std::min(cap, low_mass);
                        lower += used*value;
                        low_mass -= used;
                        sum += value;
                        squares += value*value;
                        min_f = std::min(min_f, value);
                        max_f = std::max(max_f, value);
                        ++count;
                    }
                    if (!mask[hi] && high_mass > 0.0) {
                        double used = std::min(cap, high_mass);
                        upper += used*f(hi);
                        high_mass -= used;
                    }
                }
                midpoint += tau * sum / count;
                double tv = 0.5 * residual_l1[m*groups+g] * (max_f-min_f);
                double l2 = residual_l2[m*groups+g] *
                    std::sqrt(std::max(squares - sum*sum/count, 0.0));
                symmetric_radius += std::min(tv, l2);
            }
            int index = (sample*neurons+m)*5;
            out[index] = midpoint;
            out[index+1] = lower;
            out[index+2] = upper;
            out[index+3] = midpoint - symmetric_radius;
            out[index+4] = midpoint + symmetric_radius;
        }
    }
}
