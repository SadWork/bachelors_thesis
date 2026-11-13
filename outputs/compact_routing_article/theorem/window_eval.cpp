#include <cstddef>
#include <vector>

extern "C" void first_outputs(const float *x, const float *weights,
                             const float *bias, const float *probabilities,
                             int examples, int inputs, int neurons, int methods,
                             float *out) {
    std::vector<float> activations(inputs);
    for (int b = 0; b < examples; ++b) {
        const float *row = x + b * inputs;
        for (int m = 0; m < neurons; ++m) {
            for (int i = 0; i < inputs; ++i) {
                float z = weights[m] * row[i] + bias[m];
                activations[i] = z > 0.0f ? 2.0f * z : 0.5f * z;
            }
            for (int a = 0; a < methods; ++a) {
                float sum = 0.0f;
                const float *p = probabilities + (m * methods + a) * inputs;
                for (int i = 0; i < inputs; ++i) sum += p[i] * activations[i];
                out[(a * examples + b) * neurons + m] = sum;
            }
        }
    }
}
