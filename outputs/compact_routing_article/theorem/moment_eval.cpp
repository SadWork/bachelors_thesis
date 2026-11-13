#include <algorithm>
#include <vector>

inline double act(double z) { return z > 0 ? 2.0*z : 0.5*z; }

extern "C" void first_moments(
    const float *x, const float *weights, const float *bias,
    const int *group_indices, const int *group_by_index, const int *keep_indices,
    const float *keep_values, const double *group_mass, const int *budgets,
    int examples, int inputs, int neurons, int groups, int group_size,
    int k_max, int budget_count, float *out) {
    std::vector<double> group_x(groups), group_f(groups);
    std::vector<double> head_x(groups), head_f(groups), head_mass(groups);
    std::vector<int> selected(groups);
    for (int sample = 0; sample < examples; ++sample) {
        const float *row = x + sample*inputs;
        for (int g = 0; g < groups; ++g) {
            double sum = 0.0;
            const int *ids = group_indices + g*group_size;
            for (int j = 0; j < group_size; ++j) sum += row[ids[j]];
            group_x[g] = sum;
        }
        for (int m = 0; m < neurons; ++m) {
            double w = weights[m], b = bias[m];
            auto f = [row,w,b](int i) { return act(w*row[i]+b); };
            for (int g = 0; g < groups; ++g) {
                double sum = 0.0;
                const int *ids = group_indices + g*group_size;
                for (int j = 0; j < group_size; ++j) sum += f(ids[j]);
                group_f[g] = sum;
            }
            std::fill(head_x.begin(), head_x.end(), 0.0);
            std::fill(head_f.begin(), head_f.end(), 0.0);
            std::fill(head_mass.begin(), head_mass.end(), 0.0);
            std::fill(selected.begin(), selected.end(), 0);
            double head = 0.0;
            int prev = 0;
            for (int budget_id = 0; budget_id < budget_count; ++budget_id) {
                int k = budgets[budget_id];
                for (int j = prev; j < k; ++j) {
                    int i = keep_indices[m*k_max + j];
                    int g = group_by_index[i];
                    double value = f(i);
                    double p = keep_values[m*k_max+j];
                    head += p*value;
                    head_mass[g] += p;
                    head_f[g] += value;
                    head_x[g] += row[i];
                    ++selected[g];
                }
                prev = k;
                double exact = head, omitted_mean = head, whole_mean = head;
                for (int g = 0; g < groups; ++g) {
                    int tail_count = group_size-selected[g];
                    if (tail_count == 0) continue;
                    double tau = std::max(0.0, group_mass[m*groups+g]-head_mass[g]);
                    exact += (tau/tail_count)*(group_f[g]-head_f[g]);
                    omitted_mean += tau*act(w*(group_x[g]-head_x[g])/tail_count+b);
                    whole_mean += (tau/tail_count)*(group_size*act(w*group_x[g]/group_size+b)-head_f[g]);
                }
                int index = ((budget_id*3)*examples + sample)*neurons+m;
                out[index] = exact;
                out[index+examples*neurons] = omitted_mean;
                out[index+2*examples*neurons] = whole_mean;
            }
        }
    }
}
