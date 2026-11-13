#include <metal_stdlib>
using namespace metal;

kernel void mean_forward(
    device const float* x, device const float* w, device const float* p, device const float* bias,
    device float* out, constant uint& n, constant uint& m,
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    uint pair = tid / 32;
    uint example = pair / m;
    uint row = pair % m;
    float weight = w[row];
    float shift = bias[row];
    float acc = 0.0f;
    for (uint i = lane; i < n; i += 32) {
        float u = weight * x[example * n + i] + shift;
        acc += p[row * n + i] * (u > 0.0f ? 2.0f : 0.5f) * u;
    }
    float value = simd_sum(acc);
    if (lane == 0) out[pair] = value;
}

kernel void mean_grad_parameters(
    device const float* x, device const float* w, device const float* p, device const float* bias,
    device const float* dy, device float* dw_edges, device float* dp, device float* db_edges,
    constant uint& n, constant uint& m, constant uint& batch,
    uint edge [[thread_position_in_grid]]
) {
    uint row = edge / n;
    uint i = edge % n;
    float weight = w[row];
    float shift = bias[row];
    float sum_w = 0.0f, sum_p = 0.0f, sum_b = 0.0f;
    for (uint example = 0; example < batch; ++example) {
        float value = x[example * n + i];
        float u = weight * value + shift;
        float du = dy[example * m + row] * (u > 0.0f ? 2.0f : 0.5f);
        sum_w += du * value;
        sum_p += du * u;
        sum_b += du;
    }
    dw_edges[edge] = p[edge] * sum_w;
    dp[edge] = sum_p;
    db_edges[edge] = p[edge] * sum_b;
}

kernel void mean_grad_inputs(
    device const float* x, device const float* w, device const float* pt, device const float* bias,
    device const float* dy, device float* dx, constant uint& n, constant uint& m,
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    uint input = tid / 32;
    uint example = input / n;
    uint i = input % n;
    float value = x[input];
    float acc = 0.0f;
    for (uint row = lane; row < m; row += 32) {
        float weight = w[row];
        float u = weight * value + bias[row];
        float du = dy[example * m + row] * (u > 0.0f ? 2.0f : 0.5f);
        acc += du * pt[i * m + row] * weight;
    }
    float grad_input = simd_sum(acc);
    if (lane == 0) dx[input] = grad_input;
}
