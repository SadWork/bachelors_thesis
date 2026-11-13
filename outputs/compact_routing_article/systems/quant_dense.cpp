#include <Accelerate/Accelerate.h>
#include <array>
#include <cstdint>
#include <memory>

namespace {
constexpr int width = 160, inputs = 3072, depth = 4, classes = 10;
struct DenseNet {
    std::array<BNNSFilter, depth + 1> layer{};
    bool all_half = false;
    ~DenseNet() { for (auto filter : layer) if (filter) BNNSFilterDestroy(filter); }
};

BNNSNDArrayDescriptor vector(int size, void* data = nullptr) {
    BNNSNDArrayDescriptor desc{};
    desc.layout = BNNSDataLayoutVector;
    desc.size[0] = size;
    desc.stride[0] = 1;
    desc.data = data;
    desc.data_type = BNNSDataTypeFloat32;
    return desc;
}

float birelu(float x) {
    return x > 0.f ? 2.f*x : .5f*x;
}
} // namespace

extern "C" void* bnns_dense_create(int bytes, int all_half, void* weight, void* bias, void* head, void* head_bias,
                                   const float* scale) {
    std::unique_ptr<DenseNet> net(new DenseNet{});
    net->all_half = all_half;
    BNNSFilterParameters filter_options{};
    filter_options.flags = BNNSFlagsUseClientPtr;
    filter_options.n_threads = 1;
    int offset = 0;
    for (int layer = 0; layer <= depth; ++layer) {
        int n = layer == 0 ? inputs : width;
        int m = layer == depth ? classes : width;
        BNNSLayerParametersFullyConnected parameters{};
        parameters.i_desc = vector(n);
        parameters.o_desc = vector(m);
        parameters.bias = vector(m, layer == depth ? head_bias : static_cast<char*>(bias)+layer*width*(all_half?2:4));
        if (all_half) {
            parameters.i_desc.data_type = BNNSDataTypeFloat16;
            parameters.o_desc.data_type = BNNSDataTypeFloat16;
            parameters.bias.data_type = BNNSDataTypeFloat16;
        }
        parameters.activation.function = BNNSActivationFunctionIdentity;
        auto& w = parameters.w_desc;
        w.layout = BNNSDataLayoutRowMajorMatrix;
        w.size[0] = n;
        w.size[1] = m;
        w.stride[0] = 1;
        w.stride[1] = n;
        w.data = layer == depth ? head : static_cast<char*>(weight) + offset * bytes;
        w.data_type = bytes == 1 ? BNNSDataTypeInt8 : (bytes == 2 ? BNNSDataTypeFloat16 : BNNSDataTypeFloat32);
        w.data_scale = bytes == 1 ? scale[layer] : 1.f;
        net->layer[layer] = BNNSFilterCreateLayerFullyConnected(&parameters, &filter_options);
        if (!net->layer[layer]) return nullptr;
        offset += n*m;
    }
    return net.release();
}

extern "C" void bnns_dense_free(void* handle) {
    delete static_cast<DenseNet*>(handle);
}

extern "C" int bnns_dense_run(void* handle, const float* x, int batch, float* work, float* y) {
    auto* net = static_cast<DenseNet*>(handle);
    if (net->all_half) {
        auto* buffer = reinterpret_cast<__fp16*>(work);
        auto* input = buffer + 2 * batch * width;
        auto* logits = input + batch * inputs;
        for (int i = 0; i < batch*inputs; ++i) input[i] = x[i];
        const __fp16* current = input;
        for (int layer = 0; layer <= depth; ++layer) {
            int n = layer == 0 ? inputs : width;
            int m = layer == depth ? classes : width;
            auto* next = layer == depth ? logits : buffer + (layer % 2) * batch * width;
            int status = BNNSFilterApplyBatch(net->layer[layer], batch, current, n, next, m);
            if (status != 0) return status;
            if (layer < depth)
                for (int i = 0; i < batch*m; ++i) next[i] = birelu(next[i]);
            current = next;
        }
        for (int i = 0; i < batch*classes; ++i) y[i] = logits[i];
        return 0;
    }
    const float* current = x;
    for (int layer = 0; layer <= depth; ++layer) {
        int n = layer == 0 ? inputs : width;
        int m = layer == depth ? classes : width;
        float* next = layer == depth ? y : work + (layer % 2) * batch * width;
        int status = BNNSFilterApplyBatch(net->layer[layer], batch, current, n, next, m);
        if (status != 0) return status;
        if (layer < depth)
            for (int i = 0; i < batch*m; ++i) next[i] = birelu(next[i]);
        current = next;
    }
    return 0;
}
