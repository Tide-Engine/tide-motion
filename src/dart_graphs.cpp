#include "dart_graphs.hpp"
#include "dart_graph.hpp"
#include <stdexcept>
namespace motion {
namespace {
#include "generated/dart-graphs.inc"
#include "generated/dart-batch-graphs.inc"
#include "generated/dart-grad-graphs.inc"
}
std::span<const DartSlice> dart_tensor_slices(){return dart_slices;}
std::string dart_graph_schema(std::string_view file) {
    if(file=="denoiser.onnx")return denoiser_graph();
    if(file=="decoder.onnx")return decoder_graph();
    if(file=="policy.onnx")return policy_graph();
    if(file=="text.onnx")return text_graph();
    if(file=="denoiser-batch.onnx")return denoiser_batch_graph();
    if(file=="decoder-batch.onnx")return decoder_batch_graph();
    if(file=="policy-batch.onnx")return policy_batch_graph();
    if(file=="denoiser-grad-batch.onnx")return denoiser_grad_graph();
    if(file=="decoder-grad-batch.onnx")return decoder_grad_graph();
    if(file=="transition-batch.onnx")return transition_graph();
    if(file=="transition-grad-batch.onnx")return transition_grad_graph();
    if(file=="soma-points-batch.onnx")return soma_points_graph();
    if(file=="soma-points-grad-batch.onnx")return soma_points_grad_graph();
    throw std::runtime_error("Unknown native DART graph schema");
}
}
