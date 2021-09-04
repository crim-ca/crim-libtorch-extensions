#include "stdafx.h"
#pragma hdrstop

#include <string>

#include "nn/models/NFNet.h"


//#include "Base.h"
namespace vision {
namespace models {
namespace _nfnetimpl {

// TODO: allow different base_conv
ScaledStdConv2d myconv3x3(
    int64_t in,
    int64_t out,
    int64_t stride,
    int64_t groups) {
    torch::nn::Conv2dOptions O(in, out, 3);
    O.padding(1).stride(stride).groups(groups).bias(false);

    return ScaledStdConv2d(O);
}
// TODO: allow different base_conv
ScaledStdConv2d myconv1x1(int64_t in, int64_t out, int64_t stride) {

    torch::nn::Conv2dOptions O(in, out, 1);
    O.stride(stride).bias(false);
    return ScaledStdConv2d(O);
}

int CTE_API BasicBlock::expansion = 1;
int CTE_API Bottleneck::expansion = 4;

BasicBlock::BasicBlock(
    int64_t inplanes,
    int64_t planes,
    int64_t stride,
    const torch::nn::Sequential& downsample,
    int64_t groups,
    int64_t base_width,
    int64_t _dilation,
    double _alpha,
    double _beta,
    std::string activation

    )
    : stride(stride), downsample(downsample), alpha(_alpha), beta(_beta), activation(activation), dilation(_dilation)
    {
    TORCH_CHECK(
        groups == 1 && base_width == 64,
        "BasicBlock only supports groups=1 and base_width=64");

    TORCH_CHECK(
        dilation == 1,
        "BasicBlock only supports dilation=1");

    // Both conv1 and downsample layers downsample the input when stride != 1
    conv1 = myconv3x3(inplanes, planes, stride);
    conv2 = myconv3x3(planes, planes);


    register_module("conv1", conv1);
    register_module("conv2", conv2);


    if (!downsample.is_empty())
        register_module("downsample", this->downsample);
}

Bottleneck::Bottleneck(
    int64_t inplanes,
    int64_t planes,
    int64_t stride,
    const torch::nn::Sequential& downsample,
    int64_t groups,
    int64_t base_width,
    int64_t dilation,
    double _alpha,
    double _beta,
    std::string activation)
    : stride(stride), downsample(downsample), alpha(_alpha), beta(_beta), activation(activation) {

    auto width = int64_t(planes * (base_width / 64.)) * groups;

    // Both conv2 and downsample layers downsample the input when stride != 1
    conv1 = myconv1x1(inplanes, width);
    conv2 = myconv3x3(width, width, stride, groups);
    conv3 = myconv1x1(width, planes * expansion);


    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);

    if (!downsample.is_empty())
        register_module("downsample", this->downsample);
}

torch::Tensor Bottleneck::forward(torch::Tensor X) {
    auto identity = X;

    auto out = conv1->forward(torch::relu(X)*beta);
    out = torch::relu(out);
    out = conv2->forward(out);
    out = torch::relu(out);

    out = conv3->forward(out);

    if (!downsample.is_empty())
        identity = downsample->forward(X);

    out *= alpha;
    out += identity;
   // std::cout << "bottleneck-" << X.sizes() << "-" << out.sizes() << std::endl;
    return out;
}

torch::Tensor BasicBlock::forward(torch::Tensor x) {
    auto identity = x;

    auto out = conv1->forward(torch::relu(x) * beta);
    out = torch::relu(out);


    out = conv2->forward(out);


    if (!downsample.is_empty())
        identity = downsample->forward(x);

    out *= alpha;
    out += identity;
 //   std::cout << "basic-" << x.sizes() << "-" << out.sizes() << std::endl;
    return out;
}

} // namespace _resnetimpl

NFNet18Impl::NFNet18Impl(int64_t num_classes, bool zero_init_residual)
    : NFNetImpl({ 2, 2, 2, 2 }, num_classes, zero_init_residual) {}

NFNet34Impl::NFNet34Impl(int64_t num_classes, bool zero_init_residual)
    : NFNetImpl({ 3, 4, 6, 3 }, num_classes, zero_init_residual) {}

NFNet50Impl::NFNet50Impl(int64_t num_classes, bool zero_init_residual)
    : NFNetImpl({ 3, 4, 6, 3 }, num_classes, zero_init_residual) {}

NFNet101Impl::NFNet101Impl(int64_t num_classes, bool zero_init_residual)
    : NFNetImpl({ 3, 4, 23, 3 }, num_classes, zero_init_residual) {}

NFNet152Impl::NFNet152Impl(int64_t num_classes, bool zero_init_residual)
    : NFNetImpl({ 3, 8, 36, 3 }, num_classes, zero_init_residual) {}


} // namespace models
} // namespace vision
