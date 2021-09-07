#pragma once
/**
 * @brief C++ adaptation of lukemelas' pytorch implementation of EfficientNet
 * @link https://github.com/lukemelas/EfficientNet-PyTorch @endlink
 */

#include <algorithm>
#include <tuple>
#include <random>
#include <string>

#include "nn/models/BaseModel.h"
#include "nn/modules/activation.h"
#include <torch/torch.h>
#include "cte_macros.h" 

namespace vision {
namespace models {

class  Conv2dStaticSamePadding : public torch::nn::Conv2d
{
public:
    Conv2dStaticSamePadding() = default;

    Conv2dStaticSamePadding(torch::nn::Conv2dOptions o, int64_t image_size_w, int64_t image_size_h)
        : torch::nn::Conv2d(o), static_padding(nullptr)
    {
        if (this->get()->options.stride().size() != 2) {
            torch::ExpandingArray<2ull>& strides = this->get()->options.stride();
            auto stridevals = *strides;
            stridevals[0] *= 2;
        }
        torch::ExpandingArray<2ull>& stridearr = this->get()->options.stride();
        auto stridevals = *stridearr;
        stride = stridevals[0];
        auto iw = image_size_w;
        auto ih = image_size_h;
        auto szlen = this->get()->weight.sizes().size();
        auto kh = this->get()->weight.sizes()[szlen-2];
        auto kw = this->get()->weight.sizes()[szlen - 1];

        //this->get()->weight.sizes()[1];
        auto oh = std::ceil((double)ih / stride);
        auto ow = std::ceil((double)iw / stride);
        auto dilation = *this->get()->options.dilation();
        auto pad_h = std::max<int>((oh - 1) * stride + (kh - 1) * dilation[0] + 1 - ih, 0);
        auto pad_w = std::max<int>((ow - 1) * stride + (kw - 1) * dilation[1] + 1 - iw, 0);
        if (pad_h > 0 || pad_w > 0) {
            //(padding_left, padding_right, padding_top, padding_bottom).
            static_padding = torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions(
                    {pad_w - (int)(pad_w / 2), pad_w - (int)(pad_w / 2), pad_h - (int)(pad_h / 2), pad_h - (int)(pad_h / 2)}));
        }
        else
            static_padding = nullptr;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::nn::functional::Conv2dFuncOptions opt;
        //opt.stride = stride;
        //opt.dilation = this->get()->options.dilation;
        //opt.padding = this->get()->options.padding;
        //opt.groups = this->get()->options.groups;
        namespace F = torch::nn::functional;
        /* auto conv = torch::nn::functional::conv2d(x,
                                                  this->get()->weight,
                                                  opt.bias(this->get()->bias)
                                                          .stride(stride)
                                                          .groups(this->get()->options.groups())
                                                          .padding(this->get()->options.padding())
                                                          );
                                                          */
        if (static_padding) 
            return  F::conv2d(static_padding->forward(x),
                this->get()->weight,
                F::Conv2dFuncOptions()
                .bias(this->get()->bias)
                .stride(stride)
                .groups(this->get()->options.groups())
                .padding(this->get()->options.padding()));

        return F::conv2d(x,
                         this->get()->weight,
                         F::Conv2dFuncOptions()
                                 .bias(this->get()->bias)
                                 .stride(stride)
                                 .groups(this->get()->options.groups())
                                 .padding(this->get()->options.padding()));
    }

    torch::nn::ZeroPad2d static_padding = nullptr; /*{
            nullptr}; //You are trying to default construct a module which has no default constructor...*/
    double stride;
    std::string name = "Conv2dStaticSamePadding";
};

struct  BlockArgs
{
    BlockArgs() = default;
    BlockArgs(int rep, int ksz, int strd, int expa, int inp, int outp, double se, bool skip)
        : repeats(rep)
        , kernel_size(ksz)
        , stride(strd)
        , expand_ratio(expa)
        , input_filters(inp)
        , output_filters(outp)
        , se_ratio(se)
        , id_skip(skip)
    {
    }
    int expand_ratio; // Ex. 1 or 6
    double se_ratio;
    bool id_skip;
    int repeats;
    int input_filters;  // Ex. 16, 24, 32, 40, 112, 192...
    int output_filters; // Ex. 16, 24, 40, 112, 192, 320...
    int kernel_size;
    int stride;
};

struct CTE_API EfficientNetOptions
{
public:
    EfficientNetOptions() = default;
    EfficientNetOptions(double _width_coefficient,
                       double _depth_coefficient,
                       int64_t _image_size,
                       double _dropout_rate,
                       // above params order remain for backward compatibility
                       // below with same defaults to make them available as needed but still optional
                       double _drop_connect_rate = 0.2,
                       double _batch_norm_momentum = 0.99,
                       double _batch_norm_epsilon = 0.001,
                       int _depth_divisor = 8,
                       int _min_depth = -1/*,
                       const ActivationFunction& _activation = swish*/)
        : width_coefficient(_width_coefficient)
        , depth_coefficient(_depth_coefficient)
        , image_size_w(_image_size)
        , image_size_h(_image_size)
        , dropout_rate(_dropout_rate)
        , drop_connect_rate(_drop_connect_rate)
        , batch_norm_momentum(_batch_norm_momentum)
        , batch_norm_epsilon(_batch_norm_epsilon)
        , depth_divisor(_depth_divisor)
        , min_depth(_min_depth)
        /*, activation(_activation)*/
    {}

    EfficientNetOptions(const EfficientNetOptions&) = default;

    void image_size(int64_t _image_size) { image_size_w = _image_size; image_size_h = image_size_h; }

    double width_coefficient = -1;
    double depth_coefficient = -1;
    double dropout_rate = 0.2;
    int64_t image_size_w, image_size_h;
    torch::nn::ActivationFunction activation = torch::nn::swish;
    double drop_connect_rate = 0.2;
    double batch_norm_momentum = 0.99;
    double batch_norm_epsilon = 0.001;
    int depth_divisor = 8;
    int min_depth = -1;
};

struct  MBConvBlockImpl : public torch::nn::Module
{
public:
    MBConvBlockImpl() = default;
    MBConvBlockImpl(
        BlockArgs block_args,
        EfficientNetOptions params,
        int64_t imgsize_w,
        int64_t imgsize_h);

    virtual torch::Tensor forward(torch::Tensor x, double drop_connect_rate);

private:
    BlockArgs _block_args;
    EfficientNetOptions _params;
    double _bn_mom;
    double _bn_eps;
    bool has_se = false;
    bool id_skip;
    Conv2dStaticSamePadding* _depthwise_conv = nullptr;
    Conv2dStaticSamePadding* _project_conv = nullptr;
    Conv2dStaticSamePadding* _expand_conv = nullptr;
    Conv2dStaticSamePadding* _sereduce_conv = nullptr;
    Conv2dStaticSamePadding* _seexpand_conv = nullptr;
    torch::nn::BatchNorm2d _bn0 = nullptr;
    torch::nn::BatchNorm2d _bn1 = nullptr;
    torch::nn::BatchNorm2d _bn2 = nullptr;
};
TORCH_MODULE(MBConvBlock);

struct CTE_API EfficientNetV1Impl : torch::nn::Module
{
    //bool aux_logits, transform_input;
    EfficientNetV1Impl() = default;
    EfficientNetV1Impl(const EfficientNetOptions& params, size_t num_classes = 2);
    EfficientNetOptions _params;
    Conv2dStaticSamePadding *_conv_stem=nullptr, *_conv_head=nullptr;
    torch::nn::AdaptiveAvgPool2d _avg_pooling = nullptr;
    torch::nn::Dropout _dropout = nullptr;
    torch::nn::Linear _fc = nullptr;

    std::vector<MBConvBlock*> _blocks;
    torch::nn::BatchNorm2d _bn0 = nullptr, _bn1 = nullptr;

    torch::Tensor forward(torch::Tensor x);
    torch::Tensor extract_features(torch::Tensor x);
    // https://github.com/lukemelas/EfficientNet-PyTorch/issues/13
    std::vector<BlockArgs> blockargs = {BlockArgs(1, 3, 1, 1, 32, 16, 0.25, true),
                                        BlockArgs(2, 3, 2, 6, 16, 24, 0.25, true),
                                        BlockArgs(2, 5, 2, 6, 24, 40, 0.25, true),
                                        BlockArgs(3, 3, 2, 6, 40, 80, 0.25, true),
                                        BlockArgs(3, 5, 1, 6, 80, 112, 0.25, true),
                                        BlockArgs(4, 5, 2, 6, 112, 192, 0.25, true),
                                        BlockArgs(1, 3, 1, 6, 192, 320, 0.25, true)};
};

TORCH_MODULE(EfficientNetV1);

class  EfficientNet : /*public IResizableModel, public IBaseModel, */ public EfficientNetV1Impl
{
public:
    EfficientNet(
        EfficientNetOptions params,
        size_t nboutputs
    ) : EfficientNetV1Impl(params, nboutputs) {}
    /*virtual void resizeLastLayer(size_t outputCount) {}*/
    virtual torch::Tensor forward(torch::Tensor x)
    {
        return EfficientNetV1Impl::forward(x);
    }

    EfficientNet(const EfficientNet&) = default;
};

class  EfficientNetB0 : public EfficientNet
{
public:
    EfficientNetB0(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.0, 1.0, 224, 0.2}, nboutputs) {}
};
class  EfficientNetB1 : public EfficientNet
{
public:
    EfficientNetB1(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.0, 1.1, 240, 0.2}, nboutputs) {}
};
class  EfficientNetB2 : public EfficientNet
{
public:
    EfficientNetB2(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.1, 1.2, 260, 0.3}, nboutputs) {}
};
class  EfficientNetB3 : public EfficientNet
{
public:
    EfficientNetB3(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.2, 1.4, 300, 0.3}, nboutputs) {}
};
class  EfficientNetB4 : public EfficientNet
{
public:
    EfficientNetB4(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.4, 1.8, 380, 0.4}, nboutputs) {}
};
class  EfficientNetB5 : public EfficientNet
{
public:
    EfficientNetB5(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.6, 2.2, 456, 0.4}, nboutputs) {}
};
class  EfficientNetB6 : public EfficientNet
{
public:
    EfficientNetB6(size_t nboutputs) : EfficientNet(EfficientNetOptions{1.8, 2.6, 528, 0.5}, nboutputs) {}
};
class  EfficientNetB7 : public EfficientNet
{
public:
    EfficientNetB7(size_t nboutputs) : EfficientNet(EfficientNetOptions{2.0, 3.1, 600, 0.5}, nboutputs) {}
};


} // namespace models
} // namespace vision
