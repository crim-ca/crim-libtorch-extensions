#pragma once
#include "BaseModel.h"
#include <algorithm>
#include <torch/torch.h>
#include <tuple>

/*
class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

*/
#include <random>
#include <string>

std::string random_string();

class Conv2dStaticSamePadding : public torch::nn::Conv2d
{
public:
    Conv2dStaticSamePadding() = default;

    Conv2dStaticSamePadding(torch::nn::Conv2dOptions o, int64_t image_size)
        : torch::nn::Conv2d(o), static_padding(nullptr)
    {
        //1
        //name = random_string();
        if (this->get()->options.stride().size() != 2) {
            torch::ExpandingArray<2Ui64>& strides = this->get()->options.stride();
            auto stridevals = *strides;
            stridevals[0] *= 2;
        }
        torch::ExpandingArray<2Ui64>& stridearr = this->get()->options.stride();
        auto stridevals = *stridearr;
        stride = stridevals[0];
        auto iw = image_size;
        auto ih = image_size;
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
            /*
             self.static_padding = nn.ZeroPad2d((pad_w - pad_w // 2, pad_w - pad_w // 2,
                                                pad_h - pad_h // 2, pad_h - pad_h // 2))
            */
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
            return F::conv2d(static_padding->forward(x),
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

    torch::nn::ZeroPad2d static_padding{
            nullptr}; //You are trying to default construct a module which has no default constructor...
    double stride;
    std::string name = "Conv2dStaticSamePadding";
};

struct BlockArgs
{
    /*
        'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
    */
    BlockArgs() {}
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
struct GlobalParams
{
    GlobalParams() = default;
    GlobalParams(double w, double d, int64_t res, double dor)
        : width_coefficient(w), depth_coefficient(d), image_size(res), dropout_rate(dor)
    {
    }
    GlobalParams(const GlobalParams&) = default;
    double batch_norm_momentum = 0.99;
    double batch_norm_epsilon = 0.001;
    double depth_coefficient = -1;
    double width_coefficient = -1;
    double dropout_rate = 0.2;
    double drop_connect_rate = 0.2;
    int depth_divisor = 8;
    int min_depth = -1;
    int64_t image_size;
};

struct MBConvBlockImpl : public torch::nn::Module
{
public:
    MBConvBlockImpl() = default;
    MBConvBlockImpl(BlockArgs block_args, GlobalParams globalargs);

    BlockArgs /*std::tuple*/ blockargs;
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

    virtual torch::Tensor forward(torch::Tensor x, double drop_connect_rate);
};
TORCH_MODULE(MBConvBlock);

struct EfficientNetV1Impl : torch::nn::Module
{
    //bool aux_logits, transform_input;
    EfficientNetV1Impl() = default;
    EfficientNetV1Impl(const GlobalParams& gp, size_t num_classes = 2);
    GlobalParams _gp;
    Conv2dStaticSamePadding *_conv_stem, *_conv_head;
    torch::nn::AdaptiveAvgPool2d _avg_pooling = nullptr;
    torch::nn::Dropout _dropout = nullptr;
    torch::nn::Linear _fc = nullptr;

    std::vector<MBConvBlock*> _blocks;
    torch::nn::BatchNorm2d _bn0 = nullptr, _bn1 = nullptr;

    torch::Tensor forward(torch::Tensor x);
    torch::Tensor extract_features(torch::Tensor x);
    std::vector<BlockArgs> blockargs = {BlockArgs(1, 3, 1, 1, 32, 16, 0.25, true),
                                        BlockArgs(2, 3, 2, 6, 16, 24, 0.25, true),
                                        BlockArgs(2, 5, 2, 6, 24, 40, 0.25, true),
                                        BlockArgs(3, 3, 2, 6, 40, 80, 0.25, true),
                                        BlockArgs(3, 5, 1, 6, 80, 112, 0.25, true),
                                        BlockArgs(4, 5, 2, 6, 112, 192, 0.25, true),
                                        BlockArgs(1, 3, 1, 6, 192, 320, 0.25, true)};
};

TORCH_MODULE(EfficientNetV1);

class EfficientNet : public BaseModel, public EfficientNetV1Impl
{
public:
    EfficientNet(GlobalParams gp, size_t nboutputs) : EfficientNetV1Impl(gp, nboutputs) {}
    virtual void resizeLastLayer(size_t outputCount) {}
    virtual torch::Tensor forward(torch::Tensor x)
    {
        return EfficientNetV1Impl::forward(x);
    }

protected:
    /*
        params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    */
    std::map<std::string, GlobalParams> params_dict = {{"b0", GlobalParams{1.0, 1.0, 224, 0.2}},
                                                       {"b1", GlobalParams{1.0, 1.1, 240, 0.2}},
                                                       {"b2", GlobalParams{1.1, 1.2, 260, 0.3}},
                                                       {"b3", GlobalParams{1.2, 1.4, 300, 0.3}},
                                                       {"b4", GlobalParams{1.4, 1.8, 380, 0.4}},
                                                       {"b5", GlobalParams{1.6, 2.2, 456, 0.4}},
                                                       {"b6", GlobalParams{1.8, 2.6, 528, 0.5}},
                                                       {"b7", GlobalParams{2.0, 3.1, 600, 0.5}}};
    /*   'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',*/
    //BlockArgs(int rep, int ksz, int strd, int expa, int inp, int outp, double se, bool skip)
};

class EfficientNetB0 : public EfficientNet
{
public:
    EfficientNetB0(size_t nboutputs) : EfficientNet(GlobalParams{1.0, 1.0, 224, 0.2}, nboutputs) {}
};
class EfficientNetB3 : public EfficientNet
{
public:
    EfficientNetB3(size_t nboutputs) : EfficientNet(GlobalParams{1.2, 1.4, 300, 0.3}, nboutputs) {}
};
class EfficientNetB2 : public EfficientNet
{
public:
    EfficientNetB2(size_t nboutputs) : EfficientNet(GlobalParams{1.1, 1.2, 260, 0.3}, nboutputs) {}
};