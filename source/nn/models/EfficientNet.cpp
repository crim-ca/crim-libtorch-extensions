#include "stdafx.h"
#pragma hdrstop

#include "nn/models/EfficientNet.h"

namespace vision {
namespace models {

// FIXME: move to utils
// USed to generate unique module names
std::string random_string()
{
    std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(str.begin(), str.end(), generator);

    return str.substr(0, 32); // assumes 32 < number of characters in str
}

// Drops some weights following some probability p (suggested value: 0.2)
// Function returns immediately if training flag is false
// https://stats.stackexchange.com/questions/201569/what-is-the-difference-between-dropout-and-drop-connect
torch::Tensor drop_connect(torch::Tensor inputs, double p, bool training)
{
    if (!training)
        return inputs;
    auto batchsize = inputs.size(0);
    auto keep_prob = 1 - p;
    torch::Tensor random_tensor = keep_prob * torch::ones({batchsize, 1, 1, 1});
    random_tensor += torch::rand({batchsize, 1, 1, 1});
    random_tensor = random_tensor.to(inputs.device());
    auto binary_tensor = torch::floor(random_tensor);
    auto output = inputs / keep_prob * binary_tensor;

    return output;
}


int round_filters(int filters, EfficientNetOptions p)
{
    auto multiplier = p.width_coefficient;
    if (multiplier < 0)
        return filters;
    filters *= multiplier;
    auto divisor = p.depth_divisor;
    auto min_depth = p.min_depth;
    if (min_depth < 0)
        min_depth = divisor;
    auto new_filters = std::max(min_depth, (int(filters + divisor / 2) / divisor) * divisor);
    if (new_filters < 9 * filters / 10)
        new_filters += divisor;
    return new_filters;
}

// FIXME: move to tests
// if width_coefficient==10: expected values are 32 64 88 112 128 152 208
void test_round_filters()
{
    EfficientNetOptions p(1, 1, 224, 0.2);
    p.width_coefficient = 10;
    auto a = {3, 6, 9, 11, 13, 15, 21};
    for (auto ai : a) {
        //std::cout << round_filters(ai, p) << std::endl;
    }
}

// FIXME: move to utils
// Round number of filters based on depth multiplier.
int round_repeats(int repeats, EfficientNetOptions p)
{
    auto multiplier = p.depth_coefficient;
    if (multiplier < 0)
        return repeats;
    return int(std::ceil(multiplier * repeats));
}

/*
Mobile Inverted Residual Bottleneck Block.

References:
    [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
    [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
    [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
*/
MBConvBlockImpl::MBConvBlockImpl(
    BlockArgs block_args,
    EfficientNetOptions params,
    int64_t _imgsize_w,
    int64_t _imgsize_h)
    : /*_depthwise_conv(nullptr), _expand_conv(nullptr), _project_conv(nullptr),*/ _bn0(nullptr)
    , _bn1(nullptr)
    , _bn2(nullptr)
    , _params(params)
    , _block_args(block_args)
{
    _bn_mom = 1 - params.batch_norm_momentum;
    _bn_eps = params.batch_norm_epsilon;
    has_se = block_args.se_ratio > 0 && block_args.se_ratio <= 1;
    id_skip = block_args.id_skip;
    auto imgsize_w = _imgsize_w;
    auto imgsize_h = _imgsize_h;
    // Expansion phase
    auto inp = block_args.input_filters;
    auto outp = block_args.input_filters * block_args.expand_ratio;
    _expand_conv = nullptr;
    _bn0 = nullptr;
    if (block_args.expand_ratio != 1) {
        _expand_conv = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(inp, outp, 1 /*{1}*/).bias(false), imgsize_w, imgsize_h);
        _bn0 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outp).momentum(_bn_mom).eps(_bn_eps));
    }
    auto k = block_args.kernel_size;
    auto s = block_args.stride;
    _depthwise_conv = new Conv2dStaticSamePadding(
        torch::nn::Conv2dOptions(outp, outp, k /*{k}*/).bias(false).stride(s).groups(outp), imgsize_w, imgsize_h
    );

    imgsize_w = int(std::ceil(imgsize_w / (double)(s)));
    imgsize_h = int(std::ceil(imgsize_h / (double)(s)));

    _bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outp).momentum(_bn_mom).eps(_bn_eps));

    /* # Squeeze and Excitation layer, if desired */
    if (has_se) {
        auto num_squeezed_channels = std::max(1, int(block_args.input_filters * block_args.se_ratio));
        _sereduce_conv = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(outp, num_squeezed_channels, 1), 1, 1);
        _seexpand_conv = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(num_squeezed_channels, outp, 1), 1, 1);
    }
    /*
       # Output phase
    */
    auto final_oup = block_args.output_filters;
    this->_project_conv = new Conv2dStaticSamePadding(
        torch::nn::Conv2dOptions(outp, final_oup, 1 /*{1}*/).bias(false), imgsize_w,imgsize_w
    );

    this->_bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(final_oup).momentum(_bn_mom).eps(_bn_eps));
    register_module("dw_" + _depthwise_conv->name, *_depthwise_conv);
    if (_expand_conv)
        register_module("expand_" + _expand_conv->name, *_expand_conv);
    register_module("proj_" + _project_conv->name, *_project_conv);
    if (_sereduce_conv)
        register_module("sereduce_" + _sereduce_conv->name, *_sereduce_conv);
    if (_seexpand_conv)
        register_module("seexpand_" + _seexpand_conv->name, *_seexpand_conv);
    if (_bn0)
        register_module("bn0", _bn0);
    register_module("bn1", _bn1);
    register_module("bn2", _bn2);
}

/*
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """


*/

torch::Tensor MBConvBlockImpl::forward(torch::Tensor inputs, double drop_connect_rate)
{
    auto x = inputs;
    if (_block_args.expand_ratio != 1) {
        x = _expand_conv->forward(x);    // missing?
        x = _bn0->forward(x);           // missing?
        x = _params.activation(x);
    }
    auto xx = _depthwise_conv->forward(x);
    x = _params.activation(xx);

    if (has_se) {
        x = _seexpand_conv->forward(_sereduce_conv->forward(x));
    }
    x = _bn2->forward(_project_conv->forward(x));

    auto inpf = _block_args.input_filters;
    auto outpf = _block_args.output_filters;
    if (id_skip && _block_args.stride == 1 && inpf == outpf) {
        if (drop_connect_rate > 0)
            x = drop_connect(x, drop_connect_rate, this->is_training());
        x += inputs;
    }

    return x;
}

EfficientNetV1Impl::EfficientNetV1Impl(const EfficientNetOptions& params, size_t num_classes)
    : _params(params)
{
    // stem
    auto out_channels = round_filters(32, _params);
    auto opti = torch::nn::Conv2dOptions(3, out_channels, 3).bias(false).stride(2);
    _conv_stem = new Conv2dStaticSamePadding(opti, _params.image_size_w, _params.image_size_h);
    _bn0 = torch::nn::BatchNorm2d(
        torch::nn::BatchNorm2dOptions(out_channels).momentum(_params.batch_norm_momentum).eps(_params.batch_norm_epsilon)
    );

    int lastoutpf = 0;
    int blkno = 0;
    auto imgsize_w = std::ceil(_params.image_size_w / 2.0);
    auto imgsize_h = std::ceil(_params.image_size_h / 2.0);

    for (auto ba : blockargs) {  // 7 blocks: nb repeats=1,2,2,3,3,4,1
        ba.input_filters = round_filters(ba.input_filters, _params);
        ba.output_filters = round_filters(ba.output_filters, _params);
        ba.repeats = round_repeats(ba.repeats, _params);

       // std::cout << "Layer " << blkno << ": " << ba.input_filters << "->" << ba.output_filters
       //           << ", rep=" << ba.repeats << ", stride: " <<ba.stride <<", "<< imgsize_w << "x" << imgsize_h << std::endl;

        auto blk = new MBConvBlock(ba, _params, imgsize_w, imgsize_h);
        register_module("mbconvblk_" + std::to_string(blkno), *blk);
        _blocks.push_back(blk);
        lastoutpf = ba.output_filters;
        imgsize_w = std::ceil(imgsize_w / (double)ba.stride);
        imgsize_h = std::ceil(imgsize_h / (double)ba.stride);
        if (ba.repeats > 1) {
            ba.input_filters = ba.output_filters;
            ba.stride = 1;
        }

        for (auto i = 0; i < ba.repeats - 1; i++) {
            auto blk_rep = new MBConvBlock(ba, _params, imgsize_w, imgsize_h);
        //    std::cout << "  REP : " << ba.input_filters << "->" << ba.output_filters
        //              << ", " << imgsize_w << "x" << imgsize_h << std::endl;

            register_module("mbconvblk_" + std::to_string(blkno) + "_" + std::to_string(i), *blk_rep);
            _blocks.push_back(blk_rep);
        }
        blkno++;
    }

    // Head
    auto in_channels = lastoutpf;
    out_channels = round_filters(1280, _params);
    _conv_head = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(in_channels, out_channels, 1 /*{1}*/).bias(false),
                                             imgsize_w,imgsize_h);
    _bn1 = torch::nn::BatchNorm2d(
        torch::nn::BatchNorm2dOptions(out_channels).momentum(_params.batch_norm_momentum).eps(_params.batch_norm_epsilon)
    );

    // Final linear layer
    _avg_pooling = torch::nn::AdaptiveAvgPool2d(1);
    _dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(_params.dropout_rate).inplace(true));
    _fc = torch::nn::Linear(torch::nn::LinearOptions(out_channels, num_classes).bias(false));

    register_module("conv_stem", *_conv_stem);
    register_module("conv_head", *_conv_head);

    register_module("avgpool", _avg_pooling);
    register_module("bn0", _bn0);
    register_module("bn1", _bn1);
    register_module("dropout", _dropout);
    register_module("fc", _fc);
}

// Calls extract_features to extract features, applies final linear layer, and returns logits.
torch::Tensor EfficientNetV1Impl::forward(torch::Tensor inputs)
{

    auto bs = inputs.size(0);
    auto x = extract_features(inputs);
    x = _avg_pooling->forward(x);
  //  std::cout << "tensor size before fc: " << x.sizes() << std::endl;
    x = x.view({bs,x.sizes()[1]/*1536*/});
    x = _dropout->forward(x);
    return _fc(x);
}

// Returns output of the final convolution layer
torch::Tensor EfficientNetV1Impl::extract_features(torch::Tensor inputs)
{
    //stem
    auto y = _conv_stem->forward(inputs);
    auto x = _params.activation(y);

    //blocks
    int idx = 0;
    for (auto block : _blocks) {
        auto drop_connect_rate = _params.drop_connect_rate;
        if (drop_connect_rate > 0)
            drop_connect_rate *= float(idx) / _blocks.size();
        x = block->get()->forward(x, drop_connect_rate);
        idx++;
    }

    // head
    return _params.activation(_bn1->forward(_conv_head->forward(x)));
}


} // namespace models
} // namespace vision
