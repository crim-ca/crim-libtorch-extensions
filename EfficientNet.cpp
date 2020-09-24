#include "stdafx.h"
#include "models/EfficientNet.h"
//#define USE_RELU6 

std::string random_string()
{
    std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::shuffle(str.begin(), str.end(), generator);

    return str.substr(0, 32); // assumes 32 < number of characters in str
}

/*
def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output
*/

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

/*
def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

*/

int round_filters(int filters, GlobalParams p)
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
//
// if width_coefficient==10: expected values are 32 64 88 112 128 152 208
void test_round_filters()
{
    GlobalParams p(1, 1, 224, 0.2);
    p.width_coefficient = 10;
    auto a = {3, 6, 9, 11, 13, 15, 21};
    for (auto ai : a) {
        std::cout << round_filters(ai, p) << std::endl;
    }
}

/*
def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
*/
int round_repeats(int repeats, GlobalParams p)
{
    auto multiplier = p.depth_coefficient;
    if (multiplier < 0)
        return repeats;
    return int(std::ceil(multiplier * repeats));
}
/*class SwishImpl : public torch::autograd::Function<float>
{
};
*/


torch::Tensor swish(torch::Tensor x)
{
    return x * torch::sigmoid(x);
}

/*
 def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect


        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this would do nothing
        
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
       

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1,1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()*/


/*
class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

*/



MBConvBlockImpl::MBConvBlockImpl(BlockArgs block_args, GlobalParams globalargs, int64_t _imgsize_w, int64_t _imgsize_h)
    : /*_depthwise_conv(nullptr), _expand_conv(nullptr), _project_conv(nullptr),*/ _bn0(nullptr)
    , _bn1(nullptr)
    , _bn2(nullptr)
{
    blockargs = block_args;
    _bn_mom = 1 - globalargs.batch_norm_momentum;
    _bn_eps = globalargs.batch_norm_epsilon;
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
        /*
        Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this would do nothing
        */
        _expand_conv = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(inp, outp, 1 /*{1}*/).bias(false), imgsize_w,imgsize_h);
        _bn0 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outp).momentum(_bn_mom).eps(_bn_eps));
    }
    /*
    # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
    */
    auto k = block_args.kernel_size;
    auto s = block_args.stride;
    _depthwise_conv = new Conv2dStaticSamePadding(
            torch::nn::Conv2dOptions(outp, outp, k /*{k}*/).bias(false).stride(s).groups(outp), imgsize_w,imgsize_h);//imgsize);
    /*
     self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
    */
    imgsize_w = int(std::ceil(imgsize_w / (double)(s)));
    imgsize_h = int(std::ceil(imgsize_h / (double)(s)));

    _bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outp).momentum(_bn_mom).eps(_bn_eps));

    //         image_size = calculate_output_image_size(image_size, s)
    /* # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)*/
    if (has_se) {
        auto num_squeezed_channels = std::max(1, int(block_args.input_filters * block_args.se_ratio));
        _sereduce_conv = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(outp, num_squeezed_channels, 1), 1,1);
        _seexpand_conv = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(num_squeezed_channels, outp, 1), 1,1);
    }
    /*
       # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()
    */
    auto final_oup = block_args.output_filters;
    this->_project_conv =
            new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(outp, final_oup, 1 /*{1}*/).bias(false), imgsize_w,imgsize_w);
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
def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
*/

torch::Tensor MBConvBlockImpl::forward(torch::Tensor inputs, double drop_connect_rate)
{
   // std::cout << "@@"<<inputs.dtype() << std::endl;
    torch::nn::ReLU6 relu6(torch::nn::ReLU6Options().inplace(true));
    auto x = inputs;
    if (blockargs.expand_ratio != 1) {
        #ifdef USE_RELU6
        x = relu6(_bn0->forward(_expand_conv->forward(inputs)));
        #else
        x = swish(_bn0->forward(_expand_conv->forward(inputs)));
        #endif
      
    }
    auto xx = _depthwise_conv->forward(x);
    #ifdef USE_RELU6
    x = relu6(_bn1->forward(xx));
    #else
    x = swish(_bn1->forward(xx));
    #endif
    
    if (has_se) {
        x = _seexpand_conv->forward(_sereduce_conv->forward(x));
    }
    x = _bn2->forward(_project_conv->forward(x));

    auto inpf = blockargs.input_filters;
    auto outpf = blockargs.output_filters;
    if (id_skip && blockargs.stride == 1 && inpf == outpf) {
        if (drop_connect_rate > 0)
            x = drop_connect(x, drop_connect_rate, this->is_training());

        x += inputs;
    }
    
    return x;
}

/*
def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
*/
EfficientNetV1Impl::EfficientNetV1Impl(const GlobalParams& gp, size_t num_classes)
{
    /*
     # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

    */
    /*
    auto xxx =
            MBConvBlock(BlockArgs(1, 3, 1, 1, 32, 16, 0.25, true), GlobalParams{1.2, 1.4, 300, 0.3});

    */
    //auto t = torch::zeros({1, 3, 300, 300});
    //xxx->forward(t,0.2);
    //Conv2dStaticSamePadding s(torch::nn::Conv2dOptions(3, 24, 3), 300);
    //auto outt = s.forward(t);
    
    _gp = gp;

    auto out_channels = round_filters(32, gp);
    auto opti = torch::nn::Conv2dOptions(3, out_channels, 3).bias(false).stride(2);
    _conv_stem = new Conv2dStaticSamePadding(opti, gp.image_size_w, gp.image_size_h);
    _bn0 = torch::nn::BatchNorm2d(
            torch::nn::BatchNorm2dOptions(out_channels).momentum(gp.batch_norm_momentum).eps(gp.batch_norm_epsilon));
    /*
     # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
    */
    int lastoutpf = 0;
    int blkno = 0;
    auto imgsize_w = std::ceil(gp.image_size_w/2.0);
    auto imgsize_h = std::ceil(gp.image_size_h / 2.0);

    for (auto ba : blockargs) {  // 7 blocks: nb repeats=1,2,2,3,3,4,1
        ba.input_filters = round_filters(ba.input_filters, gp);
        ba.output_filters = round_filters(ba.output_filters, gp);
        ba.repeats = round_repeats(ba.repeats, gp);
        std::cout << "Layer " << blkno << ": " << ba.input_filters << "->" << ba.output_filters
                  << ", rep=" << ba.repeats << ", stride: " <<ba.stride <<", "<< imgsize_w << "x" << imgsize_h << std::endl;
        auto blk = new MBConvBlock(ba, gp,imgsize_w,imgsize_h);
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
            auto blk_rep = new MBConvBlock(ba, gp,imgsize_w,imgsize_h);
            std::cout << "  REP : " << ba.input_filters << "->" << ba.output_filters
                      << ", " << imgsize_w << "x" << imgsize_h << std::endl;

            register_module("mbconvblk_" + std::to_string(blkno) + "_" + std::to_string(i), *blk_rep);
            _blocks.push_back(blk_rep);
        }
        blkno++;
    }
    /*    # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        */
    auto in_channels = lastoutpf;
    out_channels = round_filters(1280, gp);
    _conv_head = new Conv2dStaticSamePadding(torch::nn::Conv2dOptions(in_channels, out_channels, 1 /*{1}*/).bias(false),
                                             imgsize_w,imgsize_h);
    _bn1 = torch::nn::BatchNorm2d(
            torch::nn::BatchNorm2dOptions(out_channels).momentum(gp.batch_norm_momentum).eps(gp.batch_norm_epsilon));
    /*
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
        */
    _avg_pooling = torch::nn::AdaptiveAvgPool2d(1);
    _dropout = torch::nn::Dropout(torch::nn::DropoutOptions().p(gp.dropout_rate).inplace(true));
    _fc = torch::nn::Linear(torch::nn::LinearOptions(out_channels, num_classes).bias(false));

    register_module(random_string(), *_conv_stem);
    register_module(random_string(), *_conv_head);

    register_module("avgpool", _avg_pooling);
    register_module("bn0", _bn0);
    register_module("bn1", _bn1);
    register_module("dropout", _dropout);
    register_module("fc", _fc);
}

/*
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x*/
/*   def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x*/
torch::Tensor EfficientNetV1Impl::forward(torch::Tensor inputs)
{

    auto bs = inputs.size(0);
    auto x = extract_features(inputs);
    x = _avg_pooling->forward(x);
    x = x.view({bs,1536});
    x = _dropout->forward(x);
    return _fc(x);
}

torch::Tensor EfficientNetV1Impl::extract_features(torch::Tensor inputs)
{
    torch::nn::ReLU6 relu6(torch::nn::ReLU6Options().inplace(true));
    //stem
    auto y = _conv_stem->forward(inputs);
    //std::cout << "EF: " << inputs.dtype() << std::endl;
//    std::cout << "EF: " << y.dtype() << std::endl;
    #ifdef USE_RELU6
    auto x = relu6(_bn0->forward(y));
    #else
    auto x = swish(_bn0->forward(y));
    #endif
    
  //  std::cout << "EF: " << x.dtype() << std::endl;
    //blocks
    int idx = 0;
    for (auto block : _blocks) {
        auto drop_connect_rate = _gp.drop_connect_rate;
        if (drop_connect_rate > 0)
            drop_connect_rate *= float(idx) / _blocks.size();
        //block->get()->pretty_print(std::cout);
        x = block->get()->forward(x, drop_connect_rate);
    //    std::cout << "    EF: " << x.dtype() << std::endl;
        // Testing use of half precision...
       // x = x.to(torch::kHalf);


        idx++;
    }
    // head
    #ifdef USE_RELU6
    return relu6(_bn1->forward(_conv_head->forward(x)));
    #else
    return swish(_bn1->forward(_conv_head->forward(x)));
    #endif
    
}
