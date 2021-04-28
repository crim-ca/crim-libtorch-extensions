#include <stdafx.h>
#include <torch/torch.h>
/*
class ScaledStdConv2d(nn.Conv2d) :
    """Conv2d layer with Scaled Weight Standardization.
    Paper : `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
    https://arxiv.org/abs/2101.08692

Adapted from timm : https://github.com/rwightman/pytorch-image-models/blob/4ea593196414684d2074cbb81d762f3847738484/timm/models/layers/std_conv.py
"""

def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1,
    bias = True, gain = True, gamma = 1.0, eps = 1e-5, use_layernorm = False) :
    super().__init__(
        in_channels, out_channels, kernel_size, stride = stride,
        padding = padding, dilation = dilation, groups = groups, bias = bias)
    self.gain = nn.Parameter(torch.ones(
        self.out_channels, 1, 1, 1)) if gain else None
    # gamma * 1 / sqrt(fan - in)
    self.scale = gamma * self.weight[0].numel() * *-0.5
    self.eps = eps * *2 if use_layernorm else eps
    # experimental, slightly faster / less GPU memory use
    self.use_layernorm = use_layernorm

    def get_weight(self) :
    if self.use_layernorm :
        weight = self.scale * \
        F.layer_norm(self.weight, self.weight.shape[1:], eps = self.eps)
    else :
        mean = torch.mean(
            self.weight, dim = [1, 2, 3], keepdim = True)
        std = torch.std(
            self.weight, dim = [1, 2, 3], keepdim = True, unbiased = False)
        weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
weight = weight * self.gain
return weight

def forward(self, x) :
    return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
*/

class ScaledStdConv2d : public torch::nn::Conv2d
{
public:
    ScaledStdConv2d() = default;

    ScaledStdConv2d(torch::nn::Conv2dOptions o, bool _isgain=true, double _gamma=1.0, double _eps=1e-5, bool _uselayernorm=false)
        : torch::nn::Conv2d(o), uselayernorm(_uselayernorm), isgain(_isgain)        
    {
        if (isgain)
            gain = this->impl_->register_parameter("gain", torch::ones({ o.out_channels(),1,1,1 }));   
        if (uselayernorm)
            eps = _eps * _eps;
        else eps = _eps;
        scale = _gamma * this->impl_->weight[0].numel();
        
        
    }
    torch::Tensor get_weight() {
        namespace F = torch::nn::functional;
        torch::Tensor weight;
        if (uselayernorm) {
            auto sz = this->get()->weight.sizes();
            sz = sz.slice(1);    
            F::LayerNormFuncOptions o(sz.vec());
            o.eps(eps);
            weight = scale * F::layer_norm(this->get()->weight, o);
        }
        else {
            auto mean = torch::mean(this->get()->weight, { 1,2,3 }, true);
            auto std = torch::std(this->get()->weight, { 1,2,3 }, true);
            weight = scale * (this->get()->weight - mean) / (std + eps);
            if(isgain)
              weight = weight * gain;
        }
        return weight;
    }
    torch::Tensor forward(torch::Tensor x)
    {
        torch::nn::functional::Conv2dFuncOptions opt;
        
        return torch::nn::functional::conv2d(x,
                                                  this->get()->weight,
                                                  opt.bias(this->get()->bias)
                                                          .stride(this->get()->options.stride())
                                                          .groups(this->get()->options.groups())
                                                          .padding(this->get()->options.padding())
                                                          );
                                                          
    }

    //torch::nn::ZeroPad2d static_padding{
    //        nullptr }; //You are trying to default construct a module which has no default constructor...
    bool isgain;
    torch::Tensor gain;
    std::string name = "ScaledStdConv2d";
    double scale;
    bool uselayernorm;
    double eps;
};
