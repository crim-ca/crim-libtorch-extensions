#include "stdafx.h"
#pragma hdrstop

#include "nn/modules/activation.h"


namespace torch {
namespace nn {

torch::Tensor swish(torch::Tensor x)
{
    return x * torch::sigmoid(x);
}

torch::Tensor relu6(torch::Tensor x)
{
    torch::nn::ReLU6 relu6(torch::nn::ReLU6Options().inplace(true));
    return relu6(x);
}


} // nn
} // torch
