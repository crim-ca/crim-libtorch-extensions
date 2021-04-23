#include "stdafx.h"
#pragma hdrstop

#include "nn/modules/activation.h"


torch::Tensor swish(torch::Tensor x)
{
    return x * torch::sigmoid(x);
}

torch::Tensor relu6(torch::Tensor x)
{
    torch::nn::ReLU6 relu6(torch::nn::ReLU6Options().inplace(true));
    return relu6(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish", &swish, "Swish activation function");
    m.def("relu6", &relu6, "ReLU6 activation function");
}
