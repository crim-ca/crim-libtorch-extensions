#pragma once

#include <torch/torch.h>

namespace torch {
namespace nn {


typedef std::function<torch::Tensor(torch::Tensor)> ActivationFunction;

torch::Tensor swish(torch::Tensor x);

torch::Tensor relu6(torch::Tensor x);


} // nn
} // torch
