#pragma once

#include <torch/torch.h>
#include "torchvision/macros.h"  // VISION_API

namespace torch {
namespace nn {


typedef std::function<torch::Tensor(torch::Tensor)> ActivationFunction;

torch::Tensor VISION_API swish(torch::Tensor x);

torch::Tensor relu6(torch::Tensor x);


} // nn
} // torch
