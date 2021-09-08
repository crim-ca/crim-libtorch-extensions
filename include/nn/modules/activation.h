#pragma once

#include <torch/torch.h>
#include "cte_macros.h"  

namespace torch {
namespace nn {


typedef std::function<torch::Tensor(torch::Tensor)> ActivationFunction;

torch::Tensor CTE_API swish(torch::Tensor x);

torch::Tensor relu6(torch::Tensor x);


} // nn
} // torch
