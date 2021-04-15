#include "stdafx.h"
#pragma once

class BaseModel
{
public:
    virtual void resizeLastLayer(size_t outputCount) = 0;
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};
