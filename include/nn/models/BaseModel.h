#pragma once

#include <torch/torch.h>

/**
 * @brief Base class that all models defined by this library should derive from.
 *
 * This class is mostly to provide a shared interface to all known models to facilitate switching between them.
 */
class IBaseModel
{
public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

/**
 * @brief Extension to the base model if it supports dynamic resize of its last layer.
 *
 * It is not mandatory for all models to support this feature.
 * If it is available though, they should provide this interface as well.
 */
class IResizableModel : IBaseModel
{
public:
    virtual void resizeLastLayer(size_t outputCount) = 0;
}
