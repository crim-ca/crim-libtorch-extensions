#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch
{
namespace serialize
{
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch
{
namespace optim
{
#if 0
struct  SGDAGCOptions
{

    /* implicit */ SGDAGCOptions(double learning_rate);
    TORCH_ARG(double, learning_rate);
    TORCH_ARG(double, momentum) = 0;
    TORCH_ARG(double, dampening) = 0;
    TORCH_ARG(double, weight_decay) = 0;
    TORCH_ARG(bool, nesterov) = false;
    TORCH_ARG(double, clipping) = 1e-2;
    TORCH_ARG(double, eps) = 1e-5;
#endif
    
    struct  SGDAGCOptions : public OptimizerCloneableOptions<SGDAGCOptions> {
        SGDAGCOptions(double lr);
        TORCH_ARG(double, lr);
        TORCH_ARG(double, momentum) = 0;
        TORCH_ARG(double, dampening) = 0;
        TORCH_ARG(double, weight_decay) = 0;
        TORCH_ARG(bool, nesterov) = false;
        TORCH_ARG(double, clipping) = 0.1; // 1e-2;
        TORCH_ARG(double, eps) = 1e-5;
    public:
        void serialize(torch::serialize::InputArchive& archive) override;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        TORCH_API friend bool operator==(const torch::optim::SGDOptions& lhs, const torch::optim::SGDOptions& rhs);
        ~SGDAGCOptions() = default;
    };


class  SGDAGC : public Optimizer
{
public:
    explicit SGDAGC(std::vector<OptimizerParamGroup> param_groups,
        SGDAGCOptions defaults) : Optimizer(std::move(param_groups), std::make_unique<SGDAGCOptions>(defaults)) {
        TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
        TORCH_CHECK(defaults.momentum() >= 0, "Invalid momentum value: ", defaults.momentum());
        TORCH_CHECK(defaults.weight_decay() >= 0, "Invalid weight_decay value: ", defaults.weight_decay());
        TORCH_CHECK(!defaults.nesterov() || (defaults.momentum() > 0 && defaults.dampening() == 0), "Nesterov momentum requires a momentum and zero dampening");
    }

    explicit SGDAGC(std::vector<Tensor> params,
        SGDAGCOptions defaults) : SGDAGC({ std::move(OptimizerParamGroup(params)) }, defaults) {}

    torch::Tensor step(LossClosure closure = nullptr) override;

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

private:
    template <typename Self, typename Archive>
    static void serialize(Self& self, Archive& archive) {
        _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SGD);
    }
#if 0
    template <typename ParameterContainer>
    explicit SGDAGC(ParameterContainer&& parameters, const SGDAGCOptions& options_)
        : Optimizer(std::forward<ParameterContainer>(parameters)), options(options_)
    {
    }


    Tensor step(LossClosure closure) override;

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;
    int64_t iteration() const;

    SGDAGCOptions options;

    std::vector<Tensor> momentum_buffers;

private:
    //SGDAGC() : Optimizer(), options(0) {}

    /// Counts how often `step()` is called, for dampening.
    int64_t iteration_{0};
#endif
};
} // namespace optim
} // namespace torch