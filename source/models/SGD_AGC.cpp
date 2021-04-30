
/*
WARNING: using libtorch 1.4.1 codebase...
*/
#include <stdafx.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>
#include "models/SGD_AGC.h"


namespace torch
{
namespace optim
{

    torch::Tensor unitwise_norm(torch::Tensor x) {
        bool keepdim;
        bool dim_is_0=true;

        if (x.ndimension() <= 1) {
            keepdim = false;
        }
        else if (x.ndimension() <= 3) {
            keepdim = true;
        }
        else if (x.ndimension() == 4) {
            keepdim=true;
            dim_is_0 = false;
        }
        if (dim_is_0)
            return torch::sqrt(torch::sum(x * x, {0}, keepdim));

        return torch::sqrt(torch::sum(x * x, {1,2,3}, keepdim));
    }
SGDAGCOptions::SGDAGCOptions(double learning_rate) :  lr_(learning_rate) {}
bool operator==(const SGDAGCOptions& lhs, const SGDAGCOptions& rhs) {
    return (lhs.lr() == rhs.lr()) &&
        (lhs.momentum() == rhs.momentum()) &&
        (lhs.dampening() == rhs.dampening()) &&
        (lhs.weight_decay() == rhs.weight_decay()) &&
        (lhs.nesterov() == rhs.nesterov());
}

void SGDAGCOptions::serialize(torch::serialize::OutputArchive& archive) const {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(dampening);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(nesterov);
}

void SGDAGCOptions::serialize(torch::serialize::InputArchive& archive) {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, dampening);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, nesterov);
}


Tensor SGDAGC::step(LossClosure closure)
{
    NoGradGuard no_grad;
    Tensor loss = {};
    if (closure != nullptr) {
        at::AutoGradMode enable_grad(true);
        loss = closure();
    }
#if 0
    for (size_t i = 0; i < this->parameters().size(); ++i) {
        Tensor p = this->parameters().at(i);

        if (!p.grad().defined()) {
            continue;
        }
        auto param_norm = torch::max(unitwise_norm(p.detach()), torch::tensor(options.eps()).to(p.device()));
        auto grad_norm = unitwise_norm(p.grad().detach());
        auto max_norm = param_norm * options.clipping();
        auto trigger = grad_norm > max_norm;
        auto clipped_grad = p.grad() * (max_norm / torch::max(grad_norm, torch::tensor(1e-6).to(grad_norm.device())));
        p.grad().detach().copy_(torch::where(trigger, clipped_grad, p.grad())); // inplace
    }
#endif
    for (auto& group : param_groups_) {
        auto& options = static_cast<SGDAGCOptions&>(group.options());
        auto weight_decay = options.weight_decay();
        auto momentum = options.momentum();
        auto dampening = options.dampening();
        auto nesterov = options.nesterov();
        auto eps = options.eps();
        auto clipping = options.clipping();

        for (auto& p : group.params()) {
            if (!p.grad().defined()) {
                continue;
            }

            //std::cout << p << std::endl;
            auto param_norm = torch::max(unitwise_norm(p.detach()), torch::tensor(eps).to(p.device()));
            
            auto grad_norm = unitwise_norm(p.grad().detach());
            auto max_norm = param_norm * clipping;
            auto trigger = grad_norm > max_norm;
            auto w = (max_norm / torch::max(grad_norm, torch::tensor(1e-6).to(grad_norm.device())));
            auto clipped_grad = p.grad() * (max_norm / torch::max(grad_norm, torch::tensor(1e-6).to(grad_norm.device())));
            p.grad().detach().copy_(torch::where(trigger, clipped_grad, p.grad())); // inplace
#if 0
            //  Test to record clipping weight to disk
            // https://discuss.pytorch.org/t/iterating-over-tensor-in-c/60333/2
            if (trigger.is_contiguous() && max_norm.is_contiguous() && grad_norm.is_contiguous()) {
                bool* ptr_trig = (bool*)trigger.to(torch::kCPU).data_ptr();
                float* ptr_max = (float*)max_norm.to(torch::kCPU).data_ptr();
                float* ptr_grad = (float*)grad_norm.to(torch::kCPU).data_ptr();
                float* ptr_w = (float*)w.to(torch::kCPU).data_ptr();
                for (int ii = 0; ii < trigger.numel(); ii++)
                {
                    if(*ptr_trig++)
                      outweight << *ptr_w<<std::endl;
                    ptr_w++;
                }

            }
#endif            
            
        }


        for (auto& p : group.params()) {
            if (!p.grad().defined()) {
                continue;
            }
            auto d_p = p.grad().data();
            if (weight_decay != 0) {
                d_p = d_p.add(p.data(), weight_decay);
            }
            if (momentum != 0) {
                Tensor buf;
                auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
                if (param_state == state_.end()) {
                    buf = torch::clone(d_p).detach();
                    auto state = std::make_unique<SGDParamState>();
                    state->momentum_buffer(buf);
                    state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
                }
                else {
                    buf = static_cast<SGDParamState&>(*param_state->second).momentum_buffer();
                    buf.mul_(momentum).add_(d_p, 1 - dampening);
                }
                if (nesterov) {
                    d_p = d_p.add(buf, momentum);
                }
                else {
                    d_p = buf;
                }
            }
            p.data().add_(d_p, -1 * options.lr());
        }
    }
#if 0
    for (size_t i = 0; i < this->parameters().size(); ++i) {
        Tensor p = this->parameters().at(i);

        if (!p.grad().defined()) {
            continue;
        }

        auto update = p.grad();

        if (options.weight_decay() > 0) {
            NoGradGuard guard;
            update += options.weight_decay() * p;
        }

        if (options.momentum() != 0) {
            Tensor buf;
            auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            if (param_state == state_.end()) {
                buf = torch::clone(update).detach();
                auto state = std::make_unique<SGDParamState>();
                state->momentum_buffer(buf);
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
            }
            else {
                buf = static_cast<SGDParamState&>(*param_state->second).momentum_buffer();
                buf.mul_(options.momentum()).add_(update, 1 - options.dampening());
            }
            if (options.nesterov()) {
                update = update.add(buf, options.momentum());
            }
            else {
                update = buf;
            }
        }

        NoGradGuard guard;
        p.add_(-options.learning_rate() * update);
    }
#endif

    //iteration_ += 1;
    return loss;
}

void SGDAGC::save(torch::serialize::OutputArchive& archive) const
{
   // torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
   // torch::optim::serialize(archive, "iteration_", iteration_);
}

void SGDAGC::load(torch::serialize::InputArchive& archive)
{
    //torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
   // torch::optim::serialize(archive, "iteration_", iteration_);
}

/*int64_t SGDAGC::iteration() const
{
    return iteration_;
}*/
} // namespace optim
} // namespace torch
