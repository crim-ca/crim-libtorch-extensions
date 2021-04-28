#pragma once
#include <string>
#include <torch/nn.h>
//#include "../macros.h"
#include "Base.h"

/*namespace vision {
    namespace models {*/
        template <typename Block>
        struct NFNetImpl;

        namespace _nfnetimpl {
            // 3x3 convolution with padding
            ScaledStdConv2d myconv3x3(
                int64_t in,
                int64_t out,
                int64_t stride = 1,
                int64_t groups = 1);

            // 1x1 convolution
            ScaledStdConv2d myconv1x1(int64_t in, int64_t out, int64_t stride = 1);

            struct /*VISION_API*/ BasicBlock : torch::nn::Module {
                template <typename Block>
                friend struct /*vision::models::*/NFNetImpl;

                int64_t stride;
                double alpha, beta;
                int64_t dilation;
                std::string activation;
                torch::nn::Sequential downsample;

                torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };              

                static int expansion;

                BasicBlock(
                    int64_t inplanes,
                    int64_t planes,
                    int64_t stride = 1,
                    const torch::nn::Sequential& downsample = nullptr,
                    int64_t groups = 1,
                    int64_t base_width = 64, int64_t dilation=1,
                    double alpha = 0.2,
                    double beta = 1.0,
                    std::string _activation = "relu");

                torch::Tensor forward(torch::Tensor x);
            };

            struct /*VISION_API*/ Bottleneck : torch::nn::Module {
                template <typename Block>
                friend struct /*vision::models::*/NFNetImpl;

                int64_t stride;
                torch::nn::Sequential downsample;

                torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };                
                double alpha, beta;
                int64_t dilation;
                std::string activation;
                static int expansion;

                Bottleneck(
                    int64_t inplanes,
                    int64_t planes,
                    int64_t stride = 1,
                    const torch::nn::Sequential& downsample = nullptr,
                    int64_t groups = 1,
                    int64_t base_width = 64,
                    int64_t dilation = 1,
                    double alpha = 0.2,
                    double beta = 1.0,
                    std::string _activation = "relu");

                torch::Tensor forward(torch::Tensor X);
            };
        } // namespace _resnetimpl

        template <typename Block>
        struct NFNetImpl : torch::nn::Module {
            int64_t groups, base_width, inplanes, dilation;
            torch::nn::Conv2d conv1;            
            torch::nn::Sequential layer1, layer2, layer3, layer4;
            torch::nn::Linear fc;


            torch::nn::Sequential _make_layer(
                int64_t planes,
                int64_t blocks,
                int64_t stride = 1,bool dilate=false, double alpha=0.2, double beta=1.0);

            explicit NFNetImpl(
                const std::vector<int>& layers,
                int64_t num_classes = 1000,
                bool zero_init_residual = false,
                int64_t groups = 1,
                int64_t width_per_group = 64, int64_t dilation=1);

            torch::Tensor forward(torch::Tensor X);
        };

        template <typename Block>
        torch::nn::Sequential NFNetImpl<Block>::_make_layer(
            int64_t planes,
            int64_t blocks,
            int64_t stride, bool dilate,
            double alpha, double beta) {
            torch::nn::Sequential downsample = nullptr;

            auto previous_dilation = dilation;
            if (dilate) {
                dilation *= stride;
                stride = 1;
            }

            if (stride != 1 || inplanes != planes * Block::expansion) {
                downsample = torch::nn::Sequential(
                    _nfnetimpl::myconv1x1(inplanes, planes * Block::expansion, stride));
            }

            torch::nn::Sequential layers;
            layers->push_back(
                Block(inplanes, planes, stride, downsample, groups, base_width, previous_dilation, alpha,beta));

            inplanes = planes * Block::expansion;

            for (int i = 1; i < blocks; ++i)
                layers->push_back(Block(inplanes, planes, 1, nullptr, groups, base_width, dilation, alpha, beta));

            return layers;
        }

        template <typename Block>
        NFNetImpl<Block>::NFNetImpl(
            const std::vector<int>& layers,
            int64_t num_classes,
            bool zero_init_residual,
            int64_t groups,
            int64_t width_per_group, int64_t dilation)
            : groups(groups),
            base_width(width_per_group),
            inplanes(64), dilation(dilation),
            conv1(
                torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)),
      
            layer1(_make_layer(64, layers[0], 1, false, 0.2, 1.0)),
            layer2(_make_layer(128, layers[1], 2, false, 0.2, 1.0)),
            layer3(_make_layer(256, layers[2], 2,false, 0.2, 1.0)),
            layer4(_make_layer(512, layers[3], 2, false, 0.2, 1.0)),
            fc(512 * Block::expansion, num_classes) 
        {
            register_module("conv1", conv1);            
            register_module("fc", fc);

            register_module("layer1", layer1);
            register_module("layer2", layer2);
            register_module("layer3", layer3);
            register_module("layer4", layer4);

            for (auto& module : modules(/*include_self=*/false)) {
                if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
                    torch::nn::init::kaiming_normal_(
                        M->weight,
                        /*a=*/0,
                        torch::kFanOut,
                        torch::kReLU);                
            }

        }

        template <typename Block>
        torch::Tensor NFNetImpl<Block>::forward(torch::Tensor x) {
            x = conv1->forward(x);
            x = torch::relu(x);
            x = torch::max_pool2d(x, 3, 2, 1);

            x = layer1->forward(x);
            x = layer2->forward(x);
            x = layer3->forward(x);
            x = layer4->forward(x);

            x = torch::adaptive_avg_pool2d(x, { 1, 1 });
            x = x.reshape({ x.size(0), -1 });
            //std::cout << "*" << x << "*" << std::endl;
            x = fc->forward(x);
            //std::cout << "**" << x << "**" << std::endl;
            if (torch::isnan(x).any().item<bool>()) {
                std::cout << "*" << x << "*" << std::endl;
            }
            return x;
        }

        struct /*VISION_API*/ NFNet18Impl : NFNetImpl<_nfnetimpl::BasicBlock> {
            explicit NFNet18Impl(
                int64_t num_classes = 1000,
                bool zero_init_residual = false);
        };

        struct /*VISION_API*/ NFNet34Impl : NFNetImpl<_nfnetimpl::BasicBlock> {
            explicit NFNet34Impl(
                int64_t num_classes = 1000,
                bool zero_init_residual = false);
        };

        struct /*VISION_API*/ NFNet50Impl : NFNetImpl<_nfnetimpl::Bottleneck> {
            explicit NFNet50Impl(
                int64_t num_classes = 1000,
                bool zero_init_residual = false);
        };

        struct /*VISION_API*/ NFNet101Impl : NFNetImpl<_nfnetimpl::Bottleneck> {
            explicit NFNet101Impl(
                int64_t num_classes = 1000,
                bool zero_init_residual = false);
        };

        struct /*VISION_API*/ NFNet152Impl : NFNetImpl<_nfnetimpl::Bottleneck> {
            explicit NFNet152Impl(
                int64_t num_classes = 1000,
                bool zero_init_residual = false);
        };


        template <typename Block>
        struct /*VISION_API*/ NFNet : torch::nn::ModuleHolder<NFNetImpl<Block>> {
            using torch::nn::ModuleHolder<NFNetImpl<Block>>::ModuleHolder;
        };

        TORCH_MODULE(NFNet18);
        TORCH_MODULE(NFNet34);
        TORCH_MODULE(NFNet50);
        TORCH_MODULE(NFNet101);
        TORCH_MODULE(NFNet152);

   // } // namespace models
//} // namespace vision
