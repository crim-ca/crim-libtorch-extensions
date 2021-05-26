// From https://github.com/pytorch/examples/blob/master/cpp/transfer-learning/main.cpp
#include <memory>
#include <algorithm>
#include <vector>
#include <fstream>

#include "CLI/CLI.hpp"
#include <torch/torch.h>
#include <torch/nn/module.h>
#include "torchvision/models/resnet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include "data/DataAugmentation.h"
#include "nn/models/EfficientNet.h"
#include "nn/models/NFNet.h"
#include "nn/models/ResNet.h"
#include "optim/SGD_AGC.h"
#include "version.h"

#include "training.h"

using namespace vision::models;  // some from TorchVision, others from our Extensions


#ifdef USE_BASE_MODEL

struct ResNet34CLI: public ResNet34Impl, public IBaseModel {
    explicit ResNet34CLI(int n) : ResNet34Impl(n) {}
    torch::Tensor forward(torch::Tensor x) {
        return ResNet34Impl::forward(x);
    }
};

struct EfficientNetV1CLI: public EfficientNetV1Impl, public IBaseModel {
    explicit EfficientNetV1CLI(EfficientNetOptions o, int n) :EfficientNetV1Impl(o, n) {}
    torch::Tensor forward(torch::Tensor x) override {
        return EfficientNetV1Impl::forward(x);
    }
};

struct EfficientNetB0CLI: public EfficientNetB0, public IBaseModel {
    explicit EfficientNetB0CLI(int n) : EfficientNetB0(n) {}
    torch::Tensor forward(torch::Tensor x) override {
        return EfficientNetB0::forward(x);
    }
};

struct NFNet34CLI : public NFNet34Impl, public IBaseModel {
    explicit NFNet34CLI(int n) : NFNet34Impl(n) {}
    torch::Tensor forward(torch::Tensor x) override {
        return NFNet34Impl::forward(x);
    }
};

#else  // !USE_BASE_MODEL

using ResNet34CLI = ResNet34Impl;
using EfficientNetB0CLI = EfficientNetB0;
using NFNet34CLI = NFNet34Impl;

#endif // USE_BASE_MODEL


#define AutoMap(_enum, _item)  {#_item, _enum::_item}
enum class ArchType: int {
    EfficientNetB0,
    NFNet34,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    //ResNext50_32x4d,
    //ResNext101_32x8d,
    //WideResNet50_2,
    //WideResNet101_2,
};
std::map<std::string, ArchType> ArchMap {
    AutoMap(ArchType, EfficientNetB0),
    AutoMap(ArchType, NFNet34),
    AutoMap(ArchType, ResNet34),
};
enum class OptimType : int {
    Adam,
    SGD,
    SGDAGC,
};
std::map<std::string, OptimType> OptimMap {
    AutoMap(OptimType, Adam),
    AutoMap(OptimType, SGD),
    AutoMap(OptimType, SGDAGC),
};

/**
 * @brief TestBench CLI to evaluate different combination of model architectures and optimizers against loaded data.
 */
int main(int argc, const char* argv[]) {
    /* sample dataset locations

    M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01440764
    M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01443537
    M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01484850
    M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01440764
    M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01443537
    M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01484850
    */

    CLI::App app("TestBench for training, evaluating and testing CRIM libtorch extensions (EfficientNet, NFNet, etc.)");
    ArchType archtype { ArchType::ResNet34 };
    app.add_option("-a,--arch", archtype, "Architecture")
        ->required()
        ->transform(CLI::CheckedTransformer(ArchMap, CLI::ignore_case));
    OptimType optimtype = { OptimType::SGD };
    app.add_option("-o,--optim", optimtype, "Optimizer")
        ->required()
        ->transform(CLI::CheckedTransformer(OptimMap, CLI::ignore_case));

    std::string logfilename;
    double lr{ 0.001 };
    double clipping{ 0.01 };
    bool verbose{ false };
    bool version{ false };

    CLI::Option* log_opt = app.add_option("-l,--logfile", logfilename, "Output log filename");
    CLI::Option* lr_opt = app.add_option("--lr", lr, "Learning rate");
    CLI::Option* verbose_opt = app.add_flag("-v,--verbose", verbose, "Verbosity");
    CLI::Option* clipping_opt = app.add_option("--clipping", clipping, "Clipping threshold");
    CLI::Option* version_opt = app.add_option("--version", version, "Print the version number.");

    CLI11_PARSE(app, argc, argv);
    //https://stackoverflow.com/questions/428630/assigning-cout-to-a-variable-name

    auto start_time = std::chrono::steady_clock::now();

    std::ofstream outfile;
    bool fileopen = false;
    if (!logfilename.empty()) {
        outfile.open(logfilename, std::ios::out);
        if (outfile.is_open())
            fileopen = true;
    }
   std::ostream& outlog = (fileopen ? outfile : std::cout);

    if (version) {
        outlog << std::string(CRIM_TORCH_EXTENSIONS_VERSION) << std::endl;
        return 0;
    }

    bool has_cuda = torch::cuda::is_available();
    if (verbose) {
        if(lr_opt->count() > 0)
            outlog << "Learning rate = " << lr << std::endl;
        if(clipping_opt->count()>0)
            outlog << "Lambda = " << clipping << std::endl;
        outlog << (has_cuda ? "CUDA detected!" : "CUDA missing! Will use CPU.") << std::endl;
    }

    size_t nb_class = 3;

    #ifdef USE_BASE_MODEL
    #ifdef USE_JIT_MODULE
    using ModelPtr = torch::jit::script::Module; /*IModel torch::nn::Module;*/
    std::shared_ptr<ModelPtr> pNet;
    #else
    using ModelPtr = std::shared_ptr<IModel>;
    ModelPtr pNet;
    #endif
    #else
    using ModelPtr = IModel;
    IModel pNet;
    #endif

    //torch::nn::AnyModule pNet;
    std::vector<torch::Tensor> params;
    uint64_t image_size = 224;
    switch (archtype) {
        case ArchType::ResNet34:
            {
                //pNet = vision::models::ResNet34(nb_class);
                auto p = std::make_shared<ResNet34CLI>(nb_class);
                #ifdef USE_BASE_MODEL
                auto net = p.get();
                #else
                auto net = p;
                #endif
                params = net->parameters();
                if (has_cuda) net->to(torch::kCUDA);
                if (verbose)  outlog << *net;
                pNet = std::dynamic_pointer_cast<IModel>(p);
                //pNet = torch::nn::AnyModule(ResNet34(nb_class));
            }
            break;
        case ArchType::EfficientNetB0:
            {
                //pNet = std::make_shared<EfficientNetV1>(EfficientNetOptions{ 1.0, 1.0, 224, 0.2 }, nb_class);
                auto p = std::make_shared<EfficientNetB0CLI>(nb_class);
                #ifdef USE_BASE_MODEL
                auto net = p.get();
                #else
                auto net = p;
                #endif
                params = net->parameters();
                if (has_cuda) net->to(torch::kCUDA);
                if (verbose)  outlog << *net;
                pNet = std::dynamic_pointer_cast<IModel>(p);
                //pNet = torch::nn::AnyModule(EfficientNetB0(nb_class));
            }
            break;
        case ArchType::NFNet34:
            {
                //pNet = std::make_shared<NFNet34>(nb_class);
                auto p = std::make_shared<NFNet34CLI>(nb_class);
                #ifdef USE_BASE_MODEL
                auto net = p.get();
                #else
                auto net = p;
                #endif
                params = net->parameters();
                if (has_cuda) net->to(torch::kCUDA);
                if (verbose)  outlog << *net;
                pNet = std::dynamic_pointer_cast<IModel>(p);
                //pNet = torch::nn::AnyModule(NFNet34(nb_class));
            }
            break;
    }
    /*auto model = pNet.ptr();
    params = model->parameters();
    if (has_cuda) model->to(torch::kCUDA);
    if (verbose) outlog << *model;*/

    std::shared_ptr<torch::optim::Optimizer> pOptim;

    switch (optimtype) {
        case OptimType::Adam:
            if (verbose) outlog << "Using Adam " << std::endl;
            pOptim = std::make_shared<torch::optim::Adam>(params, torch::optim::AdamOptions(lr));
            break;
        case OptimType::SGD:
            if (verbose) outlog << "Using SGD " << std::endl;
            pOptim = std::make_shared<torch::optim::SGD>(params, torch::optim::SGDOptions(lr));
            break;
        case OptimType::SGDAGC:
            if (verbose) outlog << "Using SGDAGC " << std::endl;
            pOptim = std::make_shared<torch::optim::SGDAGC>(params, torch::optim::SGDAGCOptions(lr));
            break;
    }

//    torch::optim::Adam opt(net.get()->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));
 //   torch::optim::SGD opt(net.get()->parameters(), torch::optim::SGDOptions(1e-3));

    //torch::optim::SGDAGC opt(net.get()->parameters(), torch::optim::SGDAGCOptions(1e-3));

    std::string dataset_folder_cls1 = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01440764";
    std::string dataset_folder_cls2 = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01443537";
    std::string dataset_folder_cls3 = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01484850";
    std::string dataset_folder_cls1_val = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01440764";
    std::string dataset_folder_cls2_val = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01443537";
    std::string dataset_folder_cls3_val = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01484850";


    // Get paths of images and labels as int from the folder paths
    std::pair<std::vector<std::string>, std::vector<Label>> pair_images_labels = load_data_from_folder(
        { dataset_folder_cls1,dataset_folder_cls2, dataset_folder_cls3 }
    );
    std::pair<std::vector<std::string>, std::vector<Label>> pair_images_labels_val = load_data_from_folder(
        { dataset_folder_cls1_val,dataset_folder_cls2_val, dataset_folder_cls3_val }
    );

    //    std::pair<std::vector<std::string>, std::vector<Label>> pairs_training, pairs_validation;
    //    splitData(pair_images_labels, 0.8, pairs_training, pairs_validation);


    // Initialize DataAugmentationDataset class and read data
    auto dataAugRNG = cv::RNG();
    auto custom_dataset_train = DataAugmentationDataset(pair_images_labels.first, pair_images_labels.second, image_size, dataAugRNG)
        .map(torch::data::transforms::Stack<>());
    auto custom_dataset_valid = DataAugmentationDataset(pair_images_labels_val.first, pair_images_labels_val.second, image_size, dataAugRNG)
        .map(torch::data::transforms::Stack<>());

    using RandomDataLoader = torch::data::samplers::RandomSampler;
    size_t batch_size = 4;
    size_t max_epochs = 2;
    auto data_loader_train = torch::data::make_data_loader<RandomDataLoader>(std::move(custom_dataset_train), batch_size);
    auto data_loader_valid = torch::data::make_data_loader<RandomDataLoader>(std::move(custom_dataset_valid), batch_size);
    auto train_size = custom_dataset_train.size().value();
    auto valid_size = custom_dataset_valid.size().value();
    train(pNet, /*lin,*/ data_loader_train, data_loader_valid, pOptim, train_size, valid_size, outlog, max_epochs);

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    outlog << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    if (fileopen)
        outfile.close();

    return 0;
}
