// From https://github.com/pytorch/examples/blob/master/cpp/transfer-learning/main.cpp
#include <memory>
#include <algorithm>
#include <vector>
#include <fstream>

#include "CLI/CLI.hpp"
#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "torchvision/models/resnet.h"
#include "data/DataAugmentation.h"
#include "nn/models/EfficientNet.h"
#include "nn/models/NFNet.h"
#include "nn/models/ResNet.h"
#include "optim/SGD_AGC.h"

#include "training.h"

using DataSamples_t = std::pair<std::vector<std::string>, std::vector<int>>;


/*
struct _EfficientNet: public EfficientNetV1Impl, public IBaseModel {
    explicit _EfficientNet(EfficientNetOptions o, int n) :EfficientNetV1Impl(o, n) {}
    virtual  torch::Tensor forward(torch::Tensor x) {
        return EfficientNetV1Impl::forward(x);
    }
};

struct _NFNet : public NFNet34Impl, public IBaseModel {
    explicit _NFNet(int n) : NFNet34Impl(n) {}
    virtual  torch::Tensor forward(torch::Tensor x) {
        return NFNet34Impl::forward(x);
    }
};
*/

void splitData(DataSamples_t srcPairs, double splitProportion, DataSamples_t& trainingPairs, DataSamples_t& validationsPairs) {

    std::vector<int> vIndices;
    for (int i = 0; i < srcPairs.first.size(); i++)
        vIndices.push_back(i);

    std::vector<int>::iterator bound;
    bound = std::partition(vIndices.begin(), vIndices.end(), [&](auto a) {
        double aa = (double)std::rand() / (RAND_MAX + 1u);
        return  aa < splitProportion; });

    for (std::vector<int>::iterator it = vIndices.begin(); it != bound; ++it) {
        trainingPairs.first.push_back(srcPairs.first.at(*it));
        trainingPairs.second.push_back(srcPairs.second.at(*it));
    }


    for (std::vector<int>::iterator it = bound; it != vIndices.end(); ++it)
    {
        validationsPairs.first.push_back(srcPairs.first.at(*it));
        validationsPairs.second.push_back(srcPairs.second.at(*it));
    }

}

enum class ArchType: int { Resnet34, EfficientNetB0, NFNet34 };
enum class OptimType : int { SGD, AGC_SGD, Adam};

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

    CLI::App app("TestBench for testing EfficientNet, NFNet, etc.");
    ArchType archtype{ ArchType::Resnet34 };
    std::map<std::string, ArchType> map{
        {"resnet",       ArchType::Resnet34},
        {"efficientnet", ArchType::EfficientNetB0},
        {"nfnet",        ArchType::NFNet34}
    };
    app.add_option("-a,--arch", archtype, "Architecture")
        ->required()
        ->transform(CLI::CheckedTransformer(map, CLI::ignore_case));
    OptimType optimtype{ OptimType::SGD };
    std::map<std::string, OptimType> optimmap{
        {"sgd",     OptimType::SGD},
        {"sgdagc",  OptimType::AGC_SGD},
        {"adam",    OptimType::Adam}
    };
    app.add_option("-o,--optim", optimtype, "Optimizer")
        ->required()
        ->transform(CLI::CheckedTransformer(optimmap, CLI::ignore_case));

    std::string logfilename;
    double lr{ 0.001 };
    double clipping{ 0.01 };
    bool verbose{ false };

    CLI::Option* log_opt = app.add_option("-l,--logfile", logfilename, "Output log filename");
    CLI::Option* lr_opt = app.add_option("--lr", lr, "Learning rate");
    CLI::Option* verbose_opt = app.add_flag("-v,--verbose", verbose, "Verbosity");
    CLI::Option* clipping_opt = app.add_option("--clipping", clipping, "Clipping threshold");

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
   std::ostream& outlog = (fileopen? outfile:std::cout);

    if (verbose) {
        if(lr_opt->count()>0)
            outlog << "Learning rate = " << lr << std::endl;
        if(clipping_opt->count()>0)
            outlog << "Lambda = " << clipping << std::endl;
    }

    std::shared_ptr<IBaseModel> pNet;
    std::vector<torch::Tensor> params;
    uint64_t image_size = 224;
    switch (archtype) {
        case ArchType::Resnet34:
            {
                //pNet = vision::models::ResNet34(3);
                auto p = std::make_shared<_Resnet34>(3);
                params = p->parameters();
                p->to(torch::kCUDA);
                if (verbose)  outlog << *p;
                pNet = std::dynamic_pointer_cast<IBaseModel>(p);
            }
            break;
        case ArchType::EfficientNetB0:
            {
                //pNet = std::make_shared<EfficientNetV1>(EfficientNetOptions{ 1.0, 1.0, 224, 0.2 }, 3);
                auto p = std::make_shared< _EfficientNet >(EfficientNetOptions{ 1.0, 1.0, 224, 0.2 }, 3);
                params = p->parameters();
                p->to(torch::kCUDA);
                if (verbose)  outlog << *p;
                pNet = std::dynamic_pointer_cast<IBaseModel>(p);
            }
            break;
        case ArchType::NFNet34:
            {
                //pNet = std::make_shared<NFNet34>(3);
                auto p = std::make_shared<_NFNet>(3);
                params = p->parameters();
                if (verbose)  outlog << *p;
                pNet = std::dynamic_pointer_cast<IBaseModel>(p);
            }
            break;
    }

    std::shared_ptr<torch::optim::Optimizer> pOptim;

    switch (optimtype) {
        case OptimType::SGD:
            if (verbose) outlog << "Using SGD " << std::endl;
            pOptim = std::make_shared<torch::optim::SGD>(params, torch::optim::SGDOptions(lr));
            break;
        case OptimType::AGC_SGD:
            if (verbose) outlog << "Using AGCSGD " << std::endl;
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
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(
        { dataset_folder_cls1,dataset_folder_cls2, dataset_folder_cls3 }
    );
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels_val = load_data_from_folder(
        { dataset_folder_cls1_val,dataset_folder_cls2_val, dataset_folder_cls3_val }
    );

    //    std::pair<std::vector<std::string>, std::vector<int>> pairs_training, pairs_validation;
    //    splitData(pair_images_labels, 0.8, pairs_training, pairs_validation);


    // Initialize DataAugmentationDataset class and read data
    auto custom_dataset_trn = DataAugmentationDataset(pair_images_labels.first, pair_images_labels.second, image_size)
        .map(torch::data::transforms::Stack<>());
    auto custom_dataset_valid = DataAugmentationDataset(pair_images_labels_val.first, pair_images_labels_val.second, image_size)
        .map(torch::data::transforms::Stack<>());


    auto data_loader_trn = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_trn), 4);
    auto data_loader_valid = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_valid), 4);

    train(pNet, data_loader_trn, data_loader_valid, pOptim, custom_dataset_trn.size().value(), custom_dataset_valid.size().value(), outlog);

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    outlog << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    if (fileopen)
        outfile.close();

    return 0;
}
