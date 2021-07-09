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

#include "logger.h"
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

    try {

        CLI::App app("TestBench for training, evaluating and testing CRIM libtorch extensions (EfficientNet, NFNet, etc.)");
        #ifdef WIN32
        app.allow_windows_style_options();
        #endif

        ArchType archtype { ArchType::ResNet34 };
        app.add_option("-a,--arch", archtype, "Architecture")
            ->default_val(ArchType::ResNet34)
            ->transform(CLI::CheckedTransformer(ArchMap, CLI::ignore_case));
        OptimType optimtype { OptimType::SGD };
        app.add_option("-o,--optim", optimtype, "Optimizer")
            ->default_val(OptimType::SGD)
            ->transform(CLI::CheckedTransformer(OptimMap, CLI::ignore_case));

        std::string dataset_folder_train, dataset_folder_valid;
        std::string data_file_extension = "jpeg";
        std::string log_file_path;
        double lr{ 0.001 };
        double clipping{ 0.01 };
        size_t batch_size = 16;
        size_t max_epochs = 30;
        int workers = -1;  // allow undefined
        bool debug{ false };
        bool verbose{ false };
        bool version{ false };
        bool no_color{ false };

        app.add_option("--train", dataset_folder_train,
            "Directory where training images categorized by class sub-folders can be loaded from."
        );
        app.add_option("--valid", dataset_folder_valid,
            "Directory where validation images categorized by class sub-folders can be loaded from."
        );
        app.add_option("-e,--extension", data_file_extension,
            "Extension of image files to be considered for loading data."
        )->default_val(data_file_extension);

        app.add_option("-E,--max-epochs", max_epochs, "Maximum number of training epochs.")->default_val(max_epochs);
        app.add_option("-B,--batch-size", batch_size, "Batch size of each iteration.")->default_val(batch_size);
        app.add_option("-W,--workers", workers, "Number of data loader workers to employ.")->default_val(workers);
        CLI::Option* lr_opt = app.add_option("--lr", lr, "Learning rate")->default_val(lr);
        CLI::Option* clipping_opt = app.add_option("--clipping", clipping, "Clipping threshold")->default_val(clipping);

        app.add_option("-l,--logfile", log_file_path, "Output log path.");
        app.add_flag("-d,--debug", debug, "Log additional debug messages.")->default_val(debug);
        app.add_flag("-v,--verbose", verbose,
            "Log more verbose messages including function and lines numbers."
        )->default_val(verbose);
        app.add_flag("--no-color", no_color, "Disable color formatting of log entries.")->default_val(no_color);
        app.add_flag("-V,--version", version, "Print the version number.");

        CLI11_PARSE(app, argc, argv);
        //https://stackoverflow.com/questions/428630/assigning-cout-to-a-variable-name

        auto start_time = std::chrono::steady_clock::now();

        if (version) {
            std::cout << std::string(CRIM_TORCH_EXTENSIONS_VERSION) << std::endl;
            return EXIT_SUCCESS;
        }

        #ifndef USE_LOG_COUT

            plog::Severity level = verbose ? plog::verbose : (debug ? plog::debug :  plog::info);
            bool withFile = log_file_path.length() > 0;
            if (verbose) {
                static plog::ConsoleAppender<plog::VerboseFormatter> consoleAppender;
                static plog::ColorConsoleAppender<plog::VerboseFormatter> colorConsoleAppender;
                static plog::RollingFileAppender<plog::VerboseFormatter> fileAppender(log_file_path.c_str());
                if (withFile)
                    plog::init(level, no_color ? &consoleAppender : &colorConsoleAppender).addAppender(&fileAppender);
                else
                    plog::init(level, no_color ? &consoleAppender : &colorConsoleAppender);
            }
            else {
                static plog::ConsoleAppender<plog::MinimalFormatter> consoleAppender;
                static plog::ColorConsoleAppender<plog::MinimalFormatter> colorConsoleAppender;
                static plog::RollingFileAppender<plog::MinimalFormatter> fileAppender(log_file_path.c_str());
                if (withFile)
                    plog::init(level, no_color ? &consoleAppender : &colorConsoleAppender).addAppender(&fileAppender);
                else
                    plog::init(level, no_color ? &consoleAppender : &colorConsoleAppender);
            }

        #endif // USE_LOG_COUT

        bool has_cuda = torch::cuda::is_available();
        if (verbose) {
            if(lr_opt->count() > 0)
                LOGGER(DEBUG) << "Learning rate = " << lr << std::endl;
            if(clipping_opt->count()>0)
                LOGGER(DEBUG) << "Lambda = " << clipping << std::endl;
            LOGGER(DEBUG) << (has_cuda ? "CUDA detected!" : "CUDA missing! Will use CPU.") << std::endl;
        }

        if (dataset_folder_train.empty() || dataset_folder_valid.empty()) {
            LOGGER(ERROR) << "Invalid directories for train/valid datasets provided no data!" << std::endl;
            return EXIT_FAILURE;
        }

        LOGGER(DEBUG) << "Loading samples..." << std::endl;

        // Get paths of images and labels from the folder paths
        std::pair<std::vector<std::string>, std::vector<Label>> samples_train = load_data_from_folder(
            dataset_folder_train, data_file_extension
        );
        std::pair<std::vector<std::string>, std::vector<Label>> samples_valid = load_data_from_folder(
            dataset_folder_valid, data_file_extension
        );

        size_t nb_class_train = count_classes(samples_train.second);
        size_t nb_class_valid = count_classes(samples_valid.second);
        size_t nb_class = std::max(nb_class_train, nb_class_valid);

        LOGGER(DEBUG) << "Number of found classes: " << nb_class << std::endl;
        LOGGER(DEBUG) << "Number of train classes: " << nb_class_train << std::endl;
        LOGGER(DEBUG) << "Number of valid classes: " << nb_class_valid << std::endl;
        LOGGER(DEBUG) << "Number of train samples: " << samples_train.first.size() << std::endl;
        LOGGER(DEBUG) << "Number of valid samples: " << samples_valid.first.size() << std::endl;
        show_machine_memory();
        show_gpu_memory();

        if (nb_class < 2) {
            LOGGER(ERROR) << "Cannot train without at least 2 classes!" << std::endl;
            return EXIT_FAILURE;
        }
        if (nb_class_train == 0 || nb_class_valid == 0) {
            LOGGER(ERROR) << "Cannot run train/valid loops without samples!" << std::endl;
            return EXIT_FAILURE;
        }

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
                    LOGGER(DEBUG) << std::endl << *net << std::endl;
                    #ifdef USE_BASE_MODEL
                    pNet = std::dynamic_pointer_cast<IModel>(p);
                    #else
                    pNet = p;
                    #endif
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
                    LOGGER(DEBUG) << std::endl << *net << std::endl;
                    #ifdef USE_BASE_MODEL
                    pNet = std::dynamic_pointer_cast<IModel>(p);
                    #else
                    pNet = p;
                    #endif
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
                    LOGGER(DEBUG) << std::endl << *net << std::endl;
                    #ifdef USE_BASE_MODEL
                    pNet = std::dynamic_pointer_cast<IModel>(p);
                    #else
                    pNet = p;
                    #endif
                    //pNet = torch::nn::AnyModule(NFNet34(nb_class));
                }
                break;
        }
        /*auto model = pNet.ptr();
        params = model->parameters();
        if (has_cuda) model->to(torch::kCUDA);
        LOGGER(DEBUG) << *model;*/

        std::shared_ptr<torch::optim::Optimizer> pOptim;

        switch (optimtype) {
            case OptimType::Adam:
                LOGGER(DEBUG) << "Using Adam" << std::endl;
                pOptim = std::make_shared<torch::optim::Adam>(params, torch::optim::AdamOptions(lr));
                break;
            case OptimType::SGD:
                LOGGER(DEBUG) << "Using SGD" << std::endl;
                pOptim = std::make_shared<torch::optim::SGD>(params, torch::optim::SGDOptions(lr));
                break;
            case OptimType::SGDAGC:
                LOGGER(DEBUG) << "Using SGDAGC" << std::endl;
                pOptim = std::make_shared<torch::optim::SGDAGC>(params, torch::optim::SGDAGCOptions(lr));
                break;
        }

        //    std::pair<std::vector<std::string>, std::vector<Label>> pairs_training, pairs_validation;
        //    splitData(pair_images_labels, 0.8, pairs_training, pairs_validation);

        show_machine_memory();
        show_gpu_memory();

        // Initialize DataAugmentationDataset class and read data
        LOGGER(INFO) << "Generating data loaders with data augmentation..." << std::endl;
        auto dataAugRNG = cv::RNG();
        auto custom_dataset_train = DataAugmentationDataset(
            samples_train.first, samples_train.second, image_size, dataAugRNG
        ).map(torch::data::transforms::Stack<>());
        auto custom_dataset_valid = DataAugmentationDataset(
            samples_valid.first, samples_valid.second, image_size, dataAugRNG
        ).map(torch::data::transforms::Stack<>());

        LOGGER(INFO) << "Creating random samplers..." << std::endl;
        using RandomDataLoader = torch::data::samplers::RandomSampler;
        torch::data::DataLoaderOptions loadOpts(batch_size);
        if (workers >= 0)
            loadOpts.workers(workers);
        auto data_loader_train = torch::data::make_data_loader<RandomDataLoader>(std::move(custom_dataset_train), loadOpts);
        auto data_loader_valid = torch::data::make_data_loader<RandomDataLoader>(std::move(custom_dataset_valid), loadOpts);
        auto train_size = custom_dataset_train.size().value();
        auto valid_size = custom_dataset_valid.size().value();
        show_machine_memory();
        show_gpu_memory();

        LOGGER(INFO) << "Starting train/valid loop..." << std::endl;
        train(pNet, /*lin,*/ data_loader_train, data_loader_valid, pOptim, train_size, valid_size, max_epochs);

        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        LOGGER(INFO) << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    }
    catch (const std::exception& e) {
        LOGGER(ERROR) << "Unhandled exception occurred!" << std::endl << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
