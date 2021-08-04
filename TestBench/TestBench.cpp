
#include <stdafx.h>

// reference example code:  https://github.com/pytorch/examples/blob/master/cpp/transfer-learning/main.cpp

#include <memory>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iomanip>
#include <thread>

#include "CLI/CLI.hpp"
#include <torch/torch.h>
#include <torch/nn/module.h>
#include "torchvision/models/resnet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/utils/filesystem.hpp>

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

        std::string dataset_folder_train, dataset_folder_valid;
        std::string data_file_extension = "jpeg";
        std::string log_file_path;
        double lr{ 1e-3 };   // only param that is mandatory to initialize options
        double clipping{ -1.0 }, dampening{ -1.0 }, momentum{ -1.0 }, epsilon{ -1.0 }, weight_decay{ -1.0 };
        std::tuple<double, double> betas{ -1, -1 };
        bool nesterov, amsgrad;
        size_t batch_size = 16;
        size_t max_epochs = 30;
        size_t max_train_samples = 0, max_valid_samples = 0;
        unsigned int seed = unsigned(std::time(nullptr));  // always random if not overridden
        int workers = -1;  // allow undefined
        bool debug{ false };
        bool verbose{ false };
        bool version{ false };
        bool log_color{ true };

        auto train_opts = app.add_option_group("Training Parameters",
            "Parameters that define behavior of training iterations and data loading.");
        train_opts->add_option("--train", dataset_folder_train,
            "Directory where training images categorized by class sub-folders can be loaded from."
        )->check(CLI::ExistingDirectory)->required();
        train_opts->add_option("--valid", dataset_folder_valid,
            "Directory where validation images categorized by class sub-folders can be loaded from."
        )->check(CLI::ExistingDirectory)->required();
        train_opts->add_option("-e,--extension", data_file_extension,
            "Extension of image files to be considered for loading data."
        )->default_val(data_file_extension);
        train_opts->add_option("--max-train-samples", max_train_samples,
            "Maximum amount of samples to preserve from available training dataset directory. "
            "Consume all available samples if not specified, 0 or greater than available amount."
        )->check(CLI::PositiveNumber);
        train_opts->add_option("--max-valid-samples", max_valid_samples,
            "Maximum amount of samples to preserve from available validation dataset directory. "
            "Consume all available samples if not specified, 0 or greater than available amount."
        )->check(CLI::PositiveNumber);
        train_opts->add_option("-E,--max-epochs", max_epochs,
            "Maximum number of training epochs."
        )->check(CLI::PositiveNumber)->default_val(max_epochs);
        train_opts->add_option("-B,--batch-size", batch_size,
            "Batch size of each iteration."
        )->check(CLI::PositiveNumber)->default_val(batch_size);
        train_opts->add_option("-W,--workers", workers,
            "Number of data loader workers to employ for data loading. "
            "If not specified or negative (default), employ whichever amount available based on CPU count. "
            "If equal to zero, run in single-thread mode (worker will be on the same thread as main process). "
            "Otherwise, a positive integer indicates explicitly how many worker threads to employ."
        )->default_val(workers);
        train_opts->add_option("-S,--seed", seed,
            "Seed for initialization of random number generator applied wherever needed."
        )->check(CLI::PositiveNumber);

        auto model_opts = app.add_option_group("Model Parameters",
            "Parameters relevant for model selection and configuration.");
        ArchType arch_type { ArchType::ResNet34 };
        model_opts->add_option("-a,--arch", arch_type, "Architecture")
            ->default_val(arch_type)
            ->transform(CLI::CheckedTransformer(ArchMap, CLI::ignore_case));
        std::string ckpt_load_path, ckpt_save_path;
        model_opts->add_option("-c,--checkpoint", ckpt_load_path,
            "Model checkpoint to load (must match architecture)."
        )->check(CLI::ExistingFile);
        model_opts->add_option("-s,--save-dir", ckpt_save_path,
            "Save location of intermediate epoch model checkpoints."
        )->default_val("./checkpoints");  // doesn't need to exist, will create if missing

        // because many different optimizers use the same hyperparameter names for different purposes / defaults,
        // don't define any default here and use CLI option to detect if specific OptimizerOption default must be used
        auto optim_opts = app.add_option_group("Optimizer Hyperparameters",
            "Hyperparameters values employed by the selected optimizer. "
            "Omitted options use PyTorch defaults. Not applicable options for selected optimizer are ignored.");
        OptimType optim_type { OptimType::SGD };
        optim_opts->add_option("-o,--optim", optim_type, "Optimizer")
            ->default_val(optim_type)
            ->transform(CLI::CheckedTransformer(OptimMap, CLI::ignore_case));
        auto lr_opt = optim_opts->add_option("--lr", lr, "Learning rate")->check(CLI::PositiveNumber);
        auto epsilon_opt = optim_opts->add_option("--epsilon", epsilon, "Epsilon")->check(CLI::NonNegativeNumber);
        auto amsgrad_opt = optim_opts->add_option("--amsgrad", amsgrad, "AMSGrad");
        auto weight_decay_opt = optim_opts->add_option("--weight-decay", amsgrad, "Weight Decay")->check(CLI::NonNegativeNumber);
        auto clipping_opt = optim_opts->add_option("--clipping", clipping, "Clipping lambda threshold")->check(CLI::NonNegativeNumber);
        auto momentum_opt = optim_opts->add_option("--momentum", clipping, "Momentum")->check(CLI::NonNegativeNumber);
        auto dampening_opt = optim_opts->add_option("--dampening", clipping, "Dampening")->check(CLI::NonNegativeNumber);
        auto nesterov_opt = optim_opts->add_option("--nesterov", nesterov, "Nesterov");
        auto betas_opt = optim_opts->add_option("--betas", betas,
            "Beta1 and Beta2 (both requried if option is specified, separated by space)."
        )->check(CLI::PositiveNumber);/*->expected(2) */  // expected size automatic with tuple

        #ifndef USE_LOG_COUT

        auto log_opts = app.add_option_group("Logging Parameters", "Control logging options.");
        log_opts->add_flag("-v,--verbose", verbose,
            "Log more verbose messages including function and lines numbers."
        )->default_val(verbose);
        log_opts->add_flag("-d,--debug", debug, "Log additional debug messages.")->default_val(debug);
        log_opts->add_option("-l,--logfile", log_file_path, "Output log path.");
        log_opts->add_flag("--color,!--no-color", log_color,
            "Enable/disable color formatting of log entries."
        )->default_val(log_color);

        #endif // USE_LOG_COUT

        app.add_flag("-V,--version", version, "Print the version number.");

        CLI11_PARSE(app, argc, argv);
        //https://stackoverflow.com/questions/428630/assigning-cout-to-a-variable-name

        auto start_time = std::chrono::steady_clock::now();

        if (version) {
            // cout to force output regardless of log level / log utility
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
                    plog::init(level, log_color ? &colorConsoleAppender : &consoleAppender).addAppender(&fileAppender);
                else
                    plog::init(level, log_color ? &colorConsoleAppender : &consoleAppender);
            }
            else {
                static plog::ConsoleAppender<plog::MinimalFormatter> consoleAppender;
                static plog::ColorConsoleAppender<plog::MinimalFormatter> colorConsoleAppender;
                static plog::RollingFileAppender<plog::MinimalFormatter> fileAppender(log_file_path.c_str());
                if (withFile)
                    plog::init(level, log_color ? &colorConsoleAppender : &consoleAppender).addAppender(&fileAppender);
                else
                    plog::init(level, log_color ? &colorConsoleAppender : &consoleAppender);
            }

        #endif // USE_LOG_COUT

        uint w = 20;
        std::string sep = ": ", tab = "  ";
        auto worker_cpu = std::thread::hardware_concurrency();
        size_t worker_count = (workers < 0 ? worker_cpu : workers);
        LOGGER(INFO) << "Options" << std::endl
            << tab << std::setw(w) << std::left << "Workers" << sep << worker_count
                << (workers != worker_cpu ? " (auto)" : "") << std::endl
            << tab << std::setw(w) << std::left << "Batch Size" << sep << batch_size << std::endl
            << tab << std::setw(w) << std::left << "Max Epochs" << sep << max_epochs << std::endl;

        bool has_cuda = torch::cuda::is_available();
        LOGGER(INFO) << (has_cuda ? "CUDA detected!" : "CUDA missing! Will use CPU.") << std::endl;
        if (has_cuda)
            show_gpu_properties();

        if (dataset_folder_train.empty() || dataset_folder_valid.empty()) {
            LOGGER(ERROR) << "Invalid directories for train/valid datasets provided no data!" << std::endl;
            return EXIT_FAILURE;
        }

        if (!ckpt_load_path.empty()) {
            std::ifstream ckpt(ckpt_load_path.c_str());
            if (!ckpt.good()) {
                LOGGER(ERROR) << "Specified checkpoint file does not exist: [" << ckpt_load_path << "]" << std::endl;
                return EXIT_FAILURE;
            }
            LOGGER(INFO) << "Will attempt loading model checkpoint from file: [" << ckpt_load_path << "]" << std::endl;
        }
        ckpt_save_path = cv::utils::fs::canonical(ckpt_save_path);
        if (!cv::utils::fs::isDirectory(ckpt_save_path)) {
            cv::utils::fs::createDirectory(ckpt_save_path);
        }
        LOGGER(DEBUG) << "Will save epoch checkpoints in [" << ckpt_save_path << "]" << std::endl;

        LOGGER(INFO) << "Searching directories for samples..." << std::endl;

        // Get paths of images and labels from the folder paths
        DataSamples samples_train = load_data_from_folder(dataset_folder_train, data_file_extension);
        DataSamples samples_valid = load_data_from_folder(dataset_folder_valid, data_file_extension);

        size_t nb_class_train = count_classes(samples_train.second);
        size_t nb_class_valid = count_classes(samples_valid.second);
        size_t nb_class = std::max(nb_class_train, nb_class_valid);
        size_t nb_total_train = samples_train.first.size();
        size_t nb_total_valid = samples_valid.first.size();
        size_t nb_total = nb_total_train + nb_total_valid;

        LOGGER(INFO) << "Number of available total classes: " << nb_class << std::endl;
        LOGGER(INFO) << "Number of available train classes: " << nb_class_train << std::endl;
        LOGGER(INFO) << "Number of available valid classes: " << nb_class_valid << std::endl;
        LOGGER(INFO) << "Number of available total samples: " << nb_total << std::endl;
        LOGGER(INFO) << "Number of available train samples: " << nb_total_train << std::endl;
        LOGGER(INFO) << "Number of available valid samples: " << nb_total_valid << std::endl;

        if (max_train_samples != 0 && max_train_samples < nb_total_train) {
            samples_train = random_pick(samples_train, max_train_samples, seed);
            nb_class_train = count_classes(samples_train.second);
            nb_total_train = samples_train.first.size();
        }
        if (max_valid_samples != 0 && max_valid_samples < nb_total_train) {
            samples_valid = random_pick(samples_valid, max_train_samples, seed);
            nb_class_valid = count_classes(samples_valid.second);
            nb_total_valid = samples_valid.first.size();
        }
        if (nb_total != nb_total_train + nb_total_valid) {
            nb_total = nb_total_train + nb_total_valid;
            LOGGER(WARN) << "Number of samples was modified according to options!" << std::endl;
            LOGGER(INFO) << "Number of selected total classes: " << nb_class << std::endl;
            LOGGER(INFO) << "Number of selected train classes: " << nb_class_train << std::endl;
            LOGGER(INFO) << "Number of selected valid classes: " << nb_class_valid << std::endl;
            LOGGER(INFO) << "Number of selected total samples: " << nb_total << std::endl;
            LOGGER(INFO) << "Number of selected train samples: " << nb_total_train << std::endl;
            LOGGER(INFO) << "Number of selected valid samples: " << nb_total_valid << std::endl;
        }

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

        uint64_t image_size = 224;
        LOGGER(INFO) << "Will resize sample images to: "
                     << "(" << image_size << ", " << image_size << ") [enforced by model input layer]" << std::endl;

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
        switch (arch_type) {
            case ArchType::ResNet34:
                {
                    //pNet = vision::models::ResNet34(nb_class);
                    auto p = std::make_shared<ResNet34CLI>(nb_class);
                    #ifdef USE_BASE_MODEL
                        auto net = p.get();
                    #else
                        auto net = p;
                        if (!ckpt_load_path.empty())
                            torch::load(p, ckpt_load_path);
                    #endif

                    params = net->parameters();
                    if (has_cuda) net->to(torch::kCUDA);
                    LOGGER(INFO) << "Using ResNet34 model" << std::endl;
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
                        if (!ckpt_load_path.empty())
                            torch::load(p, ckpt_load_path);
                    #endif

                    params = net->parameters();
                    if (has_cuda) net->to(torch::kCUDA);
                    LOGGER(INFO) << "Using EfficientNetB0 model" << std::endl;
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
                        if (!ckpt_load_path.empty())
                            torch::load(p, ckpt_load_path);
                    #endif

                    params = net->parameters();
                    if (has_cuda) net->to(torch::kCUDA);
                    LOGGER(INFO) << "Using NFNet34 model" << std::endl;
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

        /*if (!ckpt_load_path.empty()) {
            LOGGER(INFO) << "Loading model checkpoint..." << std::endl;
            #ifdef USE_BASE_MODEL
            #error Not Implemented checkpoint loading with USE_BASE_MODEL
            #else
            torch::load(std::shared_ptr<torch::nn::Module>(pNet.ptr()), ckpt_load_path);
            #endif
        }*/

        // TODO: for real resume training, would require to load optimizer param-group states of matching epoch
        //       for now, only load the checkpoint as is and retrain instead of resume

        std::shared_ptr<torch::optim::Optimizer> pOptim;

        switch (optim_type) {
            case OptimType::Adam: {
                    auto opt = torch::optim::AdamOptions(lr);
                    if (lr_opt->count()) opt.lr(lr);
                    if (betas_opt->count()) opt.betas(betas);
                    if (epsilon_opt->count()) opt.eps(epsilon);
                    if (amsgrad_opt->count()) opt.amsgrad(amsgrad);
                    LOGGER(INFO) << "Using Adam optimizer" << std::endl
                        << tab << std::setw(w) << std::left << "Learning Rate" << sep << opt.lr() << std::endl
                        << tab << std::setw(w) << std::left << "Weight Decay" << sep << opt.weight_decay() << std::endl
                        << tab << std::setw(w) << std::left << "Betas" << sep
                            << std::get<0>(opt.betas()) << ", " << std::get<1>(opt.betas()) << std::endl
                        << tab << std::setw(w) << std::left << "Epsilon" << sep << opt.eps() << std::endl
                        << tab << std::setw(w) << std::left << "AMSGrad" << sep << opt.amsgrad() << std::endl;
                    pOptim = std::make_shared<torch::optim::Adam>(params, opt);
                }
                break;
            case OptimType::SGD: {
                    auto opt = torch::optim::SGDOptions(lr);
                    if (lr_opt->count()) opt.lr(lr);
                    if (weight_decay_opt->count()) opt.weight_decay(weight_decay);
                    if (momentum_opt->count()) opt.momentum(momentum);
                    if (dampening_opt->count()) opt.dampening(dampening);
                    if (nesterov_opt->count()) opt.nesterov(nesterov);
                    LOGGER(INFO) << "Using SGD optimizer" << std::endl
                        << tab << std::setw(w) << std::left << "Learning Rate" << sep << opt.lr() << std::endl
                        << tab << std::setw(w) << std::left << "Weight Decay" << sep << opt.weight_decay() << std::endl
                        << tab << std::setw(w) << std::left << "Momentum" << sep << opt.momentum() << std::endl
                        << tab << std::setw(w) << std::left << "Dampening" << sep << opt.dampening() << std::endl
                        << tab << std::setw(w) << std::left << "Nesterov" << sep << opt.nesterov() << std::endl;
                    pOptim = std::make_shared<torch::optim::SGD>(params, opt);
                }
                break;
            case OptimType::SGDAGC: {
                    auto opt = torch::optim::SGDAGCOptions(lr);
                    if (lr_opt->count()) opt.lr(lr);
                    if (weight_decay_opt->count()) opt.weight_decay(weight_decay);
                    if (momentum_opt->count()) opt.momentum(momentum);
                    if (dampening_opt->count()) opt.dampening(dampening);
                    if (nesterov_opt->count()) opt.nesterov(nesterov);
                    if (epsilon_opt->count()) opt.eps(epsilon);
                    if (clipping_opt->count()) opt.clipping(clipping);
                    LOGGER(INFO) << "Using SGDAGC optimizer" << std::endl
                        << tab << std::setw(w) << std::left << "Learning Rate" << sep << opt.lr() << std::endl
                        << tab << std::setw(w) << std::left << "Weight Decay" << sep << opt.weight_decay() << std::endl
                        << tab << std::setw(w) << std::left << "Momentum" << sep << opt.momentum() << std::endl
                        << tab << std::setw(w) << std::left << "Dampening" << sep << opt.dampening() << std::endl
                        << tab << std::setw(w) << std::left << "Nesterov" << sep << opt.nesterov() << std::endl
                        << tab << std::setw(w) << std::left << "Epsilon" << sep << opt.eps() << std::endl
                        << tab << std::setw(w) << std::left << "Clipping Lambda" << sep << opt.clipping() << std::endl;
                    pOptim = std::make_shared<torch::optim::SGDAGC>(params, opt);
                }
                break;
        }

        // DataSamples pairs_training, pairs_validation;
        // splitData(pair_images_labels, 0.8, pairs_training, pairs_validation);

        show_machine_memory();
        show_gpu_memory();

        // Initialize DataAugmentationDataset class and read data
        LOGGER(INFO) << "Generating data loaders with data augmentation..." << std::endl;
        auto dataAugRNG = cv::RNG(seed);
        auto custom_dataset_train = DataAugmentationDataset(
            samples_train.first, samples_train.second, image_size, dataAugRNG
        ).map(torch::data::transforms::Stack<>());
        auto custom_dataset_valid = DataAugmentationDataset(
            samples_valid.first, samples_valid.second, image_size, dataAugRNG
        ).map(torch::data::transforms::Stack<>());

        LOGGER(INFO) << "Creating random samplers..." << std::endl;
        using RandomDataLoader = torch::data::samplers::RandomSampler;
        torch::data::DataLoaderOptions loadOpts(batch_size);
        loadOpts.workers(worker_count);
        auto data_loader_train = torch::data::make_data_loader<RandomDataLoader>(std::move(custom_dataset_train), loadOpts);
        auto data_loader_valid = torch::data::make_data_loader<RandomDataLoader>(std::move(custom_dataset_valid), loadOpts);
        auto train_size = custom_dataset_train.size().value();
        auto valid_size = custom_dataset_valid.size().value();
        show_machine_memory();
        show_gpu_memory();

        LOGGER(INFO) << "Starting train/valid loop..." << std::endl;
        train(pNet, /*lin,*/ data_loader_train, data_loader_valid, pOptim, train_size, valid_size, max_epochs, ckpt_save_path);

        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        LOGGER(INFO) << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    }
    catch (const std::exception& e) {
        LOGGER(FATAL) << "Unhandled exception occurred!" << std::endl << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
