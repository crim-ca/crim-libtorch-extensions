// From https://github.com/pytorch/examples/blob/master/cpp/transfer-learning/main.cpp
#include <memory>
#include <algorithm>
#include <vector>
#include <fstream>

#include "CLI/CLI.hpp"
#include <torch/torch.h>
#include "torchvision/models/resnet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include "data/DataAugmentation.h"
#include "nn/models/EfficientNet.h"
#include "nn/models/NFNet.h"
#include "optim/SGD_AGC.h"

#include "training.h"

using DataSamples_t = std::pair<std::vector<std::string>, std::vector<int>>;

cv::RNG globRNG;

struct _BaseModel {
    virtual  torch::Tensor forward(torch::Tensor x)=0;
};

struct _Resnet34 : public vision::models::ResNet34Impl, public _BaseModel {
    explicit _Resnet34(int n) : ResNet34Impl(n) {}
    virtual  torch::Tensor forward(torch::Tensor x){
        return ResNet34Impl::forward(x);
      }
};

struct _EfficientNet: public EfficientNetV1Impl, public _BaseModel {
    explicit _EfficientNet(EfficientNetOptions o, int n) :EfficientNetV1Impl(o, n) {}
    virtual  torch::Tensor forward(torch::Tensor x) {
        return EfficientNetV1Impl::forward(x);
    }
};

struct _NFNet : public NFNet34Impl, public _BaseModel {
    explicit _NFNet(int n) : NFNet34Impl(n) {}
    virtual  torch::Tensor forward(torch::Tensor x) {
        return NFNet34Impl::forward(x);
    }
};


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



/*
    Function to return image read at location given as type torch::Tensor

    Resizes image to (224, 224, 3)

    Parameters
    ===========
    1. location (std::string type) - required to load image from the location

    Returns
    ===========
    torch::Tensor type - image read as tensor
*/
torch::Tensor read_data(std::string location, uint64_t image_size) {

    cv::Mat _img = cv::imread(location, 1); // 256x256
    //cv::resize(img, img, cv::Size(image_size, image_size));// , 0, 0, cv::INTER_CUBIC);

    // Data augmentation
    auto img = ImageTransform(_img, image_size,5, 5, 5, 1,1, 5.0/224, 5.0/224, 0 /*1?*/, 0.2, 0.2, globRNG);
    static int n = 0;
    char str[80];
    sprintf(str, "H:\\projets\\efficientnet-libtorch\\build\\temp\\avant%d.png", n);
    cv::imwrite(str, _img);
    sprintf(str, "H:\\projets\\efficientnet-libtorch\\build\\temp\\apres%d.png", n++);
    cv::imwrite(str, img);


    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

torch::Tensor read_label(int label) {
    /*
     Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
     Parameters
     ===========
     1. label (int type) - required to convert int to tensor

     Returns
     ===========
     torch::Tensor type - label read as tensor
    */
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, uint64_t image_size) {
    /*
     Function returns vector of tensors (images) read from the list of images in a folder
     Parameters
     ===========
     1. list_images (std::vector<std::string> type) - list of image paths in a folder to be read

     Returns
     ===========
     std::vector<torch::Tensor> type - Images read as tensors
     */
    std::vector<torch::Tensor> states;
    for(std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it, image_size);
        states.push_back(img);
    }
    return states;
}

std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    /*
     Function returns vector of tensors (labels) read from the list of labels
     Parameters
     ===========
     1. list_labels (std::vector<int> list_labels) -

     Returns
     ===========
     std::vector<torch::Tensor> type - returns vector of tensors (labels)
     */
    std::vector<torch::Tensor> labels;
    for(std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

std::pair<std::vector<std::string>,std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name) {
    /*
     Function to load data from given folder(s) name(s) (folders_name)
     Returns pair of vectors of string (image locations) and int (respective labels)
     Parameters
     ===========
     1. folders_name (std::vector<std::string> type) - name of folders as a vector to load data from

     Returns
     ===========
     std::pair<std::vector<std::string>, std::vector<int>> type - returns pair of vector of strings (image paths) and respective labels' vector (int label)
     */
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    int label = 0;
    for(auto const& value: folders_name) {
        std::string base_name = value + "\\";
        // cout << "Reading from: " << base_name << endl;
        DIR* dir;
        struct dirent *ent;
        if((dir = opendir(base_name.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
//                std::cout << filename << std::endl;
                if(filename.length() > 4 && filename.substr(filename.length() - 4) == "JPEG") {
                    // cout << base_name + ent->d_name << endl;
                    // cv::Mat temp = cv::imread(base_name + "/" + ent->d_name, 1);
//                    std::cout << "push ----"<< base_name + ent->d_name << "----"<<std::endl;
                    list_images.push_back(base_name + ent->d_name);
                    list_labels.push_back(label);
                }
            }
            closedir(dir);
        } else {
            std::cout << "Could not open directory" << std::endl;
            // return EXIT_FAILURE;
        }
        label += 1;
    }
    return std::make_pair(list_images, list_labels);
}

template<typename Dataloader>
void train(std::shared_ptr<_BaseModel> net, Dataloader& data_loader_trn, Dataloader& data_loader_valid, std::shared_ptr<torch::optim::Optimizer> optimizer, int size_trn, int size_valid, std::ostream& outlog) {

    float best_accuracy = 0.0;
    int batch_index = 0;
    outlog << "Training set size: " << size_trn << std::endl;
    outlog << "Validation set size: " << size_valid << std::endl;



    for(int i=0; i<2; i++) {
        float mse = 0;
        float Acc = 0.0;
        float valid_acc = 0.0;

        for(auto& batch: *data_loader_trn) {
            auto data = batch.data;
            auto target = batch.target.squeeze();

            // Should be of length: batch_size
            data = data.to(torch::kF32).to(torch::kCUDA);
            target = target.to(torch::kInt64).to(torch::kCUDA);

            //std::vector<torch::jit::IValue> input;
            //input.push_back(data);
            optimizer->zero_grad();

            auto output = net.get()->forward(data);
            // For transfer learning
            output = output.view({output.size(0), -1});
           // std::cout << output<<std::endl;
            //output = lin(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);

            loss.backward();
            optimizer->step();

            auto acc = output.argmax(1).eq(target).sum();

            Acc += acc.template item<float>();
            mse += loss.template item<float>();

            batch_index += 1;
        }

        for (auto& batch : *data_loader_valid) {
            auto data = batch.data;
            auto target = batch.target.squeeze();

            // Should be of length: batch_size
            data = data.to(torch::kF32).to(torch::kCUDA);
            target = target.to(torch::kInt64).to(torch::kCUDA);

            auto output = net.get()->forward(data);
            output = output.view({ output.size(0), -1 });
            auto acc = output.argmax(1).eq(target).sum();

            valid_acc += acc.template item<float>();
        }


        mse = mse/float(batch_index); // Take mean of loss
        outlog << "Epoch: " << i  << ", " << "MSE: " << mse << ", training accuracy: " << Acc/size_trn<< ", validation accuracy: " << valid_acc / size_valid << std::endl;
        outlog << "** " << mse << " "<< Acc / size_trn << " " << valid_acc/size_valid << std::endl;

        /*test(net, lin, data_loader, dataset_size);*/

        if(valid_acc/size_valid > best_accuracy) {
            best_accuracy = valid_acc/size_valid;
            std::cout << "Saving model" << std::endl;
            ///net.get().save("model.pt");   need a cast?
            //torch::save(lin, "model_linear.pt");
        }
    }
}

enum class ArchType: int { Resnet34, EfficientNetB0, NFNet34 };
enum class OptimType : int { SGD, AGC_SGD, Adam};

int main(int argc, const char* argv[]) {

    CLI::App app("TestBench for testing EfficientNet, NFNet, etc.");
    ArchType archtype{ ArchType::Resnet34 };
    std::map<std::string, ArchType> map{ {"resnet", ArchType::Resnet34}, {"efficientnet", ArchType::EfficientNetB0}, {"nfnet", ArchType::NFNet34} };
    app.add_option("-a,--arch", archtype, "Architecture")
        ->required()
        ->transform(CLI::CheckedTransformer(map, CLI::ignore_case));
    OptimType optimtype{ OptimType::SGD };
    std::map<std::string, OptimType> optimmap{ {"sgd", OptimType::SGD}, {"sgdagc", OptimType::AGC_SGD}, {"adam", OptimType::Adam} };
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

    std::shared_ptr<_BaseModel> pNet;
    std::vector<torch::Tensor> params;
    uint64_t image_size = 224;
    switch (archtype)
    {
    case ArchType::Resnet34:
        //pNet = vision::models::ResNet34(3);
        { auto p = std::make_shared<_Resnet34>(3);
        params = p->parameters();
        p->to(torch::kCUDA);
        if (verbose)  outlog << *p;
        pNet = std::dynamic_pointer_cast<_BaseModel>(p);
        }
        break;
    case ArchType::EfficientNetB0:
        {
        //pNet = std::make_shared<EfficientNetV1>(EfficientNetOptions{ 1.0, 1.0, 224, 0.2 }, 3);
        auto p = std::make_shared< _EfficientNet >(EfficientNetOptions{ 1.0, 1.0, 224, 0.2 }, 3);
        params = p->parameters();
        p->to(torch::kCUDA);
        if (verbose)  outlog << *p;
        pNet = std::dynamic_pointer_cast<_BaseModel>(p);
        }
        break;
    case ArchType::NFNet34:
        {
        //pNet = std::make_shared<NFNet34>(3);
        auto p = std::make_shared<_NFNet>(3);
        params = p->parameters();
        if (verbose)  outlog << *p;
        pNet = std::dynamic_pointer_cast<_BaseModel>(p);
        }
        break;

    }

    std::shared_ptr<torch::optim::Optimizer> pOptim;

    switch (optimtype) {
    case OptimType::SGD:
        if (verbose)  outlog << "Using SGD " << std::endl;
        pOptim = std::make_shared<torch::optim::SGD>(params, torch::optim::SGDOptions(lr));
        break;
    case OptimType::AGC_SGD:
        if (verbose)  outlog << "Using AGCSGD " << std::endl;
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


        // Initialize CustomDataset class and read data
      //  auto custom_dataset_trn = CustomDataset(pairs_training.first, pairs_training.second).map(torch::data::transforms::Stack<>());
      //  auto custom_dataset_valid = CustomDataset(pairs_validation.first, pairs_validation.second).map(torch::data::transforms::Stack<>());
    auto custom_dataset_trn = CustomDataset(pair_images_labels.first, pair_images_labels.second, image_size)
        .map(torch::data::transforms::Stack<>());
    auto custom_dataset_valid = CustomDataset(pair_images_labels_val.first, pair_images_labels_val.second, image_size)
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
