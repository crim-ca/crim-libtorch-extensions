#include "stdafx.h"
#pragma hdrstop

#include <algorithm>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include "torchvision/models/resnet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include "nn/models/EfficientNet.h"
#include "nn/models/NFNet.h"
#include "optim/SGD_AGC.h"

#include "TestBench/training.h"

using DataSamples_t = std::pair<std::vector<std::string>, std::vector<int>>;

constexpr auto OUTPUT_FNAME = "testsgdagc_clip0.1_lr1.0.txt";

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



torch::Tensor read_data(std::string location) {
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

    cv::Mat img = cv::imread(location, 1);
    cv::resize(img, img, cv::Size(224, 224));// , 0, 0, cv::INTER_CUBIC);
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

std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
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
        torch::Tensor img = read_data(*it);
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
void train(torch::nn::ModuleHolder<NFNet34Impl>& net, Dataloader& data_loader_trn, Dataloader& data_loader_valid, torch::optim::Optimizer& optimizer, int size_trn, int size_valid) {
    /*
     This function trains the network on our data loader using optimizer.

     Also saves the model as model.pt after every epoch.
     Parameters
     ===========
     1. net (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. lin (torch::nn::Linear type) - last FC layer with revised out_features depending on the number of classes
     3. data_loader (DataLoader& type) - Training data loader
     4. optimizer (torch::optim::Optimizer& type) - Optimizer like Adam, SGD etc.
     5. size_t (dataset_size type) - Size of training dataset

     Returns
     ===========
     Nothing (void)
     */
    float best_accuracy = 0.0;
    int batch_index = 0;
    std::cout << "Training set size: " << size_trn << std::endl;
    std::cout << "Validation set size: " << size_valid << std::endl;
    std::cout << "Saving stats in " << OUTPUT_FNAME << std::endl;
    std::ofstream MyFile(OUTPUT_FNAME, std::ios::out);

    for(int i=0; i<25; i++) {
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
            optimizer.zero_grad();

            auto output = net.get()->forward(data);
            // For transfer learning
            output = output.view({output.size(0), -1});
           // std::cout << output<<std::endl;
            //output = lin(output);

            auto loss = torch::nll_loss(torch::log_softmax(output, 1), target);

            loss.backward();
            optimizer.step();

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
        std::cout << "Epoch: " << i  << ", " << "MSE: " << mse << ", training accuracy: " << Acc/size_trn<< ", validation accuracy: " << valid_acc / size_valid << std::endl;
        MyFile << mse << " "<< Acc / size_trn << " " << valid_acc/size_valid << std::endl;

        /*test(net, lin, data_loader, dataset_size);

        if(Acc/dataset_size > best_accuracy) {
            best_accuracy = Acc/dataset_size;
            std::cout << "Saving model" << std::endl;
            net.save("model.pt");
            torch::save(lin, "model_linear.pt");
        }*/
    }
    MyFile.close();
}

template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size) {
    /*
     Function to test the network on test data

     Parameters
     ===========
     1. network (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. lin (torch::nn::Linear type) - last FC layer with revised out_features depending on the number of classes
     3. loader (Dataloader& type) - test data loader
     4. data_size (size_t type) - test data size

     Returns
     ===========
     Nothing (void)
     */
    network.eval();

    float Loss = 0, Acc = 0;

    for (const auto& batch : *loader) {
        auto data = batch.data;
        auto targets = batch.target.squeeze();

        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);

        std::vector<torch::jit::IValue> input;
        input.push_back(data);

        auto output = network.forward(input).toTensor();
        output = output.view({output.size(0), -1});
        output = lin(output);

        auto loss = torch::nll_loss(torch::log_softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

int main(int argc, const char * argv[]) {


    std::string dataset_folder_cls1 = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01440764";
    std::string dataset_folder_cls2 = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01443537";
    std::string dataset_folder_cls3 = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\train\\n01484850";
    std::string dataset_folder_cls1_val = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01440764";
    std::string dataset_folder_cls2_val = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01443537";
    std::string dataset_folder_cls3_val = "M:\\data22-brs\\AARISH\\01\\nobackup\\imagenet\\training_256x256_rgb\\val\\n01484850";


    // Get paths of images and labels as int from the folder paths
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder({dataset_folder_cls1,dataset_folder_cls2, dataset_folder_cls3});
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels_val = load_data_from_folder({ dataset_folder_cls1_val,dataset_folder_cls2_val, dataset_folder_cls3_val });

//    std::pair<std::vector<std::string>, std::vector<int>> pairs_training, pairs_validation;
//    splitData(pair_images_labels, 0.8, pairs_training, pairs_validation);


    // Initialize CustomDataset class and read data
  //  auto custom_dataset_trn = CustomDataset(pairs_training.first, pairs_training.second).map(torch::data::transforms::Stack<>());
  //  auto custom_dataset_valid = CustomDataset(pairs_validation.first, pairs_validation.second).map(torch::data::transforms::Stack<>());
    auto custom_dataset_trn = CustomDataset(pair_images_labels.first, pair_images_labels.second).map(torch::data::transforms::Stack<>());
    auto custom_dataset_valid = CustomDataset(pair_images_labels_val.first, pair_images_labels_val.second).map(torch::data::transforms::Stack<>());

    auto netx = vision::models::ResNet34(3);
    auto nety = EfficientNetV1(GlobalParams{ 1.0, 1.0, 224, 0.2 }, 3);
    auto net = NFNet34(3);
    try {
        net.get()->to(torch::kCUDA);
    }
    catch (std::exception & e) {
        std::cout << e.what() << std::endl;
    }


//    torch::optim::Adam opt(net.get()->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));
 //   torch::optim::SGD opt(net.get()->parameters(), torch::optim::SGDOptions(1e-3));

    torch::optim::SGDAGC opt(net.get()->parameters(), torch::optim::SGDAGCOptions(1e-3));

    auto data_loader_trn = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_trn), 4);
    auto data_loader_valid = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_valid), 4);

    train(net, data_loader_trn, data_loader_valid, opt, custom_dataset_trn.size().value(), custom_dataset_valid.size().value());

    return 0;
}
