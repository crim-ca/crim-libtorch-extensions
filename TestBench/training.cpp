#include "stdafx.h"
#pragma hdrstop

#include <algorithm>
#include <vector>
#include <fstream>
#include <torch/torch.h>
#include "torchvision/models/resnet.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

#include "data/DataAugmentation.h"
#include "nn/models/BaseModel.h"
#include "nn/models/EfficientNet.h"
#include "nn/models/NFNet.h"
#include "optim/SGD_AGC.h"

#include "training.h"

/// Function to return image read at given path location
torch::Tensor read_data(std::string location, uint64_t image_size, cv::RNG& rng) {
    cv::Mat image_raw = cv::imread(location, 1);

    // Data augmentation
    auto img = ImageTransform(
        /*img*/             image_raw,
        /*size*/            image_size,
        /*yaw_sigma*/       5,
        /*ptch_sigma*/      5,
        /*roll_sigma*/      5,
        /*blur_max_sigma*/  1,
        /*noise_max_sigma*/ 1,
        /*x_slide_sigma*/   5.0 / image_size,
        /*y_slide_sigma*/   5.0 / image_size,
        /*aspect_range*/    0,          // 1?
        /*hflip_ratio*/     0.2,
        /*vflip_ratio*/     0.2,
        /*rng*/             rng);

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

/// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as tensor.
torch::Tensor read_label(int label) {
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

/// Function returns vector of tensors (images) read from the list of images in a folder
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, uint64_t image_size, cv::RNG& rng) {

    std::vector<torch::Tensor> states;
    for(auto it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it, image_size, rng);
        states.push_back(img);
    }
    return states;
}

/// Function returns vector of tensors (labels) read from the list of labels
std::vector<torch::Tensor> process_labels(std::vector<Label> list_labels) {
    std::vector<torch::Tensor> labels;
    for(std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

/// Function to load data from given folder(s) name(s) (folders_name)
std::pair<std::vector<std::string>, std::vector<Label>> load_data_from_folder(std::vector<std::string> folders_name) {
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

/// Splits data samples into training and validation sets according to specified ratio
void split_data(DataSamples srcPairs, double splitProportion, DataSamples& trainingPairs, DataSamples& validationsPairs) {

    std::vector<size_t> vIndices;
    for (auto i = 0; i < srcPairs.first.size(); i++)
        vIndices.push_back(i);

    auto bound = std::partition(vIndices.begin(), vIndices.end(), [&](auto a) {
        double aa = (double)std::rand() / (RAND_MAX + 1u);
        return  aa < splitProportion;
    });

    for (auto it = vIndices.begin(); it != bound; ++it) {
        trainingPairs.first.push_back(srcPairs.first.at(*it));
        trainingPairs.second.push_back(srcPairs.second.at(*it));
    }

    for (auto it = bound; it != vIndices.end(); ++it)
    {
        validationsPairs.first.push_back(srcPairs.first.at(*it));
        validationsPairs.second.push_back(srcPairs.second.at(*it));
    }
}
