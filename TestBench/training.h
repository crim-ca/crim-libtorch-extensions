#pragma once
#ifndef __TRAINING_H__
#define __TRAINING_H__

#ifdef WIN32
#include "windows/dirent.h"
#else
#include /*GNU*/ <dirent.h>
#endif

#include <iostream>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>


/**
 * @brief Function to return image read at given path location
 *
 * @param location      path where to find the image
 * @param image_size    image resize dimension
 * @return              image read as tensor
 */
torch::Tensor read_data(std::string location);

/**
 * @brief Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as tensor.
 *
 * @param label     number to map the label into tensor
 * @return          label read as tensor
*/
torch::Tensor read_label(int label);

/**
 * @brief Process vector of tensors (images) read from the list of images in a folder.
 *
 * @param list_images       list of image paths to load and process
 * @return                  list of processed images as tensors
 */
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, uint64_t image_size);

/**
 * @brief Process vector of tensors (labels) read from the list of labels
 *
 * @param list_labels       list of labaels to load
 * @return                  list of labels converted to tensors
 */
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels);

/**
 * @brief Load data from given folder(s) name(s) (folders_name)
 *
 * @param folders_name      name of folders as a vector to load data from
 * @return                  Returns pair of vectors of string (image locations) and int (respective labels)
 */
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name);

/**
 * @brief Trains the neural network on our data loader using optimizer.
 *
 * During training, saves checkpoint backups of the model as `model.pt` after every epoch.
 *
 * @tparam Dataloader   Type of data loader employed by the training operation.
 *
 * @param net           Pre-trained model without last FC layer
 * @param lin           Last FC layer with revised output features count depending on the number of classes
 * @param data_loader   Training data loader
 * @param optimizer     Optimizer to use (e.g.: Adam, SGD, etc.)
 * @param size_train    Size of training dataset
 * @param size_valid    Size of validation dataset
 */
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);

/**
 * @brief Evaluate trained network inference on test data
 *
 * @tparam Dataloader   Type of data loader employed by the training operation.
 *
 * @param net           Pre-trained model without last FC layer
 * @param lin           Last FC layer with revised output features count depending on the number of classes
 * @param loader        Test data loader
 * @param data_size     Test data size
 */
template<typename Dataloader>
void test(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& loader, size_t data_size);

/**
 * @brief Dataset that loads and pre-processes the images and corresponding labels with data augmentation.
 */
class DataAugmentationDataset : public torch::data::Dataset<DataAugmentationDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
    const cv::RNG& rng = nullptr;
public:
    /**
     * @brief Initialize the Data Augmentation Dataset
     *
     * @param list_images   images to load and process
     * @param list_labels   labels mapping of loaded images
     * @param image_size    resize dimension to process images
     * @param rng           random number generator employed to randomize data augmentation
     */
    DataAugmentationDataset(
        std::vector<std::string> list_images, std::vector<int> list_labels, uint64_t image_size, const cv::RNG& rng
    ) : rng(rng) {
        states = process_images(list_images, image_size, this->rng);
        labels = process_labels(list_labels);
        ds_size = states.size();
    }
    DataAugmentationDataset(std::vector<std::string> list_images, std::vector<int> list_labels, uint64_t image_size)
        : rng(cv::RNG())
        , DataAugmentationDataset(list_images, list_labels, image_size) {};

    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return { sample_img.clone(), sample_label.clone() };
    };

    torch::optional<size_t> size() const override {
        return ds_size;
    };
};

#endif // __TRAINING_H__
