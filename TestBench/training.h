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

//######################################################################################################################
// Utility definitions

/**
 * @brief Generic sample label definitions.
 */
using Label = int;
/**
 * @brief Generic data sample definitions.
 */
using DataSamples = std::pair<std::vector<std::string>, std::vector<Label>>;
/**
 * @brief Generic model interface with expected shared methods.
 */

#define USE_BASE_MODEL
//#define USE_JIT_MODULE

#ifdef USE_BASE_MODEL
using IModel = IBaseModel;
#else
using IModel = torch::nn::AnyModule;
#endif

//######################################################################################################################

/**
 * @brief Function to return image read at given path location
 *
 * @param location      path where to find the image
 * @param image_size    image resize dimension
 * @param rng           data augmentation random number generator
 * @return              image read as tensor
 */
torch::Tensor read_data(std::string location, uint64_t image_size, cv::RNG& rng);

/**
 * @brief Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as tensor.
 *
 * @param label     number to map the label into tensor
 * @return          label read as tensor
*/
torch::Tensor read_label(Label label);

/**
 * @brief Process vector of tensors (images) read from the list of images in a folder.
 *
 * @param list_images       list of image paths to load and process
 * @param image_size        image resize dimension
 * @param rng               data augmentation random number generator
 * @return                  list of processed images as tensors
 */
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, uint64_t image_size, cv::RNG& rng);

/**
 * @brief Process vector of tensors (labels) read from the list of labels
 *
 * @param list_labels       list of labaels to load
 * @return                  list of labels converted to tensors
 */
std::vector<torch::Tensor> process_labels(std::vector<Label> list_labels);

/**
 * @brief Load data from given folder(s) name(s) (folders_name)
 *
 * @param folders_name      name of folders as a vector to load data from
 * @return                  Returns pair of vectors of string (image locations) and int (respective labels)
 */
std::pair<std::vector<std::string>, std::vector<Label>> load_data_from_folder(std::vector<std::string> folders_name);

/**
 * @brief Dataset that loads and pre-processes the images and corresponding labels with data augmentation.
 */
class DataAugmentationDataset : public torch::data::Dataset<DataAugmentationDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
    cv::RNG& rng;
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
        std::vector<std::string> list_images, std::vector<Label> list_labels, uint64_t image_size, cv::RNG& rng
    ) : rng(rng)
    {
        states = process_images(list_images, image_size, this->rng);
        labels = process_labels(list_labels);
        ds_size = states.size();
    }

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

/**
 * @brief Trains the neural network on our data loader using optimizer.
 *
 * During training, saves checkpoint backups of the model as `model.pt` after every epoch.
 *
 * @tparam Dataloader   Type of data loader employed by the training operation. Derived from Torch Sampler.
 *
 * @param net                   Pre-trained model without last FC layer
 * @param lin                   Last FC layer with revised output features count depending on the number of classes
 * @param data_loader_train     Training sample set data loader
 * @param data_loader_train     Validation sample set data loader
 * @param optimizer             Optimizer to use (e.g.: Adam, SGD, etc.)
 * @param size_train            Size of training dataset
 * @param size_valid            Size of validation dataset
 * @param outlgo                Stream handler to output log details
 * @param max_epochs            Maximum number of training epochs
 */
template<typename Dataloader>
void train(
    #ifdef USE_BASE_MODEL
    #ifdef USE_JIT_MODULE
    std::shared_ptr<torch::jit::script::Module> net,
    #else
    std::shared_ptr<IModel> net,
    #endif
    #else
    IModel net,
    #endif
    /*torch::nn::Linear lin, */
    Dataloader& data_loader_train,
    Dataloader& data_loader_valid,
    std::shared_ptr<torch::optim::Optimizer> optimizer,
    size_t size_train,
    size_t size_valid,
    std::ostream& outlog,
    size_t max_epochs = 2
) {
    float best_accuracy = 0.0;
    size_t batch_index = 0;
    outlog << "Training set size: " << size_train << std::endl;
    outlog << "Validation set size: " << size_valid << std::endl;

    for(size_t epoch=0; epoch<max_epochs; epoch++) {
        float mse = 0;
        float Acc = 0.0;
        float valid_acc = 0.0;

        for(auto& batch: *data_loader_train) {
            auto data = batch.data;
            auto target = batch.target.squeeze();

            // Should be of length: batch_size
            data = data.to(torch::kF32).to(torch::kCUDA);
            target = target.to(torch::kInt64).to(torch::kCUDA);

            //std::vector<torch::jit::IValue> input;
            //input.push_back(data);
            optimizer->zero_grad();
            #ifdef USE_BASE_MODEL
            auto output = net->forward(data);
            #else
            auto output = net.forward(data);
            #endif

            // For transfer learning
            output = output.view({output.size(0), -1});
            /*
            outlog << output <<std::endl;
            output = lin(output);
            */

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

            #ifdef USE_BASE_MODEL
            auto output = net->forward(data);
            #else
            auto output = net.forward(data);
            #endif
            output = output.view({ output.size(0), -1 });
            auto acc = output.argmax(1).eq(target).sum();

            valid_acc += acc.template item<float>();
        }


        mse = mse/float(batch_index); // Take mean of loss
        outlog << "Epoch: " << epoch  << ", " << "MSE: " << mse << ", training accuracy: "
               << Acc / size_train << ", validation accuracy: " << valid_acc / size_valid << std::endl;
        outlog << "** " << mse << " "<< Acc / size_train << " " << valid_acc / size_valid << std::endl;

        /*test(net, data_loader, dataset_size, lin);*/

        if(valid_acc/size_valid > best_accuracy) {
            best_accuracy = valid_acc/size_valid;
            outlog << "Saving model" << std::endl;
            ///net.get().save("model.pt");   need a cast?
            //torch::save(lin, "model_linear.pt");
        }
    }
}

#if 0
/**
 * @brief Evaluate trained network inference on test data
 *
 * @tparam Dataloader   Type of data loader employed by the training operation. Derived from Torch Sampler.
 *
 * @param net           Pre-trained model without last FC layer
 * @param lin           Last FC layer with revised output features count depending on the number of classes
 * @param loader        Test data loader
 * @param data_size     Test data size
 */
template<typename Dataloader>
void test(
    #ifdef USE_BASE_MODEL
    #ifdef USE_JIT_MODULE
    std::shared_ptr<torch::jit::script::Module> net,
    #else
    std::shared_ptr<IModel> net,
    #endif
    #else
    IModel net,
    #endif
    /*torch::nn::Linear lin, */
    Dataloader& loader,
    size_t data_size
) {

    #ifdef USE_BASE_MODEL
    auto pNet = net;
    #else
    auto pNet = net.get();
    #endif
    #if defined(USE_BASE_MODEL) && !defined(USE_JIT_MODULE)
    std::dynamic_pointer_cast<torch::jit::script::Module>(pNet)->eval();
    #else
    pNet->eval();
    #endif

    float Loss = 0, Acc = 0;

    for (const auto& batch : *loader) {
        auto data = batch.data;
        auto targets = batch.target.squeeze();

        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);

        std::vector<torch::jit::IValue> input;
        input.push_back(data);

        #ifdef USE_BASE_MODEL
        auto output = net->forward(input);
        #else
        auto output = pNet.forward(input).toTensor();
        #endif
        output = output.view({output.size(0), -1});
        /*output = lin(output);*/
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }

    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

/**
 * @brief Splits data samples into training and validation sets according to specified ratio
 *
 * @param srcPairs[in]          image and label samples from which to pick items to place into sets
 * @param splitProportion[in]   ratio of training / validation set repartition
 * @param trainingPairs[out]    selected samples part of the training set
 * @param validationsPairs[out] selected samples part of the validation set
 */
void split_data(
    DataSamples srcPairs,
    double splitProportion,
    DataSamples& trainingPairs,
    DataSamples& validationsPairs
);
#endif

#endif // __TRAINING_H__
