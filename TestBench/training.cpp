#include "stdafx.h"
#pragma hdrstop

// memory
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <exception>
#include <fstream>
#include <set>
#include <vector>

#include "cuda.h"
#include "cuda_runtime_api.h"

#include <ATen/cuda/CUDAContext.h>
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
    LOGGER(DEBUG) << "Read Image: [" << location << "]" << std::endl;
    cv::Mat image_raw = cv::imread(location, 1);

    // Data augmentation
    LOGGER(DEBUG) << "Transform Image: [" << location << "]" << std::endl;
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
        /*resize*/          true,
        /*rng*/             rng);

    LOGGER(DEBUG) << "Create Tensor: [" << location << "]" << std::endl;
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor = img_tensor.to(at::kFloat)/255.0;

    img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);   //0.485, 0.456, 0.406
    img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);   //0.229, 0.224, 0.225
    img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);


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

/// Process vector of tensors (labels) read from the list of labels
std::vector<torch::Tensor> process_labels(std::vector<Label> list_labels) {
    std::vector<torch::Tensor> labels;
    for(auto it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

/// Load data and labels corresponding to images from given folder(s)
std::pair<std::vector<std::string>, std::vector<Label>> load_data_from_folder(
    std::vector<std::string> folders_path,
    std::string extension,
    Label label
) {
    std::vector<std::string> list_images;
    std::vector<Label> list_labels;
    for (auto path: folders_path) {
        std::replace(path.begin(), path.end(), '\\', '/');
        std::string base_name = path + "/";
        DIR* dir;
        struct dirent *ent;
        if ((dir = opendir(base_name.c_str())) != NULL) {
            size_t img_count = 0;
            while((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                if (filename.length() && filename.find(".") != std::string::npos) {

                    // extract extension and compare lowercase to keep matches
                    std::istringstream iss(filename);
                    std::string s, ext;
                    while (std::getline(iss, s, '.'))
                        ext = s;
                    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
                    if (ext == extension) {
                        list_images.push_back(base_name + ent->d_name);
                        list_labels.push_back(label);
                        img_count++;
                    }
                }
            }
            LOGGER(DEBUG)
                << "Loading directory: [" << base_name << "] (extension: "
                << extension << ") found samples: " << img_count << std::endl;
            closedir(dir);
        } else {
            std::cout << "Could not open directory" << std::endl;
            // return EXIT_FAILURE;
        }
    }
    return std::make_pair(list_images, list_labels);
}

/// Load data and labels corresponding to images from multiple sub-folders.
std::pair<std::vector<std::string>, std::vector<Label>> load_data_from_folder(
    std::string folder_path,
    std::string extension
) {
    DIR* dir;
    struct dirent *ent;
    std::vector<std::string> subdirs;
    if ((dir = opendir(folder_path.c_str())) != NULL) {
        std::vector<std::string> folders;
        while((ent = readdir(dir)) != NULL) {
            if (ent->d_type != DT_DIR)
                continue;
            std::string dirname = ent->d_name;
            if (dirname == ".." || dirname == ".")
                continue;
            std::string dirpath = folder_path + "/" + dirname;
            subdirs.push_back(dirpath);
        }
    } else {
        auto msg = "Invalid directory does not exist [" + folder_path + "]";
        throw std::runtime_error(msg);
    }

    Label label = 0;
    std::vector<std::string> subimages;
    std::vector<Label> sublabels;
    std::sort(subdirs.begin(), subdirs.end());
    for (auto subdir : subdirs) {
        auto data = load_data_from_folder({ subdir }, extension, label);
        subimages.insert(subimages.end(), data.first.begin(), data.first.end());
        sublabels.insert(sublabels.end(), data.second.begin(), data.second.end());
        label++;
    }
    return std::make_pair(subimages, sublabels);
}

/// Counts the number of unique classes using a set of labeled data
size_t count_classes(const std::vector<Label> labels) {
    std::set<Label> unique;
    for (auto label : labels)
        unique.insert(label);
    return unique.size();
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

/// Display human-readable byte size with units
std::string humanizeBytes(size_t bytes) {
    std::ostringstream out;
    out.precision(2);
    out << std::fixed;

    float fBytes = static_cast<float>(bytes);
    if (bytes <= 0) out << "0B";
    else if (bytes >= 1073741824) out << fBytes / 1073741824. << "GiB";
    else if (bytes >= 1048576) out << fBytes / 1048576. << "MiB";
    else if (bytes >= 1024) out << fBytes / 1024. << "KiB";

    return out.str();
};

/// Displays how much memory is avaialble on the machine.
void show_machine_memory() {
    // reference: https://stackoverflow.com/a/2513561/5936364
    size_t free, total;
    #ifdef WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        free = status.ullAvailPhys;
        total = status.ullTotalPhys;
    #else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        long avail = sysconf(_SC_AVPHYS_PAGES);
        free = avail * page_size;
        total = pages * page_size;
    #endif
    float ratio = static_cast<float>(free) / static_cast<float>(total);
    LOGGER(INFO)
        << "Machine memory: free=" << humanizeBytes(free)
        << ", total=" << humanizeBytes(total)
        << std::setprecision(1) << std::fixed
        << " (avail=" << ratio * 100. << "%, used=" << (1. - ratio) * 100. << "%) " << std::endl;
}

/// Displays how much memory is being used by all accessible GPU devices
void show_gpu_memory() {
    if (!torch::cuda::is_available()) {
        LOGGER(INFO) << "No GPU - no memory applicable!" << std::endl;
        return;
    }

    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        float ratio = static_cast<float>(free) / static_cast<float>(total);
        LOGGER(INFO) << "GPU " << id
            << " memory: free=" << humanizeBytes(free)
            << ", total=" << humanizeBytes(total)
            << std::setprecision(1) << std::fixed
            << " (avail=" << ratio * 100. << "%, used=" << (1. - ratio) * 100. << "%) " << std::endl;
    }
}

void show_gpu_properties() {
    if (!torch::cuda::is_available()) {
        LOGGER(INFO) << "No GPU - cannot retrieve properties!" << std::endl;
        return;
    }

    auto nb_devices = torch::cuda::device_count();
    LOGGER(INFO) << "CUDA visible devices count: " << nb_devices << std::endl;
    for (auto i_device = 0; i_device < nb_devices; i_device++) {
        auto prop = at::cuda::getDeviceProperties(i_device);
        LOGGER(INFO) << std::endl // skip log header
                        << "Properties of CUDA device " << i_device << std::endl
                        << "  Device Name:            " << prop->name << std::endl
                        << "  Compute Capabilities:   " << prop->major << "." << prop->minor << std::endl
                        << "  Total Available Memory: " << humanizeBytes(prop->totalGlobalMem) << std::endl;
    }
}
