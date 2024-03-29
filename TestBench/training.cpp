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
#include <random>
#include <set>
#include <thread>
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
    LOGGER(VERBOSE) << "Read Image: [" << location << "]" << std::endl;
    cv::Mat image_raw = cv::imread(location, 1);

    // Data augmentation
    LOGGER(VERBOSE) << "Transform Image" << std::endl;
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

    LOGGER(VERBOSE) << "Create Tensor" << std::endl;
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor = img_tensor.to(at::kFloat)/255.0;

    // following are ImageNet normalization values
    // FIXME: provide CLI input to override those (?)
    const float mean[3] = { 0.485, 0.456, 0.406 };
    const float stddev[3] = { 0.229, 0.224, 0.225 };
    LOGGER(VERBOSE) << "Normalize Tensor" << std::endl;
    for (int i = 0; i < 3; i++)
        img_tensor[0][0] = img_tensor[0][i].sub(mean[i]).div(stddev[i]);

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
DataSamples load_data_from_folder(
    const std::vector<std::string>& folders_path,
    const std::string extension,
    const Label label
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
            #
            LOGGER(DEBUG)
                << "Loading directory: [" << base_name << "] (extension: "
                << extension << ") found samples: " << img_count << std::endl;
            closedir(dir);
        } else {
            LOGGER(WARN) << "Could not open directory: [" << base_name << "]" << std::endl;
            // return EXIT_FAILURE;
        }
    }
    return std::make_pair(list_images, list_labels);
}

void load_data_task(
    const std::vector<std::string>& dir_paths,
    const std::vector<Label>& dir_labels,
    const std::vector<size_t>& indices,
    DataSamples& samples,
    const std::string extension,
    const size_t worker
) {
    for (auto i : indices) {
        LOGGER(VERBOSE) << "Parallel worker (" << worker << ") load (" << i << ")" << std::endl;
        auto data = load_data_from_folder({ dir_paths[i] }, extension, dir_labels[i]);
        samples.first.insert(samples.first.end(), data.first.begin(), data.first.end());
        samples.second.insert(samples.second.end(), data.second.begin(), data.second.end());
    }
}

/// Load data and labels corresponding to images from multiple sub-folders corresponding to respective classes.
DataSamples load_data_from_class_folder_tree(
    const std::string folder_path,
    const std::string extension,
    const size_t workers
) {
    DIR* dir;
    struct dirent *ent;
    std::vector<std::string> subdirs;

    // find all subdirs with potential image samples
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
    std::sort(subdirs.begin(), subdirs.end());
    std::vector<DataSamples> thread_samples(workers);
    Label label_index = 0;

    if (workers <= 1) {
        LOGGER(VERBOSE) << "Loading single thread" << std::endl;
        for (auto sub_dir : subdirs) {
            auto data = load_data_from_folder({ sub_dir }, extension, label_index);
            thread_samples[0].first.insert(thread_samples[0].first.end(), data.first.begin(), data.first.end());
            thread_samples[0].second.insert(thread_samples[0].second.end(), data.second.begin(), data.second.end());
            label_index++;
        }
        return thread_samples[0];
    }

    std::vector<Label> subdir_labels(subdirs.size());
    std::iota(subdir_labels.begin(), subdir_labels.end(), label_index);

    // prepare worker ranges
    LOGGER(VERBOSE) << "Parallel workers: " << workers << std::endl;
    std::vector<std::thread> thread_workers(workers);
    auto offset = subdirs.size() / workers;
    auto remain = subdirs.size() % workers;
    std::vector<size_t> offsets(workers, offset);
    for (auto i = 0; i < remain; ++i)
        offsets[i]++;  // share remainder since each dir can have a lot of samples, don't dump all work on last thread

    // parallel load samples from dirs
    LOGGER(VERBOSE) << "Starting parallel loading..." << std::endl;
    size_t start = 0;
    for (int i = 0; i < workers; i++) {
        LOGGER(VERBOSE) << "Parallel worker " << i
            << ": (start: " << start << " end: " << start + offsets[i] << " amount: " << offsets[i] << ")" << std::endl;
        std::vector<size_t> indices(offsets[i]);
        std::iota(indices.begin(), indices.end(), start);
        LOGGER(VERBOSE) << "Parallel worker " << i
            << ": load (start: " << indices[0] << " end: " << indices[indices.size()-1] << ")" << std::endl;
        thread_workers[i] = std::thread(load_data_task,
            std::ref(subdirs),
            std::ref(subdir_labels),
            std::move(indices),
            std::ref(thread_samples[i]),
            extension,
            i
        );
        start += offsets[i];
    }
    for (auto& thread : thread_workers)
        thread.join();

    // merge results from threads
    LOGGER(VERBOSE) << "Merging parallel worker results..." << std::endl;
    std::vector<std::string> images;
    std::vector<Label> labels;
    std::size_t total_samples = 0;
    for (const auto& sub : thread_samples)
        total_samples += sub.first.size();
    images.reserve(total_samples);
    labels.reserve(total_samples);
    for (const auto& sub : thread_samples) {
        images.insert(images.end(), sub.first.begin(), sub.first.end());
        labels.insert(labels.end(), sub.second.begin(), sub.second.end());
    }
    return std::make_pair(images, labels);
}

/// Randomly picks the specified amount of samples from available ones
DataSamples random_pick(DataSamples& samples, size_t amount, unsigned int seed) {
    DataSamples picked;
    if (samples.first.size() <= amount)
        return samples;
    if (amount == 0)
        return picked;
    auto engine = std::default_random_engine(seed);
    picked.first.reserve(amount);
    picked.second.reserve(amount);
    std::vector<size_t> indices(samples.first.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), engine);
    indices.resize(amount);
    for (auto i : indices) {
        picked.first.push_back(samples.first[i]);
        picked.second.push_back(samples.second[i]);
    }
    return picked;
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
    std::ostringstream oss;
    if (nb_devices > 0)
        oss << "Properties of CUDA devices" << std::endl;
    for (auto i_device = 0; i_device < nb_devices; i_device++) {
        auto prop = at::cuda::getDeviceProperties(i_device);
        oss << "CUDA device " << i_device << std::endl
            << "  Device Name:            " << prop->name << std::endl
            << "  Compute Capabilities:   " << prop->major << "." << prop->minor << std::endl
            << "  Total Available Memory: " << humanizeBytes(prop->totalGlobalMem) << std::endl;
    }
    LOGGER(INFO) << oss.str();
}
