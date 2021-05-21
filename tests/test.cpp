#include <fstream>

#include <torch/torch.h>

#include "nn/models/EfficientNet.h"


int main(int argc, const char* argv[]) {

    auto net = std::make_shared<vision::models::EfficientNetB0>(2);
    std::cout << "OK: " << *net.get() << std::endl;
    return 0;
}
