#include <fstream>
#include <memory>

#include <torch/torch.h>

#include "nn/models/NFNet.h"


int main(int argc, const char* argv[]) {
    std::cout << vision::models::NFNet18(2) << std::endl;
    std::cout << vision::models::NFNet34(2) << std::endl;
    std::cout << vision::models::NFNet50(2) << std::endl;
    std::cout << vision::models::NFNet101(2) << std::endl;
    std::cout << vision::models::NFNet152(2) << std::endl;
    return 0;
}
