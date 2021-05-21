#include <fstream>
#include <memory>

#include <torch/torch.h>

#include "nn/models/EfficientNet.h"


int main(int argc, const char* argv[]) {
    std::cout << vision::models::EfficientNetB0(2) << std::endl;
    std::cout << vision::models::EfficientNetB1(2) << std::endl;
    std::cout << vision::models::EfficientNetB2(2) << std::endl;
    std::cout << vision::models::EfficientNetB3(2) << std::endl;
    std::cout << vision::models::EfficientNetB4(2) << std::endl;
    std::cout << vision::models::EfficientNetB5(2) << std::endl;
    std::cout << vision::models::EfficientNetB6(2) << std::endl;
    std::cout << vision::models::EfficientNetB7(2) << std::endl;
    return 0;
}
