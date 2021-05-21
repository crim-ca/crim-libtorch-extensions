#include "stdafx.h"  // includes pytorch and extensions
#pragma hdrstop

#ifdef NO_PYTHON
#warning crim_torch_extensions python/bindings included without python support

#else
#include <pybind11/pybind11.h>
#include <pybind11/attr.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include "nn/models/EfficientNet.h"
#include "nn/modules/activation.h"
using namespace torch::nn;
using namespace vision::models;


// module name must match "setup.py" and "CMakeLists.txt"
PYBIND11_MODULE(crim_torch_extensions, module) {
    py::doc("EfficientNet PyTorch extension with Python/C++ bindings.");

    py::module_ mod_nn = module.def_submodule("nn", "Neural Network utilities.");
    py::module_ mod_vision = module.def_submodule("vision", "Vision related utilities.");
    py::module_ mod_vis_models = mod_vision.def_submodule("models", "Vision models.");

    /*
        Activation Functions
    */
    py::module_ mod_activation = mod_nn.def_submodule("activation", "Activation functions.");

    // std::function<torch::Tensor(torch::Tensor)> ActivationFunction
    mod_activation.def("swish", &swish, "Swish activation function");
    mod_activation.def("relu6", &relu6, "ReLU6 activation function");

    /*
        EfficientNet
    */

    py::module_ mod_efficientnet = mod_vis_models.def_submodule("efficientnet",
        R"delim(
            EfficientNet implementation.

            Variant | Width Coefficient | Depth Coefficient | Image Resolution  | Dropout Rate
            -----------------------------------------------------------------------------------
            B0      | 1.0               | 1.0               | 224               | 0.2
            B1      | 1.0               | 1.1               | 240               | 0.2
            B2      | 1.1               | 1.2               | 260               | 0.3
            B3      | 1.2               | 1.4               | 300               | 0.3
            B4      | 1.4               | 1.8               | 380               | 0.4
            B5      | 1.6               | 2.2               | 456               | 0.4
            B6      | 1.8               | 2.6               | 528               | 0.5
            B7      | 2.0               | 3.1               | 600               | 0.5
        )delim"
    );

    // allow override of parameter values with class instance, using 'py::dynamic_attr'
    py::class_<EfficientNetOptions>(mod_efficientnet, "EfficientNetOptions", py::dynamic_attr())
        .def(py::init<double, double, int64_t, double, double, double, double, int, int/*,
                      const py::function ActivationFunction&*/  >(),
            "Initialize EfficientNet hyper-parameters with provided values.",
            py::arg("width_coefficient"),
            py::arg("depth_coefficient"),
            py::arg("image_size"),
            py::arg("dropout_rate"),
            py::arg("drop_connect_rate") = 0.2,
            py::arg("batch_norm_momentum") = 0.99,
            py::arg("batch_norm_epsilon") = 0.001,
            py::arg("depth_divisor") = 8,
            py::arg("min_depth") = -1/*,
            py::arg("activation") = &swish*/
        )
        .def_readwrite("width_coefficient",     &EfficientNetOptions::width_coefficient)
        .def_readwrite("depth_coefficient",     &EfficientNetOptions::depth_coefficient)
        .def("image_size",                      &EfficientNetOptions::image_size)
        .def_readwrite("image_size_w",          &EfficientNetOptions::image_size_w)
        .def_readwrite("image_size_h",          &EfficientNetOptions::image_size_h)
        .def_readwrite("dropout_rate",          &EfficientNetOptions::dropout_rate)
        .def_readwrite("dropout_rate",          &EfficientNetOptions::dropout_rate)
        .def_readwrite("drop_connect_rate",     &EfficientNetOptions::drop_connect_rate)
        .def_readwrite("batch_norm_momentum",   &EfficientNetOptions::batch_norm_momentum)
        .def_readwrite("batch_norm_epsilon",    &EfficientNetOptions::batch_norm_epsilon)
        .def_readwrite("depth_divisor",         &EfficientNetOptions::depth_divisor)
        .def_readwrite("min_depth",             &EfficientNetOptions::min_depth)
        .def_readwrite("activation",            &EfficientNetOptions::activation)
    ;

    /*
        Base EfficientNet bindings

        Because PyTorch uses some templates to facilitate the use of shared_ptr using a ModuleWrapper
        (i.e.: when calling PYTORCH_MODULE(Module) to generate from ModuleImpl), we must employ their
        binding function instead of plain py::class_. Otherwise pointer references would be missing and
        template generation will fail due to missing (or rather mismatching) constructors.

        See details in:
            https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/python.h
    */
    //py::class_<EfficientNet>(e, "EfficientNet")  // normally
    torch::python::bind_module<EfficientNet>(mod_efficientnet, "EfficientNet")  // using pytorch's binding
        .def(py::init<const EfficientNetOptions&, size_t>(),
            "Initialize EfficientNet with hyper-parameters, output class count and activation function.",
            py::arg("params"), py::arg("num_classes") = 2  // FIXME: move num_classes to params
        )
        .def("forward", &EfficientNet::forward, "EfficientNet inference forward pass from input Tensor.",
            py::arg("inputs")
        )
        .def("extract_features", &EfficientNet::extract_features,
            py::arg("inputs")
        )
    ;

    /*
        Specialized EfficientNet B# bindings
    */
    torch::python::bind_module<EfficientNetB0>(mod_efficientnet, "EfficientNetB0")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB1>(mod_efficientnet, "EfficientNetB1")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB2>(mod_efficientnet, "EfficientNetB2")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB3>(mod_efficientnet, "EfficientNetB3")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB4>(mod_efficientnet, "EfficientNetB4")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB5>(mod_efficientnet, "EfficientNetB5")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB6>(mod_efficientnet, "EfficientNetB6")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    torch::python::bind_module<EfficientNetB7>(mod_efficientnet, "EfficientNetB7")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;

    /*
        NFNet
    */
    // TODO: implement NFNet bindings
}

#endif
