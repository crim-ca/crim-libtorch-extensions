#include "stdafx.h"  // includes pytorch and extensions
#pragma hdrstop

#include "nn/models/EfficientNet.h"
#include "nn/modules/activation.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::doc("EfficientNet PyTorch extension with Python/C++ bindings.");

    py::module_ a = m.def_submodule("activation", "Activation functions.");

    auto swish_func = a.def("swish", &swish, "Swish activation function");
    a.def("relu6", &relu6, "ReLU6 activation function");

    py::module_ e = m.def_submodule("efficientnet", "EfficientNet implementation.");

    py::class_<EfficientNetParams>(e, "EfficientNetParams", py::dynamic_attr())
        .def(py::init<double, double, int64_t, double>(),
            "Initialize EfficientNet hyper-parameters with provided values."/*,
            py::arg("width_coefficient"), 
            py::arg("depth_coefficient"), 
            py::arg("image_size"), 
            py::arg("dropout_rate"), 
            py::arg("drop_connect_rate") = 0.2, 
            py::arg("batch_norm_momentum") = 0.99, 
            py::arg("batch_norm_epsilon") = 0.001, 
            py::arg("depth_divisor") = 8, 
            py::arg("min_depth") = -1,
            py::arg("activation") = swish_func*/
        )
        .def_readwrite("width_coefficient",     &EfficientNetParams::width_coefficient)
        .def_readwrite("depth_coefficient",     &EfficientNetParams::depth_coefficient)
        .def("image_size",                      &EfficientNetParams::image_size)
        .def_readwrite("image_size_w",          &EfficientNetParams::image_size_w)
        .def_readwrite("image_size_h",          &EfficientNetParams::image_size_h)
        .def_readwrite("dropout_rate",          &EfficientNetParams::dropout_rate)
        .def_readwrite("dropout_rate",          &EfficientNetParams::dropout_rate)
        .def_readwrite("drop_connect_rate",     &EfficientNetParams::drop_connect_rate)
        .def_readwrite("batch_norm_momentum",   &EfficientNetParams::batch_norm_momentum)
        .def_readwrite("batch_norm_epsilon",    &EfficientNetParams::batch_norm_epsilon)
        .def_readwrite("depth_divisor",         &EfficientNetParams::depth_divisor)
        .def_readwrite("min_depth",             &EfficientNetParams::min_depth)
        .def_readwrite("activation",            &EfficientNetParams::activation)
    ;
/*
    py::class_<EfficientNet>(e, "EfficientNet")
        .def(py::init<EfficientNetParams, size_t>(), 
            "Initialize EfficientNet with hyper-parameters, output class count and activation function.",
            py::arg("params"), py::arg("num_classes") = 2  // FIXME: move num_classes to params
        )
        .def("forward", &EfficientNet::forward, "EfficientNet inference forward pass from input Tensor.",
            py::arg("inputs")
        )
        .def("extract_features", &EfficientNet::extract_features,
            py::arg("inputs")
        )
    ;*/
}
