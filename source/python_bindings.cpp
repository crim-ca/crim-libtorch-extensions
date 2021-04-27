#include "stdafx.h"  // includes pytorch and extensions
#pragma hdrstop

#ifdef NO_PYTHON
#warning efficientnet python_binding included without python support

#else
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "nn/models/EfficientNet.h"
#include "nn/modules/activation.h"


// module name must match "setup.py" and "CMakeLists.txt"
PYBIND11_MODULE(efficientnet_core, m) {
    py::doc("EfficientNet PyTorch extension with Python/C++ bindings.");

    py::module_ a = m.def_submodule("activation", "Activation functions.");

    auto swish_func = a.def("swish", &swish, "Swish activation function");
    a.def("relu6", &relu6, "ReLU6 activation function");

    py::module_ e = m.def_submodule("efficientnet", "EfficientNet implementation.");

    py::class_<EfficientNetOptions>(/*e*/m, "EfficientNetOptions", py::dynamic_attr())
        .def(py::init<double, double, int64_t, double, double, double, double, int, int, ActivationFunction>(),
            "Initialize EfficientNet hyper-parameters with provided values.",
            py::arg("width_coefficient"),
            py::arg("depth_coefficient"),
            py::arg("image_size"),
            py::arg("dropout_rate"),
            py::arg("drop_connect_rate") = 0.2,
            py::arg("batch_norm_momentum") = 0.99,
            py::arg("batch_norm_epsilon") = 0.001,
            py::arg("depth_divisor") = 8,
            py::arg("min_depth") = -1,
            py::arg("activation") = swish_func
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
        Because PyTorch uses some templates to facilitate the use of shared_ptr using a ModuleWrapper
        (i.e.: when calling PYTORCH_MODULE(Module) to generate from ModuleImpl), we must employ their
        binding function instead of plain py::class_. Otherwise pointer references would be missing and
        template generation will fail due to missing (or rather mismatching) constructors.

        See details in:
            https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/python.h
    */
    //py::class_<EfficientNet>(e, "EfficientNet")  // normally
    torch::python::bind_module<EfficientNet>(e, "EfficientNet")  // using pytorch's binding
        .def(py::init<const EfficientNetOptions&, size_t>(),
            "Initialize EfficientNet with hyper-parameters, output class count and activation function.",
            py::arg("params"), py::arg("num_classes") = 2  // FIXME: move num_classes to params
        )
        /*.def("forward", &EfficientNet::forward, "EfficientNet inference forward pass from input Tensor.",
            py::arg("inputs")
        )
        .def("extract_features", &EfficientNet::extract_features,
            py::arg("inputs")
        )*/
    ;

/*
    py::class_<EfficientNetB0>(e, "EfficientNetB0")
        .def(py::init<size_t>(),
            "initialize",
            py::arg("num_classes")
        )
    ;
    */
}

#endif
