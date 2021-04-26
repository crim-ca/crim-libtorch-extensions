# EfficientNet LibTorch

EfficientNet model implementation using `libtorch` ([PyTorch][pytorch] C++ sources) with Python/C++ bindings.

[pytorch]: https://github.com/pytorch/pytorch

- [EfficientNet LibTorch](#efficientnet-libtorch)
  - [Build and Install](#build-and-install)
  - [Usage](#usage)
  - [Debugging Compilation Problems](#debugging-compilation-problems)
    - [Missing or Wrong CMake](#missing-or-wrong-cmake)
    - [Failed to compute shorthash](#failed-to-compute-shorthash)


## Build and Install

To install the package, you must first build the depenencies.
This process has been simplified by wrapping the C++ Extension with `CMake` through the `setup.py`.

You will need to define the paths to the relevant libraries compiled for your system:

| Variable           | Description                                               |
| ------------------ | --------------------------------------------------------- |
| PYTORCH_DIR        | Installation path of the Torch C++ library compiled from sources (or precompoled matching your system).  |
| PYBIND11_DIR       | Installation path of PyBind11 library <br> (hint: can reuse PyTorch's `third_party` submodule)           |
| PYTHON_EXECUTABLE  | Path to the Python binary to find depenencies, headers and other references. <br> (RECOMMENDED: use virtual environment, e.g.: `conda`)   |

**Note**
To have GPU-enabled runtime, make sure that PyTorch and EfficientNet libraries all find references to CUDA/cuDNN.

Once the above variables where defined, you must activate your environmenet, and then install the package.

``` shell
conda activate <myenv>

# in (<myenv>)
python setup.py install
```

Installation of the packages in the activated environment will be processed. If problem occurs, refer
to the logs to find missing pieces of information (often it is due to a missing or not found path).

If everything succeeded, you should be able to move on to using the package.


## Usage

Once the package was built and installed, it can be called directly in Python.

You can test that references are found correctly using for exemple the following code:

``` shell
❯ python
Python 3.7.7 (default, May  7 2020, 21:25:33)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.

>>> import efficientnet_libtorch
>>> efficientnet_libtorch.activation.swish
<built-in method swish of PyCapsule object at 0x7f109a3eecf0>
>>>
```

## Debugging Compilation Problems

### Missing or Wrong CMake

If `CMake` cannot be found, following can be defined in your environment.
Using those definitions in a `conda` activate script will avoid to redefine them each time.
(e.g.: `<CONDA_PREFIX>/etc/conda/activate.d/gcc.sh` or other shell script)

``` shell
export CMAKE_EXECUTABLE="/usr/bin/cmake"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-${ENV_DIR}}"
```

### Failed to compute shorthash

This error can sometime occurs when attempting to find CUDA libraries.
(see: https://github.com/pytorch/pytorch/issues/53350)

Simply define the following in your environment:

``` shell
export CUDA_NVRTC_LIB="<CUDA_PATH>/include/nvrtc.h"
```

Where `<CUDA_PATH>` is the same as matches references in `CMake` (e.g.: `/usr/local/cuda-11.2` if using `CUDA 11.2`).
