# CRIM LibTorch Extensions

Provides multiple algorithm implementation with Python/C++ bindings
as extensions to `libtorch` ([PyTorch][pytorch] C++ sources).

- [Included Extensions](#included-extensions)
- [Build and Install](#build-and-install)
  - [Compile C++ Only](#compile-c-only)
  - [Compile Python Bindings](#compile-python-bindings)
- [Usage](#usage)
- [Development notes](#development-notes)
  - [Precompiled headers](#precompiled-headers)
- [Debugging Compilation Problems](#debugging-compilation-problems)
  - [Missing or Wrong CMake](#missing-or-wrong-cmake)
  - [Failed to compute shorthash](#failed-to-compute-shorthash)
- [Debugging Runtime Errors](#debugging-runtime-errors)
  - [ImportError `generic_type` with Unknown Reference](#importerror-generic_type-with-unknown-reference)
  - [Unrecognized Symbol Error](#unrecognized-symbol-error)

[CLI11]: https://github.com/CLIUtils/CLI11
[pytorch]: https://github.com/pytorch/pytorch
[torchvision]: https://github.com/pytorch/vision

## Included Extensions

- Models:
  - *EfficientNet* (variants: B0-B7 and custom) <br>
    Adaptation from [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

  - *NFNet* - Normalizer-Free Networks (variants: 18, 34, 50, 101, 152 and custom) <br>
    Adaptation from [nfnets-pytorch](https://github.com/vballoli/nfnets-pytorch)

- Optimizers
  - *SGD ADC* - Stochastic Gradient Descent with Adaptive Gradient Clipping <br>
    Adaptation from [nfnets-pytorch](https://github.com/vballoli/nfnets-pytorch)

## Build and Install

The source code provides 2 different installation modes:

- Exclusive C++ library, with optional CLI `TestBench` and `tests` executables
- Python package with module bindings to C++ implementations.

In both cases, there are common settings to define.
These require that you first build the dependencies prior to installing any variant.

You will need to define the below paths to the relevant libraries compiled for your system.
Then, move on to the next sub-section according to the variant you want to compile.

| Variable           | Description                                               |
| ------------------ | --------------------------------------------------------- |
| TORCH_DIR          | Installation path of Torch C++ library compiled from sources (or precompiled matching your system).  |
| TORCHVISION_DIR    | Installation path of TorchVision C++ library compiled from sources (or precompiled matching your system).  |
| OPENCV_DIR         | Installation path of OpenCV C++ library (needed if using `DataAugmentation`, `TestBench` CLI or `tests`) |

**Note**
For backward compatibility, `PYTORCH_DIR` is also used as alias to `TORCH_DIR`.
The `TORCH_DIR` format should be preferred since variable names employed by CMake
within [PyTorch][pytorch] sources use this convention.

**Note**
To have GPU-enabled runtime, make sure that all libraries find references to CUDA/cuDNN.
Any missing references to dependencies along the way will reduce performances of the final result.

### Compile C++ Only

| Variable           | Description                                               |
| ------------------ | --------------------------------------------------------- |
| CLI11_DIR          | Installation path of [CLI11][CLI11] library (required only by `TestBench` CLI) <br>  |

You can then call `CMake` as follows:

```shell
mkdir build
cd build
cmake ..
```

You can pass any missing variables as follows:

```shell
cmake -D<SOME_VAR>=<PATH> ..
```

If you are having problems figuring out where things go wrong, you can try with debug output:

```shell
cmake --log-level=debug ..
```

### Compile Python Bindings

| Variable           | Description                                               |
| ------------------ | --------------------------------------------------------- |
| PYBIND11_DIR       | Installation path of PyBind11 library <br> (hint: can reuse PyTorch's `third_party` submodule)  |
| PYTHON_EXECUTABLE  | Path to the Python binary to find dependencies, headers and other references. <br> (RECOMMENDED: use virtual environment, e.g.: `conda`)   |

**Hint**
Sources of [PyTorch][pytorch] and [TorchVision][torchvision] provide a `setup.py` script that helps build and install
bindings from C++ libraries by automatically wrapping the process with `CMake` and `Ninja`.
A similar procedure is used for extensions in this repository.

Once the above variables where defined, you must activate your environment, and then install the package.
This process has been simplified by wrapping the C++ Extension with `CMake` through the `setup.py`.

``` shell
conda activate <myenv>
python setup.py install
```

To enable debug log outputs, employ the following method:

``` shell
DISTUTILS_DEBUG=1 python setup.py install
```

Installation of the packages in the activated environment will be processed. If any problem occurs, refer
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

>>> import crim_libtorch_extensions
>>> crim_libtorch_extensions.activation.swish
<built-in method swish of PyCapsule object at 0x7f109a3eecf0>
>>>
```

## Development notes

### Precompiled headers

Every `*.cpp` file must start with these two lines (including before any comment):

```c++
#include "stdafx.h"
#pragma hdrstop
```

This ensures the file uses precompiled headers and includes basic required dependencies (`Windows.h`) and defines
when corresponding platform and build options are detected.


## Debugging Compilation Problems

### Missing or Wrong CMake

If `CMake` cannot be found, following can be defined in your environment.
Using those definitions in a `conda` activate script
(e.g.: `<CONDA_PREFIX>/etc/conda/activate.d/gcc.sh` or other shell script)
will avoid having to run this step manually each time.

```shell
# use CONDA_PREFIX that is generated by 'conda activate <myenv>'
export CMAKE_EXECUTABLE="/usr/bin/cmake"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
```

### Failed to compute shorthash

This error can sometime occur when attempting to find CUDA libraries.
(see: https://github.com/pytorch/pytorch/issues/53350)

Simply define the following in your environment:

```shell
export CUDA_NVRTC_LIB="<CUDA_PATH>/include/nvrtc.h"
```

Where `<CUDA_PATH>` is the same as matched references in `CMake` (e.g.: `/usr/local/cuda-11.2` if using `CUDA 11.2`).

## Debugging Runtime Errors

### ImportError `generic_type` with Unknown Reference

Whenever an error in a similar form as the following occurs:

```python
ImportError: generic_type: type "EfficientNet" referenced unknown base type "torch::nn::Module"
```

It means that `torch` was not properly imported *before* importing the library extensions.
Because linking is done dynamically against `torch`, it must always be imported first as follows:

```python
import torch
import crim_libtorch_extensions
```

### Unrecognized Symbol Error

The most common cause of unrecognized symbols at runtime is due to missing linking libraries or inconsistent references.
Most of the automatic resolution of Python bindings against PyTorch definitions is done inplace and as needed.
Therefore, those missing links will be showed only at runtime.

You must make sure that the imported `torch` package (before importing the extensions) are indeed the
same libraries that were used to compile the extensions. There are high chances of incompatibilities and missing
symbols between compilation from different sources. For example, if the library linked against during `import torch`
refers to a package installed via `pip` or `conda`, but extensions were compiled from source [PyTorch][pytorch]
libraries, they will most probably not match.

If the above situation occurs, uninstall any `pip` or `conda` installation. Then, build and install
[PyTorch from Sources](https://github.com/pytorch/pytorch#from-source) following their directives.
This should be not much more than preparing your environment variables to find references and define custom options,
and then call `python setup.py install`. Finally, rebuild and install the extensions with the source libraires using
the same command inside this repository.
