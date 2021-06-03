## CHANGES

[Unreleased](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions) (latest)
------------------------------------------------------------------------------------------------------------------------
____________

* Data
  * Data Augmentation utilities for reading and preprocessing images to form a training, validation and test datasets.

* Operators
  * Implementation of `swish` activation function.
  * Reference to `relu6` activation function using `Torch`'s implementation of use by `EfficientNet`.

* Modules
  * Provide `EfficientNet` model first implementations (variants B0 to B7).
  * Provide `NFNet` model first implementations (variants 18, 34, 50, 101, 152).

* Optimizers
  * Provide `SGDAGC` optimizer first implementation.

* Python
  * Bindings for `EfficientNet` models and corresponding activation functions.

* CLI
  * Initial implementation of `TestBench` utility for minimal experiments of the provide models and optimizers.

* CI/CD
  * Definition of `setup.py` and `CMake` utilities to build and install libraries.
