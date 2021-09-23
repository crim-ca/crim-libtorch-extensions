## CHANGES

[Unreleased](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions) (latest)
------------------------------------------------------------------------------------------------------------------------
____________

* Fix invalid comparison value when limiting number of samples with ``--max-valid-samples`` option.

[0.5.1](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions/browse?at=refs/tags/0.5.1) (2021-09-02)
------------------------------------------------------------------------------------------------------------------------
____________

* Increase logging level of `TestBench` batch iteration reporting to `INFO` since this information is useful for
  monitoring the current state/progress of long running training epochs.
* Add `LICENSE` file and corresponding metadata in `setup.py`.

[0.5.0](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions/browse?at=refs/tags/0.5.0) (2021-08-04)
------------------------------------------------------------------------------------------------------------------------
____________

* Data
  - Improve data discovery when searching directories for samples by using multi-threaded loops.

* Models
  - `EfficientNet`: replace randomly generated string names of `conv_head` and `conv_stem` Conv2d layers by their
    real names to ensure they can be matched when reloading the model from checkpoint file (error not found otherwise).

* Optimizers
  - Initialize Optimizer options with specified input parameters.

* CLI
  - Provide input options to configure every hyperparameter employed by every supported Optimizer.
  - Add options `--max-train-samples` and `--max-valid-samples` to allow subset selection of batch samples
    to reduce initial training/validation iterations (for testing purposes).
  - Add option `--save-dir` to indicate where to save intermediate epoch model checkpoints and best accuracy one found.
  - Add option `--checkpoint` to attempt reload of a model from checkpoint `.pt` file.
  - Improve options definition with groups, better descriptions and more input validation of values and paths.

* Fixes
  - Remove left over and incorrectly included files from previously deleted `include/models` directory.
  - Add `acout` logging when not using `plog` to ensure that multi-threaded calls don't generate overlaped outputs.

[0.4.0](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions/browse?at=refs/tags/0.4.0) (2021-08-02)
------------------------------------------------------------------------------------------------------------------------
____________

* Data
  - Remove deform condition to avoid `rect` definition smaller than expected.
  - Add data normalization of Tensors against ImageNet default mean and standard deviation values.

* Modules
  - Reinsert missing calls to some layers' forward functions in `EfficientNet`.

* CLI
  - Add normalization, try/catch to training loop.
  - Lower per-sample logging operations of reading/transform/tensor creation to `VERBOSE`
    to provide batch/epoch-only logging using `DEBUG` level.

* Fixes
  - Add `VISION_API` defines for symbol export (support Windows DLL).
  - Add missing header (SGD).
  - Fix missing initializations.
  - Remove problematic `using` directive.
  - Comment out useless code outputs.

[0.3.0](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions/browse?at=refs/tags/0.3.0) (2021-07-15)
------------------------------------------------------------------------------------------------------------------------
____________

* Data
  * Add resize image to data augmentation transform following other operations.

* CLI
  * Implement inline batch data loading to avoid memory error over larger datasets.
  * Improve options hanlding to allow `--version` call without other unncessary inputs in that case.
  * Allow Windows-like flag parameters (eg: `/arch`) when applicable.
  * Use [plog](https://github.com/SergiusTheBest/plog) for better flexibility over logging output and configuration.
  * Add utilities and much more log reporting of memory usage, loaded samples, and operations.

* Fixes
  * Adjust uninitialized dataset loader image size variable causing problems during creation of tensors with resize.
  * Remove leftover duplicate files from previous merge operations.

[0.2.0](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions/browse?at=refs/tags/0.2.0) (2021-06-29)
------------------------------------------------------------------------------------------------------------------------
____________

* Remove duplicate files following merge resolution with other pending branches.
* Fix GNU `dirent.h` incorrectly including Windows-specific header by moving it into dedicated `windows` directory.
* Add utility targets to local `Makefile` to dispatch build, install, and basic test calls.

[0.1.0](https://www.crim.ca/stash/projects/VISI/repos/crim-libtorch-extensions/browse?at=refs/tags/0.1.0) (2021-06-03)
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
