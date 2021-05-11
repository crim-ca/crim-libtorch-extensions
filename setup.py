"""
Setup the C++/Python extension interface for EfficientNet.

CMakeExtension:
    https://github.com/artbataev/end2end/blob/master/setup.py
Torch Setup:
    https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension
"""
import multiprocessing
import os
import platform
import subprocess
import sys
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  #, CppExtension


#class CMakeExtension(CUDAExtension):  #(CppExtension):  #(Extension):
    #def __init__(self, name, *args, **kwargs):
        # don't invoke the original build_ext for this special extension
        #super().__init__(name, sources=[], *args, **kwargs)
        # self.sources = [os.path.abspath(s) for s in sources]


class CMakeBuild(BuildExtension):  #(build_ext):

    def __init__(self, *args, **kwargs):
        super(CMakeBuild, self).__init__(*args, **kwargs)
        self.python_exe = sys.executable
        self.pytorch_dir = None
        self.pybind11_dir = None
        self.cmake = None

    @property
    def cmake(self):
        if self._cmake is None:
            cmake_bin = os.getenv("CMAKE_EXECUTABLE", "cmake")
            cmake_bin = subprocess.check_output(["which", cmake_bin]).decode().strip()
            self.announce("CMAKE_EXECUTABLE: {}".format(cmake_bin))
            self._cmake = cmake_bin
        return self._cmake

    @cmake.setter
    def cmake(self, cmake):
        self._cmake = cmake

    def find_torch_dir(self):
        """
        Attempt finding precompiled Torch with ``TORCH_DIR``, ``TORCH_LIBRARY`` or revert back to PyPI package install.
        """
        pytorch_dir = os.getenv("PYTORCH_DIR")
        if not pytorch_dir:
            pytorch_dir = os.getenv("TORCH_DIR")
        pytorch_lib = os.getenv("TORCH_LIBRARY")
        pytorch_lib_path = "lib/libtorch.so" if platform.system() != "Windows" else "lib/x64/torch.lib"
        if pytorch_dir and os.path.isdir(pytorch_dir) and os.path.isfile(os.path.join(pytorch_dir, pytorch_lib_path)):
            pytorch_lib = os.path.join(pytorch_dir, pytorch_lib_path)
        elif pytorch_lib and os.path.isfile(pytorch_lib) and os.path.isdir(pytorch_lib.replace(pytorch_lib_path, "")):
            pytorch_dir = pytorch_lib.replace(pytorch_lib_path, "")
        else:
            try:
                import torch  # noqa
                pytorch_dir = os.path.dirname(torch.__file__)
                pytorch_lib = os.path.join(pytorch_dir, pytorch_lib_path)
            except ImportError:
                sys.stderr.write("Pytorch is required to build this package\n")
                sys.exit(-1)
        if not os.path.isdir(pytorch_dir) or not os.path.isfile(pytorch_lib):
            sys.stderr.write("Pytorch is required to build this package. "
                             "Set TORCH_DIR for pre-compiled from sources, or install with pip.\n")
        self.announce("Found PyTorch dir: {}".format(pytorch_dir))
        return pytorch_dir

    def find_pybind_dir(self):
        pybind_dir =  os.getenv("PYBIND11_DIR", "")
        if not os.path.isdir(pybind_dir):
            raise RuntimeError("Library pybind11 required but not valid: [{}]".format(pybind_dir))
        self.announce("Found PyBind11 dir: {}".format(pybind_dir))
        self.pybind11_dir = pybind_dir
        return self.pybind11_dir

    def run(self):
        try:
            _ = subprocess.check_output([self.cmake, "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        self.pytorch_dir = self.find_torch_dir()
        self.pybind11_dir = self.find_pybind_dir()
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_path))
        build_dir = os.path.join(ext_dir, self.build_temp)
        self.announce("Extension Path: {}".format(ext_path))
        self.announce("Extension Dir:  {}".format(ext_dir))
        self.announce("Ext Build Path: {}".format(self.build_temp))
        self.announce("Ext Build Dir:  {}".format(build_dir))

        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
                      # "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
                      "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
                      "-DWITH_PYTHON=ON",
                      "-DTORCH_DIR={}".format(self.pytorch_dir),
                      "-DPYBIND11_DIR={}".format(self.pybind11_dir),
                      # "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # should be set by FindTorch
                      ]

        config = "Debug" if self.debug else "Release"
        build_args = ["--config", config]

        if platform.system() == "Darwin":
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), ext_dir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + config]

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        env = os.environ.copy()
        # configure/generate
        src = os.path.abspath(".")
        cmd = [self.cmake, "--log-level=DEBUG", src] + cmake_args
        self.announce("Configure with CMake:\n{}".format(cmd))
        subprocess.check_call(cmd, cwd=build_dir, env=env)
        # compile
        if not self.dry_run:
            jobs = os.getenv("CMAKE_JOBS", multiprocessing.cpu_count())
            cmd = [self.cmake, "--build", build_dir, "--", "-j{}".format(jobs)]
            subprocess.check_call(cmd, cwd=build_dir, env=env)


with open("README.md") as f:
    README = f.read()

with open("VERSION") as ver:
    VERSION = ver.readline().strip()

with open("requirements.txt") as r:
    REQUIRES = []
    for line in r.readlines():
        if line.startswith("#"):
            continue
        REQUIRES.append(line.strip())

# package will be available as import with that name
#   submodules are defined on the C++ side by pybind11
#   extension name must match with "python/bindings.cpp" and "CMakeLists.txt"
TORCH_EXTENSION_NAME = "crim_libtorch_extensions"
setup(
    name=TORCH_EXTENSION_NAME,
    version=VERSION,
    description="Extension implementations with PyTorch C++ (libtorch) and Python bindings.",
    long_description=README,
    author="CRIM",
    maintainer="Francis Charette-Migneault",
    maintainer_email="francis.charette-migneault@crim.ca",
    contact="CRIM",
    contact_email="info@crim.ca",
    # license="",  # FIXME: pick a license for publish, also adjust classifier below
    keywords=", ".join([
        "PyTorch",
        "libtorch",
        "extensions",
        "EfficientNet",
        "",
    ]),
    url="https://www.crim.ca/stash/scm/visi/efficientnet-libtorch.git",
    zip_safe=False,
    python_requires=">=3.6, <4",
    install_requires=REQUIRES,
    packages=find_packages(exclude=["tests"]),
    package_data={"": ["*.so"]},
    test_suite="tests",
    # tests_require=TEST_REQUIREMENTS,
    ext_modules=[
        CUDAExtension(
            name=TORCH_EXTENSION_NAME,
            sources=[],
            extra_compile_args=[],
        )
    ],
    cmdclass={
        "build_ext": CMakeBuild
    },
    platforms=[
        "linux_x86_64",
        # "win32"
    ],
    # https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        # "Development Status :: 3 - Alpha",
        # "Development Status :: 4 - Beta",
        "Environment :: Console",   # CLI11 utility
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA ",
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",  # others could work, but this one tested
        "Environment :: Win32 (MS Windows)",
        "Intended Audience :: Developers",
        "Intended Audience :: Education"
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: ??? License"   # MIT or Apache ?,
        "Natural Language :: English",
        # to validate Windows
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
