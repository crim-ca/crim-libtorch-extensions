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
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[], *args, **kwargs)
        self.sources = [os.path.abspath(s) for s in sources]


class CMakeBuild(build_ext):

    def __init__(self, *args, **kwargs):
        super(CMakeBuild, self).__init__(*args, **kwargs)
        self.python_exe = subprocess.check_output(["which", "python"]).decode().strip()
        self.pytorch_dir = None
        self.cmake = None

    @property
    def cmake(self):
        if self._cmake is None:
            cmake_bin = os.getenv("CMAKE_EXECUTABLE", "cmake")
            cmake_bin = subprocess.check_output(["which", cmake_bin]).decode().strip()
            print("CMAKE_EXECUTABLE:", cmake_bin)
            self._cmake = cmake_bin
        return self._cmake

    @cmake.setter
    def cmake(self, cmake):
        self._cmake = cmake

    @staticmethod
    def find_torch_dir():
        """
        Attempt finding precompiled Torch with ``TORCH_DIR``, ``TORCH_LIBRARY`` or revert back to PyPI package install.
        """
        pytorch_dir = os.getenv("TORCH_DIR")
        pytorch_lib = os.getenv("TORCH_LIBRARY")
        pytorch_lib_path = "lib/libtorch.so" if platform.system() != "Windows" else "lib/x64/torch.lib"
        if pytorch_dir and os.path.isdir(pytorch_dir) and os.path.isfile(os.path.join(pytorch_dir, pytorch_lib_path)):
            pytorch_lib = os.path.join(pytorch_dir, pytorch_lib_path)
        elif pytorch_lib and os.path.isfile(pytorch_lib) and os.path.isdir(pytorch_lib.replace(pytorch_lib_path, "")):
            pytorch_dir = pytorch_lib.replace(pytorch_lib_path, "")
        else:
            try:
                import torch
                pytorch_dir = os.path.dirname(torch.__file__)
                pytorch_lib = os.path.join(pytorch_dir, pytorch_lib_path)
            except ImportError:
                sys.stderr.write("Pytorch is required to build this package\n")
                sys.exit(-1)
        if not os.path.isdir(pytorch_dir) or not os.path.isfile(pytorch_lib):
            sys.stderr.write("Pytorch is required to build this package. "
                             "Set TORCH_DIR for pre-compiled from sources, or install with pip.\n")
        print("Found PyTorch dir:", pytorch_dir)
        return pytorch_dir

    def run(self):
        try:
            _ = subprocess.check_output([self.cmake, "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        self.pytorch_dir = self.find_torch_dir()
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_path))
        build_dir = os.path.join(ext_dir, self.build_temp)
        print("Extension Path:", ext_path)
        print("Extension Dir:", ext_dir)
        print("Ext Build Path:", self.build_temp)
        print("Ext Build Dir:", build_dir)

        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
                      "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
                      "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
                      "-DTORCH_DIR={}".format(self.pytorch_dir),
                      "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # for kenlm - avoid seg fault
                      # "-DPYTHON_EXECUTABLE=".format(sys.executable),
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
        cwd = os.getcwd()
        os.chdir(build_dir)
        self.spawn([self.cmake, " ".join(ext.sources)] + cmake_args)
        jobs = os.getenv("CMAKE_JOBS", multiprocessing.cpu_count())
        cmd = [self.cmake, "--build", build_dir, "--", "-j{}".format(jobs)]
        if not self.dry_run:
            self.spawn(cmd)
        os.chdir(cwd)


with open("version.txt") as ver:
    version = ver.readline().strip()

# package will be available as import with that name
# any submodules are defined on the C++ side by pybind11
TORCH_EXTENSION_NAME = "efficientnet_libtorch"
setup(
    name=TORCH_EXTENSION_NAME,
    version=version,
    description="Implentation of extensions with PyTorch C++ (libtorch) and Python bindings.",
    author="CRIM",
    ext_modules=[
        CMakeExtension(
            name=TORCH_EXTENSION_NAME,
            sources=["."],
            extra_compile_args=[],
        )
    ],
    cmdclass={
        "build_ext": CMakeBuild
    }
)
