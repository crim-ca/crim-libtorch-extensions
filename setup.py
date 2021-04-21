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

import torch


class CMakeExtension(Extension):
    def __init__(self, name, sources, *args, **kwargs):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[], *args, **kwargs)
        self.sources = [os.path.abspath(s) for s in sources]


class CMakeBuild(build_ext):
    cmake = None

    def __init__(self, *args, **kwargs):
        super(CMakeBuild, self).__init__(*args, **kwargs)
        self.pytorch_dir = os.path.dirname(torch.__file__)
        self.python_exe = subprocess.check_output(["which", "python"]).decode().strip()
        self.get_cmake()

    def get_cmake(self):
        if self.cmake is None:
            cmake_bin = os.getenv("CMAKE_EXECUTABLE", "cmake")
            cmake_bin = subprocess.check_output(["which", cmake_bin]).decode().strip()
            print("CMAKE_EXECUTABLE:", cmake_bin)
            self.cmake = cmake_bin
        return self.cmake

    def run(self):
        try:
            cmake = self.get_cmake()
            _ = subprocess.check_output([cmake, "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        try:
            import torch
        except ImportError:
            sys.stderr.write("Pytorch is required to build this package\n")
            sys.exit(-1)

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        print("Extension Path:", ext_path)
        print("Ext Build Path:", self.build_temp)
        ext_dir = os.path.abspath(os.path.dirname(ext_path))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
                      "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
                      "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
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

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(ext_dir))
        cmake = self.get_cmake()
        self.spawn([cmake, " ".join(ext.sources)] + cmake_args)
        if not self.dry_run:
            self.spawn([cmake, "--build", self.build_temp, "--", "-j{}".format(multiprocessing.cpu_count())])
        os.chdir(cwd)


setup(
    name="efficientnet",
    description="EfficientNet C++ implementation with PyTorch integration as extension.",
    author="CRIM",
    ext_modules=[
        CMakeExtension(
            name="efficientnet_cpp",
            sources=["."],
            extra_compile_args=[],
        )
    ],
    cmdclass={
        "build_ext": CMakeBuild
    }
)
