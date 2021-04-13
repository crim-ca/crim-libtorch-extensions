from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="efficientnet",
    description="EfficientNet C++ implementation with PyTorch integration as extension.",
    author="CRIM",
    ext_modules=[
        cpp_extension.CppExtension(
            name="efficientnet_cpp",
            sources=["source/models/EfficientNet.cpp"],
            extra_compile_args=[],
        )
    ],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension
    }
)
