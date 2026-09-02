from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="transformer_kernels",
    ext_modules=[
        CUDAExtension(
            name="transformer_kernels_cuda",
            sources=[
                "bindings/bindings.cpp",
                "src/softmax/softmax_naive.cu",
            ],
            include_dirs=["include"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)