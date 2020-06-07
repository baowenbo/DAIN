#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from compiler_args import nvcc_args, cxx_args

setup(
    name='mindepthflowprojection_cuda',
    ext_modules=[
        CUDAExtension('mindepthflowprojection_cuda', [
            'mindepthflowprojection_cuda.cc',
            'mindepthflowprojection_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
