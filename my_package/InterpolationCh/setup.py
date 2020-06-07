#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

from nvcc_args import nvcc_args

setup(
    name='interpolationch_cuda',
    ext_modules=[
        CUDAExtension('interpolationch_cuda', [
            'interpolationch_cuda.cc',
            'interpolationch_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
