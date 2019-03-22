# DAIN
Depth-Aware Video Frame Interpolation

[Wenbo Bao](https://sites.google.com/view/wenbobao/home)
[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Chao Ma](https://sites.google.com/site/chaoma99/),
Xiaoyun Zhang, 
Zhiyong Gao, 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

IEEE Conference on Computer Vision and Pattern Recognition, Long Beach, CVPR 2019


### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Test Pre-trained Models](#test-pre-trained-models)

### Introduction
We propose the Depth-Aware video frame INterpolation (DAIN) model to explicitly detect the occlusion by exploring the depth cue.
We develop a depth-aware flow projection layer to synthesize intermediate flows that preferably sample closer objects than farther ones.
Our method achieves state-of-the-art performance on the Middlebury dataset.

### Citation
If you find the code and datasets useful in your research, please cite:

    @inproceedings{DAIN,
        author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan}, 
        title     = {Depth-Aware Video Frame Interpolation}, 
        booktitle = {IEEE Conferene on Computer Vision and Pattern Recognition},
        year      = {2019}
    }
    
### Requirements and Dependencies
- Python (we test with Python = 3.6.8 in Anaconda3 = 4.1.1)
- Cuda & Cudnn (we test with Cuda = 9.0 and Cudnn = 7.0)
- PyTorch (the customized depth-aware flow projection and other layers require ATen API in PyTorch = 1.0.0)
- GCC (Compiling PyTorch 1.0.0 extension files (.c/.cu) requires gcc = 4.9.1 and nvcc = 9.0 compilers)
- NVIDIA GPU (we use Titan X (Pascal) with compute = 6.1, but we support compute_50/52/60/61 devices, check [this](https://github.com/baowenbo/DAIN/blob/master/my_package/DepthFlowProjection/setup.py))

### Installation
Download repository:

    $ git clone https://github.com/baowenbo/DAIN.git

Before building Pytorch extensions, be sure you have `pytorch version >= 1.0.0`:
    
    $ python -c "import torch; print(torch.__version__)"
    
Generate our PyTorch extensions:
    
    $ cd DAIN
    $ cd my_package 
    $ ./build.sh

Generate the Correlation package required by [PWCNet](also see https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
    
    $ cd ../PWCNet/correlation_package_pytorch1_0
    $ ./build.sh


### Test Pre-trained Models
Make model weights dir and Middlebury dataset dir:

    $ cd DAIN
    $ mkdir model_weights
    $ mkdir MiddleBurySet
    
Download pretrained models, 

    $ cd model_weights
    $ wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth
    
and Middlebury dataset:
    
    $ cd ../MiddleBurySet
    $ wget http://vision.middlebury.edu/flow/data/comp/zip/other-color-allframes.zip
    $ unzip other-color-allframes.zip
    $ wget http://vision.middlebury.edu/flow/data/comp/zip/other-gt-interp.zip
    $ unzip other-gt-interp.zip
    $ cd ..

We are good to go by:

    $ CUDA_VISIBLE_DEVICES=0 python demo_MiddleBury.py

The interpolated results are under `MiddleBurySet/other-result-author/[random numer]/`.


    
