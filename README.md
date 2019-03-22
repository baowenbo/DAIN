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
1. [Training LapSRN](#training-lapsrn)
1. [Training MS-LapSRN](#training-ms-lapsrn)
1. [Third-Party Implementation](#third-party-implementation)

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
- Cuda & Cudnn (we test with Cuda = 9.0 and Cudnn = 7.0)
- PyTorch (the customized depth-aware flow projection and other layers require ATen API in PyTorch1.0.0)


