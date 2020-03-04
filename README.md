# Overview
This is a modification of DAIN that allows the usage of Google Colab and is able to interpolate frame sequences or videos instead of 2 frames. The original code is limited to interpolation between 2 frames and this code includes a simple fix.

[Original File by btahir can be found here.](https://github.com/baowenbo/DAIN/issues/44)
This is a modification by Styler00Dollar.

### Credits for the original DAIN implementation (Depth-Aware Video Frame Interpolation)
[Project](https://sites.google.com/view/wenbobao/dain) **|** [Paper](http://arxiv.org/abs/1904.00830)

[Wenbo Bao](https://sites.google.com/view/wenbobao/home),
[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Chao Ma](https://sites.google.com/site/chaoma99/),
Xiaoyun Zhang, 
Zhiyong Gao, 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

IEEE Conference on Computer Vision and Pattern Recognition, Long Beach, CVPR 2019

This work is developed based on our TPAMI work [MEMC-Net](https://github.com/baowenbo/MEMC-Net), where we propose the adaptive warping layer. Please also consider referring to it.

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Testing Pre-trained Models](#testing-pre-trained-models)
1. [Downloading Results](#downloading-results)
1. [Slow-motion Generation](#slow-motion-generation)
1. [Training New Models](#training-new-models) 

### Introduction
We propose the **D**epth-**A**ware video frame **IN**terpolation (**DAIN**) model to explicitly detect the occlusion by exploring the depth cue.
We develop a depth-aware flow projection layer to synthesize intermediate flows that preferably sample closer objects than farther ones.
Our method achieves state-of-the-art performance on the Middlebury dataset. 
We provide videos [here](https://www.youtube.com/watch?v=-f8f0igQi5I&t=5s).

<!--![teaser](http://vllab.ucmerced.edu/wlai24/LapSRN/images/emma_text.gif)-->

<!--[![teaser](https://img.youtube.com/vi/icJ0WbPsE20/0.jpg)](https://www.youtube.com/watch?v=icJ0WbPsE20&feature=youtu.be)
<!--<iframe width="560" height="315" src="https://www.youtube.com/embed/icJ0WbPsE20" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![teaser](http://vllab1.ucmerced.edu/~wenbobao/DAIN/kart-turn_compare.gif)


<!--哈哈我是注释，不会在浏览器中显示。
Beanbags
https://drive.google.com/open?id=170vdxANGoNKO5_8MYOuiDvoIXzucv7HW
Dimentrodon
https://drive.google.com/open?id=14n7xvb9hjTKqfcr7ZpEFyfMvx6E8NhD_
DogDance
https://drive.google.com/open?id=1YWAyAJ3T48fMFv2K8j8wIVcmQm39cRof
Grove2
https://drive.google.com/open?id=1sJLwdQdL6JYXSQo_Bev0aQMleWacxCsN
Grove3
https://drive.google.com/open?id=1jGj3UdGppoJO02Of8ZaNXqDH4fnXuQ8O
Hydrangea
https://drive.google.com/open?id=1_4kVlhvrmCv54aXi7vZMk3-FtRQF7s0s
MiniCooper
https://drive.google.com/open?id=1pWHtyBSZsOTC7NTVdHTrv1W-dxa95BLo
RubberWhale
https://drive.google.com/open?id=1korbXsGpSgJn7THBHkLRVrJMtCt5YZPB
Urban2
https://drive.google.com/open?id=1v57RMm9x5vM36mCgPy5hresXDZWtw3Vs
Urban3
https://drive.google.com/open?id=1LMwSU0PrG4_GaDjWRI2v9hvWpYwzRKca
Venus
https://drive.google.com/open?id=1piPnEexuHaiAr4ZzWSAxGi1u1Xo_6vPp
Walking
https://drive.google.com/open?id=1CgCLmVC_WTVTAcA_IdWbLqR8MS18zHoa
-->

<p float="middle">
<img src="https://drive.google.com/uc?export=view&id=1YWAyAJ3T48fMFv2K8j8wIVcmQm39cRof" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1CgCLmVC_WTVTAcA_IdWbLqR8MS18zHoa" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1pWHtyBSZsOTC7NTVdHTrv1W-dxa95BLo" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=170vdxANGoNKO5_8MYOuiDvoIXzucv7HW" width="200"/>
</p>

<p float="middle">
<img src="https://drive.google.com/uc?export=view&id=1sJLwdQdL6JYXSQo_Bev0aQMleWacxCsN" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1jGj3UdGppoJO02Of8ZaNXqDH4fnXuQ8O" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1v57RMm9x5vM36mCgPy5hresXDZWtw3Vs" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1LMwSU0PrG4_GaDjWRI2v9hvWpYwzRKca" width="200"/>
</p>

<p float="middle">
<img src="https://drive.google.com/uc?export=view&id=1piPnEexuHaiAr4ZzWSAxGi1u1Xo_6vPp" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1korbXsGpSgJn7THBHkLRVrJMtCt5YZPB" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=1_4kVlhvrmCv54aXi7vZMk3-FtRQF7s0s" width="200"/>
<img src="https://drive.google.com/uc?export=view&id=14n7xvb9hjTKqfcr7ZpEFyfMvx6E8NhD_" width="200"/>
</p>

### Citation
If you find the code and datasets useful in your research, please cite:

    @inproceedings{DAIN,
        author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan}, 
        title     = {Depth-Aware Video Frame Interpolation}, 
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2019}
    }
    @article{MEMC-Net,
         title={MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement},
         author={Bao, Wenbo and Lai, Wei-Sheng, and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
         doi={10.1109/TPAMI.2019.2941941},
         year={2018}
    }

### Contact
[Wenbo Bao](mailto:bwb0813@gmail.com); [Wei-Sheng (Jason) Lai](mailto:phoenix104104@gmail.com)

### License
See [MIT License](https://github.com/baowenbo/DAIN/blob/master/LICENSE)
