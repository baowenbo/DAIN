# MegaDepth: Learning Single-View Depth Prediction from Internet Photos

This is a code of the algorithm described in "MegaDepth: Learning Single-View Depth Prediction from Internet Photos, Z. Li and N. Snavely, CVPR 2018". The code skeleton is based on "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix". If you use our code or models for academic purposes, please consider citing:

    @inproceedings{MDLi18,
	  	title={MegaDepth: Learning Single-View Depth Prediction from Internet Photos},
	  	author={Zhengqi Li and Noah Snavely},
	  	booktitle={Computer Vision and Pattern Recognition (CVPR)},
	  	year={2018}
	}

#### Examples of single-view depth predictions on the photos we randomly downloaded from Internet:
<img src="https://github.com/lixx2938/MegaDepth/blob/master/demo.jpg" width="300"/> <img src="https://github.com/lixx2938/MegaDepth/blob/master/demo.png" width="300"/>
<img src="https://github.com/lixx2938/MegaDepth/blob/master/demo_img/demo_2.jpg" width="300"/> <img src="https://github.com/lixx2938/MegaDepth/blob/master/demo_img/demo_2.png" width="300"/>
<img src="https://github.com/lixx2938/MegaDepth/blob/master/demo_img/demo_3.jpg" width="300"/> <img src="https://github.com/lixx2938/MegaDepth/blob/master/demo_img/demo_3.png" width="300"/>
<img src="https://github.com/lixx2938/MegaDepth/blob/master/demo_img/demo_4.jpg" width="300"/> <img src="https://github.com/lixx2938/MegaDepth/blob/master/demo_img/demo_4.png" width="300"/>

#### Dependencies:
* The code was written in Pytorch 0.2 and Python 2.7, but it should be easy to adapt it to Python 3 and latest Pytorch version if needed.
* You might need skimage, h5py libraries installed for python before running the code.

#### Single-view depth prediction on any Internet photo:
* Download pretrained model from: http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth and put it in "checkpoints/test_local/best_generalization_net_G.pth
* In python file "models/HG_model.py", in init function, change to "model_parameters = self.load_network(model, 'G', 'best_generalization')"
* run demo code 
```bash
    python demo.py
```
You should see an inverse depth prediction saved as demo.png from an original photo demo.jpg. If you want to use RGB maps for visualization, like the figures in our paper, you have to install/run semantic segmentation from https://github.com/kazuto1011/pspnet-pytorch trained on ADE20K to mask out sky, because inconsistent depth prediction of unmasked sky will not make RGB visualization resonable.


#### Evaluation on the MegaDepth test splits:
* Download MegaDepth V1 dataset from project website: http://www.cs.cornell.edu/projects/megadepth/.
* Download pretrained model (specific for MD dataset) from http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_vanila_net_G.pth and put it in "checkpoints/test_local/best_vanila_net_G.pth" 
* Download test list files from http://www.cs.cornell.edu/projects/megadepth/dataset/data_lists/test_lists.tar.gz, it should include two folders corresponding to images with landscape and portrait orientations.
* To compute scale invarance RMSE on MD testset, change the variable "dataset_root" in python file "rmse_error_main.py" to the root directory of MegaDepth_v1 folder, and change variable "test_list_dir_l" and "test_list_dir_p" to corresponding folder paths of test lists, and run:
```bash
    python rmse_error_main.py
```
* To compute Structure from Motion Disagreement Rate (SDR), change the variable "dataset_root" in python file "rmse_error_main.py" to the root directory of MegaDepth_v1 folder, and change variable "test_list_dir_l" and "test_list_dir_p" to corresponding folder paths of test lists, and run:
```bash
    python SDR_compute.py
```
* If you want to run our model on arbitrary Internet photos, please download pretrained model from http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth, which has much better generalization ability (qualitatively speaking) to completely unknown scenes.

