################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
import h5py
import torch.utils.data as data
import pickle
import numpy as np
import torch
import os, os.path
import math, random
import sys
from skimage.transform import resize
from skimage import io



def make_dataset(list_dir):
    # subgroup_name1 = "/dataset/image_list/"
    file_name = list_dir + "imgs_MD.p"
    file_name_1 = open( file_name, "rb" )
    images_list = pickle.load( file_name_1)
    file_name_1.close()

    file_name_t= list_dir + "targets_MD.p"
    file_name_2 = open( file_name_t, "rb" )
    targets_list = pickle.load(file_name_2)
    file_name_2.close()
    return images_list, targets_list

# test for si-RMSE
class ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, input_height, input_width, transform=None, 
                 loader=None, is_flip = True):
        # load image list from hdf5
        img_list , targets_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        # img_list_1, img_list_2 = selfshuffle_dataset(img_list)
        self.root = root
        self.list_dir = list_dir
        self.img_list = img_list
        self.targets_list = targets_list
        self.transform = transform
        # self.loader = loader
        self.input_height = input_height
        self.input_width = input_width
        self.is_flip = is_flip


    def load_MD(self, img_path, depth_path):

        MD_img = np.float32(io.imread(img_path))/255.0

        hdf5_file_read = h5py.File(depth_path,'r')
        gt = hdf5_file_read.get('/depth')
        gt = np.array(gt)

        assert(gt.shape[0] == MD_img.shape[0])
        assert(gt.shape[1] == MD_img.shape[1])

        color_rgb = np.zeros((self.input_height,self.input_width,3))
        MD_img = resize(MD_img, (self.input_height, self.input_width), order = 1)

        if len(MD_img.shape) == 2:
            color_rgb[:,:,0] = MD_img.copy()
            color_rgb[:,:,1] = MD_img.copy()
            color_rgb[:,:,2] = MD_img.copy()
        else:
            color_rgb = MD_img.copy()

        if np.sum(gt > 1e-8) > 10:
            gt[ gt > np.percentile(gt[gt > 1e-8], 98)] = 0
            gt[ gt < np.percentile(gt[gt > 1e-8], 1)] = 0

        max_depth = np.max(gt) + 1e-9
        gt = gt/max_depth
        gt = resize(gt, (self.input_height, self.input_width), order = 0)
        gt = gt*max_depth

        mask = np.float32(gt > 1e-8)

        color_rgb = np.ascontiguousarray(color_rgb)
        gt = np.ascontiguousarray(gt)
        mask = np.ascontiguousarray(mask)

        hdf5_file_read.close()

        return color_rgb, gt, mask

    def __getitem__(self, index):
        # 00xx/1/
        targets_1 = {}
        # targets_1['L'] = []
        targets_1['path'] = []

        img_path_suff = self.img_list[index]
        targets_path_suff = self.targets_list[index]

        img_path = self.root + "/MegaDepth_v1/" + img_path_suff
        depth_path = self.root + "/MegaDepth_v1/" + targets_path_suff

        img, gt, mask = self.load_MD(img_path, depth_path)
        
        gt[mask < 0.1] = 1.0

        targets_1['path'] = targets_path_suff
        targets_1['gt_0'] = torch.from_numpy(gt).float()
        targets_1['mask_0'] = torch.from_numpy(mask).float()

        final_img = torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()

        return final_img, targets_1

    def __len__(self):
        return len(self.img_list)


#  Test for SDR 
class ImageFolder_TEST(data.Dataset):

    def __init__(self, root, list_dir, _input_height, _input_width):
        # load image list from hdf5
        img_list , targets_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.img_list = img_list
        self.input_height = _input_height
        self.input_width = _input_width
        self.half_window = 1

    def load_SfM_ORD(self, img_path, targets_path):

        sfm_image = np.float32(io.imread(img_path))/255.0
        resized_sfm_img = resize(sfm_image, (self.input_height, self.input_width), order = 1)

        color_rgb = np.zeros((self.input_height, self.input_width,3))

        if len(sfm_image.shape) == 2:
            color_rgb[:,:,0] = resized_sfm_img.copy()
            color_rgb[:,:,1] = resized_sfm_img.copy()
            color_rgb[:,:,2] = resized_sfm_img.copy()
        else:
            color_rgb = resized_sfm_img.copy()

        if color_rgb.shape[2] == 4:
            return color_rgb, 0, 0 ,0, 0, 0

        hdf5_file_read = h5py.File(targets_path,'r')
        gt = hdf5_file_read.get('/SfM_features')
        gt = np.array(gt)

        y_A = np.round( gt[0,:] * float(self.input_height) )
        x_A = np.round( gt[1,:] * float(self.input_width) )
        y_B = np.round( gt[2,:] * float(self.input_height) )
        x_B = np.round( gt[3,:] * float(self.input_width) )
        ord_ = gt[4,:]

        hdf5_file_read.close()

        return color_rgb, y_A, x_A ,y_B, x_B, ord_

    def __getitem__(self, index):
        # 00xx/1/
        targets_1 = {}
        # targets_1['L'] = []
        targets_1['path'] = []
        targets_1['sdr_xA'] = []
        targets_1['sdr_yA'] = []
        targets_1['sdr_xB'] = []
        targets_1['sdr_yB'] = []
        targets_1['sdr_gt'] = []

        img_path_suff = self.img_list[index]
        img_path = self.root + "/MegaDepth_v1/" + img_path_suff
        folder_name = img_path_suff.split('/')[-4]
        img_name = img_path_suff.split('/')[-1]
        sparse_sift_path = self.root + "/sparse_features/" + folder_name + "/" + img_name + ".h5"

        # no sift features
        if not os.path.isfile(sparse_sift_path) or not os.path.isfile(img_path):

            img = np.zeros((self.input_height, self.input_width,3))
            targets_1['has_SfM_feature'] = False

        else:

            img, y_A, x_A ,y_B, x_B, ordinal = self.load_SfM_ORD(img_path, sparse_sift_path)

            targets_1['sdr_xA'].append(torch.from_numpy(x_A).long())
            targets_1['sdr_yA'].append(torch.from_numpy(y_A).long())
            targets_1['sdr_xB'].append(torch.from_numpy(x_B).long())
            targets_1['sdr_yB'].append(torch.from_numpy(y_B).long())
            targets_1['sdr_gt'].append(torch.from_numpy(ordinal).float())
            targets_1['has_SfM_feature'] = True

        final_img = torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()


        return final_img, targets_1



    def __len__(self):
        return len(self.img_list)



