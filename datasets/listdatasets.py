import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import random

def Vimeo_90K_loader(root, im_path, input_frame_size = (3, 256, 448), output_frame_size = (3, 256, 448), data_aug = True):


    root = os.path.join(root,'sequences',im_path)

    if data_aug and random.randint(0, 1):
        path_pre2 = os.path.join(root,  "im1.png")
        path_mid = os.path.join(root,  "im2.png")
        path_pre1 = os.path.join(root,  "im3.png")
    else:
        path_pre1 = os.path.join(root,  "im1.png")
        path_mid = os.path.join(root,  "im2.png")
        path_pre2 = os.path.join(root,  "im3.png")

    im_pre2 = imread(path_pre2)
    im_pre1 = imread(path_pre1)
    im_mid = imread(path_mid)

    h_offset = random.choice(range(256 - input_frame_size[1] + 1))
    w_offset = random.choice(range(448 - input_frame_size[2] + 1))

    im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    im_mid = im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]

    if data_aug:
        if random.randint(0, 1):
            im_pre2 = np.fliplr(im_pre2)
            im_mid = np.fliplr(im_mid)
            im_pre1 = np.fliplr(im_pre1)
        if random.randint(0, 1):
            im_pre2 = np.flipud(im_pre2)
            im_mid = np.flipud(im_mid)
            im_pre1 = np.flipud(im_pre1)

    X0 = np.transpose(im_pre1,(2,0,1))
    X2 = np.transpose(im_pre2, (2, 0, 1))

    y = np.transpose(im_mid, (2, 0, 1))
    return X0.astype("float32")/ 255.0, \
            X2.astype("float32")/ 255.0,\
            y.astype("float32")/ 255.0



class ListDataset(data.Dataset):
    def __init__(self, root, path_list,  loader=Vimeo_90K_loader):

        self.root = root
        self.path_list = path_list
        self.loader = loader

    def __getitem__(self, index):
        path = self.path_list[index]
        # print(path)
        image_0,image_2,image_1 = self.loader(self.root, path)
        return image_0,image_2,image_1

    def __len__(self):
        return len(self.path_list)
