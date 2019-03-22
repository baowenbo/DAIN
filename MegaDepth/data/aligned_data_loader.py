import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from data.image_folder import ImageFolder_TEST
from builtins import object
import sys
import h5py


class PairedData(object):
    def __init__(self, data_loader, flip):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.flip = flip
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
    

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1

        final_img, target_1 = next(self.data_loader_iter)

        return {'img_1': final_img, 'target_1': target_1}


class AlignedDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, _input_height, _input_width, _is_flip, _shuffle):
        transform = None
        dataset = ImageFolder(root=_root, \
                list_dir =_list_dir, input_height = _input_height, input_width = _input_width, transform=transform, is_flip = _is_flip)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 16, shuffle= _shuffle, num_workers=int(3))

        self.dataset = dataset
        flip = False
        self.paired_data = PairedData(data_loader, flip)

    def name(self):
        return 'RMSEDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)



class AlignedDataLoader_TEST(BaseDataLoader):
    def __init__(self,_root, _list_dir, _input_height, _input_width):

        dataset = ImageFolder_TEST(root=_root, \
                list_dir =_list_dir, _input_height = _input_height, _input_width = _input_width)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle= False, num_workers=int(3))
        self.dataset = dataset
        flip = False
        self.paired_data = PairedData(data_loader, flip)

    def name(self):
        return 'TestSDRDataLoader'

    def load_data(self):
        return self.paired_data


    def __len__(self):
        return len(self.dataset)
