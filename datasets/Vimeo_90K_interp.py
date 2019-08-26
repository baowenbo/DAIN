import os.path
import random
# import glob
import math
from .listdatasets import ListDataset,Vimeo_90K_loader


def make_dataset(root, list_file):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    # the last line is invalid in test set.
    # print("The last sample is : " + raw_im_list[-1])
    raw_im_list = raw_im_list[:-1]
    assert len(raw_im_list) > 0
    random.shuffle(raw_im_list)

    return  raw_im_list

def Vimeo_90K_interp(root, split=1.0, single=False, task = 'interp' ):
    train_list = make_dataset(root,"tri_trainlist.txt")
    test_list = make_dataset(root,"tri_testlist.txt")
    train_dataset = ListDataset(root, train_list, loader=Vimeo_90K_loader)
    test_dataset = ListDataset(root, test_list, loader=Vimeo_90K_loader)
    return train_dataset, test_dataset