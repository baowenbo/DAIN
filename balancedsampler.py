from torch.utils.data.sampler import Sampler
import torch

class RandomBalancedSampler(Sampler):
    """Samples elements randomly, with an arbitrary size, independant from dataset length.
    this is a balanced sampling that will sample the whole dataset with a random permutation.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, epoch_size):
        self.data_size = len(data_source)
        self.epoch_size = epoch_size
        self.index = 0

    def __next__(self):
        if self.index == 0:
            #re-shuffle the sampler
            self.indices = torch.randperm(self.data_size)
        self.index = (self.index+1)%self.data_size
        return self.indices[self.index]

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return min(self.data_size,self.epoch_size) if self.epoch_size>0 else self.data_size

class SequentialBalancedSampler(Sampler):
    """Samples elements dequentially, with an arbitrary size, independant from dataset length.
    this is a balanced sampling that will sample the whole dataset before resetting it.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, epoch_size):
        self.data_size = len(data_source)
        self.epoch_size = epoch_size
        self.index = 0

    def __next__(self):
        self.index = (self.index+1)%self.data_size
        return self.index

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return min(self.data_size,self.epoch_size) if self.epoch_size>0 else self.data_size
