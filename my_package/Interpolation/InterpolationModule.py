# modules/InterpolationLayer.py
from torch.nn import Module
from .InterpolationLayer import InterpolationLayer

class InterpolationModule(Module):
    def __init__(self):
        super(InterpolationModule, self).__init__()
        # self.f = InterpolationLayer()

    def forward(self, input1, input2):
        return InterpolationLayer.apply(input1, input2)

    #we actually dont need to write the backward code for a module, since we have 

