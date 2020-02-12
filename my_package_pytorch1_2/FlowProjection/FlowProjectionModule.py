# modules/FlowProjectionModule.py
from torch.nn import Module
from .FlowProjectionLayer import FlowProjectionLayer #, FlowFillholeLayer

class FlowProjectionModule(Module):
    def __init__(self, requires_grad = True):
        super(FlowProjectionModule, self).__init__()

        self.f = FlowProjectionLayer(requires_grad)

    def forward(self, input1):
        return self.f(input1)

# class FlowFillholeModule(Module):
#     def __init__(self,hole_value = -10000.0):
#         super(FlowFillholeModule, self).__init__()
#         self.f = FlowFillholeLayer()
#
#     def forward(self, input1):
#         return self.f(input1)

    #we actually dont need to write the backward code for a module, since we have

