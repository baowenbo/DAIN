# modules/FlowProjectionModule.py
from torch.nn.modules.module import Module
from .DepthFlowProjectionLayer import DepthFlowProjectionLayer #, FlowFillholeLayer

__all__ =['DepthFlowProjectionModule']

class DepthFlowProjectionModule(Module):
    def __init__(self, requires_grad = True):
        super(DepthFlowProjectionModule, self).__init__()
        self.requires_grad = requires_grad
        # self.f = DepthFlowProjectionLayer(requires_grad)

    def forward(self, input1, input2):
        return DepthFlowProjectionLayer.apply(input1, input2,self.requires_grad)

# class FlowFillholeModule(Module):
#     def __init__(self,hole_value = -10000.0):
#         super(FlowFillholeModule, self).__init__()
#         self.f = FlowFillholeLayer()
#
#     def forward(self, input1):
#         return self.f(input1)

    #we actually dont need to write the backward code for a module, since we have

