# modules/FlowProjectionModule.py
from torch.nn.modules.module import Module
from .minDepthFlowProjectionLayer import minDepthFlowProjectionLayer #, FlowFillholeLayer

__all__ =['minDepthFlowProjectionModule']

class minDepthFlowProjectionModule(Module):
    def __init__(self, requires_grad = True):
        super(minDepthFlowProjectionModule, self).__init__()
        self.requires_grad = requires_grad
        # self.f = minDepthFlowProjectionLayer(requires_grad)

    def forward(self, input1, input2):
        return minDepthFlowProjectionLayer.apply(input1, input2,self.requires_grad)

# class FlowFillholeModule(Module):
#     def __init__(self,hole_value = -10000.0):
#         super(FlowFillholeModule, self).__init__()
#         self.f = FlowFillholeLayer()
#
#     def forward(self, input1):
#         return self.f(input1)

    #we actually dont need to write the backward code for a module, since we have

