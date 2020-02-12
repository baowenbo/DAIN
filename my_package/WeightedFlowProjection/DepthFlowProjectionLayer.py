# this is for wrapping the customized layer
import torch
from torch.autograd import Function
#import _ext.my_lib as my_lib
import depthflowprojection_cuda as my_lib

class DepthFlowProjectionLayer(Function):
    def __init__(self,requires_grad):
        super(DepthFlowProjectionLayer,self).__init__()
        # self.requires_grad = requires_grad

    @staticmethod
    def forward(ctx, input1, input2, requires_grad):
        # print("Depth Aware Flow Projection")
        assert(input1.is_contiguous())
        assert(input2.is_contiguous())
        # self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it
        # self.input2 = input2.contiguous()
        fillhole = 1 if requires_grad == False else 0
        # if input1.is_cuda:
        #     self.device = torch.cuda.current_device()
        # else:
        #     self.device = -1

        # count = torch.zeros(input1.size(0),1,input1.size(2),input1.size(3)) # for accumulating the homography projections
        # output = torch.zeros(input1.size())

        if input1.is_cuda:
            # output = output.cuda()
            # count = count.cuda()
            # print("correct")
            count = torch.cuda.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.DepthFlowProjectionLayer_gpu_forward(input1,input2, count,output, fillhole)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            count = torch.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
            output = torch.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.DepthFlowProjectionLayer_cpu_forward(input1,input2, count, output,fillhole)
        if err != 0:
            print(err)
        # output = output/count # to divide the counter

        # self.count = count #to keep this
        # self.output = output

        ctx.save_for_backward(input1, input2,count,output)
        ctx.fillhole = fillhole

        # print(self.input1[0, 0, :10, :10])
        # print(self.count[0, 0, :10, :10])
        # print(self.input1[0, 0, -10:, -10:])
        # print(self.count[0, 0, -10:, -10:])

        # the function returns the output to its caller
        return output

    @staticmethod
    def backward(ctx, gradoutput):
        # print("Backward of Filter Interpolation Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        # gradinput1 = torch.zeros(self.input1.size())

        input1, input2, count, output = ctx.saved_tensors
        # fillhole = ctx.fillhole

        if input1.is_cuda:
            # print("CUDA backward")
            # gradinput1 = gradinput1.cuda(self.device)
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()

            err = my_lib.DepthFlowProjectionLayer_gpu_backward(input1,input2,
                                                               count, output,
                                                               gradoutput, gradinput1,gradinput2)
            # print(err)
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.FloatTensor().resize_(input2.size()).zero_()
            err = my_lib.DepthFlowProjectionLayer_cpu_backward(input1, input2,
                                                               count, output,
                                                               gradoutput, gradinput1,gradinput2)
            # print(err)
            if err != 0:
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)

        return gradinput1,gradinput2,None
