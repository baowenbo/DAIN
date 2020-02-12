# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import flowprojection_cuda as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class FlowProjectionLayer(Function):
    def __init__(self,requires_grad):
        super(FlowProjectionLayer,self).__init__()
        self.requires_grad = requires_grad

    @staticmethod
    def forward(ctx, input1, requires_grad):
        assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        # self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it

        fillhole = 1 if requires_grad == False else 0
        # if input1.is_cuda:
        #     self.device = torch.cuda.current_device()
        # else:
        #     self.device = -1

        # count = torch.zeros(input1.size(0),1,input1.size(2),input1.size(3)) # for accumulating the homography projections
        # output = torch.zeros(input1.size())

        if input1.is_cuda :
            # output = output.cuda()
            # count = count.cuda()
            count = torch.cuda.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.FlowProjectionLayer_gpu_forward(input1, count,output, fillhole)
        else:
            output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.FlowProjectionLayer_cpu_forward(input1, count, output, fillhole)
        if err != 0:
            print(err)
        # output = output/count # to divide the counter

        ctx.save_for_backward(input1, count)
        ctx.fillhole = fillhole
        # self.count = count #to keep this
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

        input1, count, output = ctx.saved_tensors

        if input1.is_cuda:
            # print("CUDA backward")
            # gradinput1 = gradinput1.cuda(self.device)
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.FlowProjectionLayer_gpu_backward(input1, count, gradoutput, gradinput1)
            # print(err)
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            err = my_lib.FlowProjectionLayer_cpu_backward(input1, count,  gradoutput, gradinput1)
            # print(err)
            if err != 0:
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)

        return gradinput1, None

class FlowFillholelayer(Function):
    def __init__(self):
        super(FlowFillholelayer,self).__init__()

    def forward(self, input1):
        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        # count = torch.zeros(input1.size(0),1,input1.size(2),input1.size(3)) # for accumulating the homography projections
        output = torch.zeros(input1.size())

        if input1.is_cuda :
            output = output.cuda()
            # count = count.cuda()
            err = my_lib.FlowFillholelayer_gpu_forward(input1, output)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.FlowFillholelayer_cpu_forward(input1, output)
        if err != 0:
            print(err)
        # output = output/count # to divide the counter

        # self.count = count #to keep this
        # print(self.input1[0, 0, :10, :10])
        # print(self.count[0, 0, :10, :10])
        # print(self.input1[0, 0, -10:, -10:])
        # print(self.count[0, 0, -10:, -10:])

        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    # def backward(self, gradoutput):
    #     # print("Backward of Filter Interpolation Layer")
    #     # gradinput1 = input1.new().zero_()
    #     # gradinput2 = input2.new().zero_()
    #     gradinput1 = torch.zeros(self.input1.size())
    #     if self.input1.is_cuda:
    #         # print("CUDA backward")
    #         gradinput1 = gradinput1.cuda(self.device)
    #         err = my_lib.FlowProjectionLayer_gpu_backward(self.input1, self.count, gradoutput, gradinput1)
    #         # print(err)
    #         if err != 0 :
    #             print(err)
    #
    #     else:
    #         # print("CPU backward")
    #         # print(gradoutput)
    #         err = my_lib.FlowProjectionLayer_cpu_backward(self.input1, self.count,  gradoutput, gradinput1)
    #         # print(err)
    #         if err != 0:
    #             print(err)
    #         # print(gradinput1)
    #         # print(gradinput2)
    #
    #     # print(gradinput1)
    #
    #     return gradinput1