# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import _ext.my_lib as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class SeparableConvLayer(Function):
    def __init__(self,filtersize):
        self.filtersize = filtersize
        super(SeparableConvLayer,self).__init__()

    def forward(self, input1,input2,input3):
        intBatches = input1.size(0)
        intInputDepth = input1.size(1)
        intInputHeight = input1.size(2)
        intInputWidth = input1.size(3)
        intFilterSize = min(input2.size(1), input3.size(1))
        intOutputHeight = min(input2.size(2), input3.size(2))
        intOutputWidth = min(input2.size(3), input3.size(3))

        assert(intInputHeight - self.filtersize == intOutputHeight - 1)
        assert(intInputWidth - self.filtersize == intOutputWidth - 1)
        assert(intFilterSize == self.filtersize)

        assert(input1.is_contiguous() == True)
        assert(input2.is_contiguous() == True)
        assert(input3.is_contiguous() == True)

        output = input1.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it
        self.input2 = input2.contiguous() # TODO: Note that this is simply a shallow copy?
        self.input3 = input3.contiguous()
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        if input1.is_cuda :
            output = output.cuda()
            err = my_lib.SeparableConvLayer_gpu_forward(input1, input2,input3, output)

        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.SeparableConvLayer_cpu_forward(input1, input2,input3, output)
        if err != 0:
            print(err)
        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of Interpolation Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        gradinput1 = torch.zeros(self.input1.size())
        gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        if self.input1.is_cuda:
            # print("CUDA backward")
            gradinput1 = gradinput1.cuda(self.device)
            gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)

            # the input1 image should not require any gradients
            # print("Does input1 requires gradients? " + str(self.input1.requires_grad))

            err = my_lib.SeparableConvLayer_gpu_backward(self.input1,self.input2,self.input3, gradoutput,gradinput1,gradinput2,gradinput3)
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            err = my_lib.SeparableConvLayer_cpu_backward(self.input1, self.input2, self.input3, gradoutput, gradinput1, gradinput2, gradinput3)
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)

        return gradinput1, gradinput2,gradinput3