# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import filterinterpolation_cuda as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class FilterInterpolationLayer(Function):
    def __init__(self):
        super(FilterInterpolationLayer,self).__init__()
    @staticmethod
    def forward(ctx, input1,input2,input3):

        assert(input1.is_contiguous())
        assert(input2.is_contiguous())
        assert (input3.is_contiguous())
        # self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it
        # self.input2 = input2.contiguous() # TODO: Note that this is simply a shallow copy?
        # self.input3 = input3.contiguous()

        # if input1.is_cuda:
        #     self.device = torch.cuda.current_device()
        # else:
        #     self.device = -1

        # output =  torch.zeros(input1.size())


        if input1.is_cuda :
            # output = output.cuda()
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            my_lib.FilterInterpolationLayer_gpu_forward(input1, input2, input3, output)
        else:
            output = torch.FloatTensor(input1.data.size())
            my_lib.FilterInterpolationLayer_cpu_forward(input1, input2, input3, output)

        ctx.save_for_backward(input1, input2,input3)
        # the function returns the output to its caller
        return output

    @staticmethod
    def backward(ctx, gradoutput):
        # print("Backward of Filter Interpolation Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        # gradinput1 = torch.zeros(self.input1.size())
        # gradinput2 = torch.zeros(self.input2.size())
        # gradinput3 = torch.zeros(self.input3.size())

        input1, input2, input3= ctx.saved_tensors

        gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()
        gradinput3 = torch.cuda.FloatTensor().resize_(input3.size()).zero_()
        if input1.is_cuda:
            # print("CUDA backward")
            # gradinput1 = gradinput1.cuda(self.device)
            # gradinput2 = gradinput2.cuda(self.device)
            # gradinput3 = gradinput3.cuda(self.device)

            err = my_lib.FilterInterpolationLayer_gpu_backward(input1,input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            err = my_lib.FilterInterpolationLayer_cpu_backward(input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3)
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)

        return gradinput1, gradinput2,gradinput3

# calculate the weights of flow         
class WeightLayer(Function):
    def __init__(self, lambda_e = 10.0/255.0, lambda_v = 1.0, Nw = 3):
        #lambda_e = 10.0 , lambda_v = 1.0,  Nw = 3,
        super(WeightLayer,self).__init__()
        self.lambda_e = lambda_e
        self.lambda_v = lambda_v
        self.Nw = Nw

    # flow1_grad
    def forward(self, input1,input2,input3):

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # ref1 image
        self.input2 = input2.contiguous() # ref2 image
        self.input3 = input3.contiguous()
        # self.flow1_grad = flow1_grad.contiguous() # ref1 flow's grad

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output =  torch.zeros(input1.size(0), 1 , input1.size(2), input1.size(3))

        if input1.is_cuda :
            output = output.cuda()
            err = my_lib.WeightLayer_gpu_forward(input1, input2, input3,
                                                 # flow1_grad,
                                                 output,
                 self.lambda_e,  self.lambda_v, self.Nw
            )
            if err != 0 :
                print(err)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.WeightLayer_cpu_forward(input1, input2, input3,  output,
                 self.lambda_e ,  self.lambda_v, self.Nw
            )
            if err != 0 :
                print(err)

        self.output = output # save this for fast back propagation
        #  the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of WeightLayer Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        gradinput1 = torch.zeros(self.input1.size())
        gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        # gradflow1_grad = torch.zeros(self.flow1_grad.size())
        if self.input1.is_cuda:
            #print("CUDA backward")
            gradinput1 = gradinput1.cuda(self.device)
            gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            # gradflow1_grad = gradflow1_grad.cuda(self.device)

            err = my_lib.WeightLayer_gpu_backward(
                self.input1,self.input2,self.input3, self.output,
                gradoutput,
                gradinput1, gradinput2, gradinput3,
                self.lambda_e,  self.lambda_v, self.Nw
            )
            if err != 0 :
                print(err)

        else:
            #print("CPU backward")
            # print(gradoutput)
            err = my_lib.WeightLayer_cpu_backward(
                    self.input1, self.input2,self.input3, self.output,
                gradoutput,
                gradinput1, gradinput2, gradinput3,
                self.lambda_e, self.lambda_v, self.Nw
                )
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)
        # print("from 1:")
        # print(gradinput3[0,0,...])

        return gradinput1, gradinput2, gradinput3
  
class PixelValueLayer(Function):
    def __init__(self, sigma_d = 3, tao_r = 0.05, Prowindow = 2 ):
        super(PixelValueLayer,self).__init__()
     
        self.sigma_d = sigma_d
        self.tao_r = tao_r #maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input1, input3, flow_weights):

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # ref1 image
        #self.input2 = input2.contiguous() # ref2 image
        self.input3 = input3.contiguous() # ref1 flow
        self.flow_weights = flow_weights.contiguous() # ref1 flow weights

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output = torch.zeros(input1.size())
        

        if input1.is_cuda:
            output = output.cuda()            
            err = my_lib.PixelValueLayer_gpu_forward(
                input1,  input3, flow_weights,   output,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.PixelValueLayer_cpu_forward(
                input1,  input3, flow_weights, output,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)

        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of PixelValueLayer Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        gradinput1 = torch.zeros(self.input1.size())
        #gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        gradflow_weights = torch.zeros(self.flow_weights.size())

        if self.input1.is_cuda:
            # print("CUDA backward")
            gradinput1 = gradinput1.cuda(self.device)
            #gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            gradflow_weights = gradflow_weights.cuda(self.device)

            err = my_lib.PixelValueLayer_gpu_backward(
                self.input1,self.input3, self.flow_weights,
                gradoutput,
                gradinput1,  gradinput3, gradflow_weights,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)

        else:
            #print("CPU backward")
            # print(gradoutput)
            err = my_lib.PixelValueLayer_cpu_backward(
                self.input1,  self.input3, self.flow_weights,
                gradoutput,
                gradinput1,   gradinput3, gradflow_weights,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)
        # print("from 2:")
        # print(gradinput3[0,0,...])
        # print("Image grad:")
        # print(gradinput1[0,:,:4,:4])
        # print("Flow grad:")
        # print(gradinput3[0,:,:4,:4])
        # print("Flow_weights grad:")
        # print(gradflow_weights[0,:,:4,:4])
        return gradinput1,  gradinput3, gradflow_weights

class PixelWeightLayer(Function):
    def __init__(self,threshhold, sigma_d =3, tao_r =0.05, Prowindow = 2 ):
        super(PixelWeightLayer,self).__init__()
        self.threshhold  = threshhold
        self.sigma_d = sigma_d
        self.tao_r = tao_r #maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input3, flow_weights):

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        #self.input1 = input1.contiguous() # ref1 image
        #self.input2 = input2.contiguous() # ref2 image
        self.input3 = input3.contiguous() # ref1 flow
        self.flow_weights = flow_weights.contiguous() # ref1 flow weights

        if input3.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output =  torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)])

        if input3.is_cuda :
            output = output.cuda()            
            err = my_lib.PixelWeightLayer_gpu_forward(
                input3, flow_weights,   output,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.PixelWeightLayer_cpu_forward(
                input3, flow_weights, output,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)

        self.output = output
        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of PixelWeightLayer Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        #gradinput1 = torch.zeros(self.input1.size())
        #gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        gradflow_weights = torch.zeros(self.flow_weights.size())

        if self.input3.is_cuda:
            # print("CUDA backward")
            #gradinput1 = gradinput1.cuda(self.device)
            #gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            gradflow_weights = gradflow_weights.cuda(self.device)

            err = my_lib.PixelWeightLayer_gpu_backward(
                self.input3, self.flow_weights,  self.output,
                gradoutput,
                gradinput3, gradflow_weights,
                self.threshhold,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            err = my_lib.PixelWeightLayer_cpu_backward(
                self.input3, self.flow_weights, self.output,
                gradoutput,
                gradinput3, gradflow_weights,
                self.threshhold,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)
        # print("from 3:")
        # print(gradinput3[0,0,...])

        return gradinput3, gradflow_weights
		
#class ReliableValueLayer(Function):
#    def __init__(self, Nw =3, tao_r =0.05, Prowindow = 2 ):
#        super(ReliableValueLayer,self).__init__()
#     
#        self.Nw = Nw
#        self.tao_r = tao_r #maybe not useable
#        self.Prowindow = Prowindow
#
#    def forward(self, input3, flow_weight1):
#
#        # assert(input1.is_contiguous())
#        # assert(input2.is_contiguous())
#        #self.input1 = input1.contiguous() # ref1 image
#        #self.input2 = input2.contiguous() # ref2 image
#        self.input3 = input3.contiguous() # ref1 flow
#        self.flow_weight1 = flow_weight1.contiguous() # ref1 flow weights
#
#        if input3.is_cuda:
#            self.device = torch.cuda.current_device()
#        else:
#            self.device = -1
#
#        output =  torch.zeros([intpu3.size(0), 1, input3.size(2), input3.size(3)])
#        #output2 =  torch.zeros(input1.size())
#        #weight1 =  torch.zeros(input1.size())
#        #weight2 =  torch.zeros(input1.size())
#        
#
#        if input1.is_cuda :
#            output = output.cuda()            
#            my_lib.ReliableValueLayer_gpu_forward(
#                        input3, flow_weight1, output,
#                        self.sigma_d,    self.tao_r ,  self.Prowindow )
#        else:
#            # output = torch.cuda.FloatTensor(input1.data.size())
#            my_lib.ReliableValueLayer_cpu_forward(
#                        input3, flow_weight1, output,
#                        self.sigma_d,    self.tao_r ,  self.Prowindow )
#
#        # the function returns the output to its caller
#        return output
#
#    #TODO: if there are multiple outputs of this function, then the order should be well considered?
#    def backward(self, gradoutput):
#        # print("Backward of Filter Interpolation Layer")
#        # gradinput1 = input1.new().zero_()
#        # gradinput2 = input2.new().zero_()
#        #gradinput1 = torch.zeros(self.input1.size())
#        #gradinput2 = torch.zeros(self.input2.size())
#        gradinput3 = torch.zeros(self.input3.size())
#        gradflow_weight1 = torch.zeros(self.flow_weight1.size())
#        
#        if self.input1.is_cuda:
#            # print("CUDA backward")
#            #gradinput1 = gradinput1.cuda(self.device)
#            #gradinput2 = gradinput2.cuda(self.device)
#            gradinput3 = gradinput3.cuda(self.device)
#            gradflow_weight1 = gradflow_weight1.cuda(self.device)
#
#            err = my_lib.ReliableValueLayer_gpu_backward(
#                     self.input3, self.flow_weight1, gradoutput, 
#                     gradinput3,    gradflow_weight1,                        
#                    self.sigma_d,    self.tao_r ,  self.Prowindow )
#            if err != 0 :
#                print(err)
#
#        else: 
#            # print("CPU backward")
#            # print(gradoutput)
#            err = my_lib.ReliableValueLayer_cpu_backward(
#                    self.input3,self.flow_weight1, gradoutput, 
#                    gradinput3,    gradflow_weight1,        
#                    self.sigma_d,    self.tao_r ,  self.Prowindow )
#            # print(err)
#            if err != 0 :
#                print(err)
#            # print(gradinput1)
#            # print(gradinput2)
#
#        # print(gradinput1)
#
#        return gradinput3,gradflow_weight1    
class ReliableWeightLayer(Function):
    def __init__(self, threshhold, sigma_d =3, tao_r =0.05, Prowindow = 2 ):
        super(ReliableWeightLayer,self).__init__()

        self.threshhold = threshhold
        self.sigma_d = sigma_d
        self.tao_r = tao_r #maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input3):

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        #self.input1 = input1.contiguous() # ref1 image
        #self.input2 = input2.contiguous() # ref2 image
        self.input3 = input3.contiguous() # ref1 flow
        #self.flow_weight1 = flow_weight1.contiguous() # ref1 flow weights

        if input3.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output =  torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)] )
        #output2 =  torch.zeros(input1.size())
        #weight1 =  torch.zeros(input1.size())
        #weight2 =  torch.zeros(input1.size())

        if input3.is_cuda :
            output = output.cuda()            
            err = my_lib.ReliableWeightLayer_gpu_forward(
                input3, output,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.ReliableWeightLayer_cpu_forward(
                input3, output,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)
        self.output= output # used for inihibiting some unreliable gradients.
        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        #print("Backward of ReliableWeightLayer Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        #gradinput1 = torch.zeros(self.input1.size())
        #gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        #gradflow_weight1 = torch.zeros(self.flow_weight1.size())
        
        if self.input3.is_cuda:
            #print("CUDA backward")
            #gradinput1 = gradinput1.cuda(self.device)
            #gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            #gradflow_weight1 = gradflow_weight1.cuda(self.device)

            err = my_lib.ReliableWeightLayer_gpu_backward(
                 self.input3,   self.output,
                 gradoutput,
                 gradinput3,
                 self.threshhold,
                 self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            err = my_lib.ReliableWeightLayer_cpu_backward(
                self.input3, self.output,
                gradoutput,
                gradinput3,
                self.threshhold,
                self.sigma_d,    self.tao_r ,  self.Prowindow
            )
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)
        # print("from 4:")
        # print(gradinput3[0,0,...])

        return gradinput3