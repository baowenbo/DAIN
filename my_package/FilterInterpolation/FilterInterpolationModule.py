# modules/AdaptiveInterpolationLayer.py
from torch.nn import Module
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
from .FilterInterpolationLayer import FilterInterpolationLayer,WeightLayer, PixelValueLayer,PixelWeightLayer,ReliableWeightLayer

class FilterInterpolationModule(Module):
    def __init__(self):
        super(FilterInterpolationModule, self).__init__()
        # self.f = FilterInterpolationLayer()

    def forward(self, input1, input2, input3):
        return FilterInterpolationLayer.apply(input1, input2, input3)

    #we actually dont need to write the backward code for a module, since we have

#class WeightModule(Module):
#    def __init__(self):
#        super(WeightModule, self).__init__()
#        self.f = WeightLayer()
#
#    def forward(self, input1, input2, input3):
#        return self.f(input1, input2, input3)
class AdaptiveWeightInterpolationModule(Module):
    def __init__(self,  training = False, threshhold = 1e-6,
                 lambda_e = 30.0/255.0, lambda_v = 1.0, Nw = 3.0,
                 sigma_d =1.5,  tao_r = 0.05, Prowindow = 2 ):
        super(AdaptiveWeightInterpolationModule, self).__init__()

        self.calc_weight1 = WeightLayer(lambda_e, lambda_v, Nw )
        self.padder1 = torch.nn.ReplicationPad2d([0, 1 , 0, 1])
        self.interpolate1 = PixelValueLayer(sigma_d, tao_r , Prowindow)
        self.interpolate1_1 = PixelWeightLayer(101* threshhold, sigma_d,tao_r, Prowindow)
        #        self.interpolate_R1 = ReliableValueLayer(Nw, tao_r , Prowindow)
        self.interpolate_R1_1 = ReliableWeightLayer(101* threshhold, sigma_d,tao_r, Prowindow)
        
        self.calc_weight2 = WeightLayer(lambda_e, lambda_v,Nw)
        self.padder2 = torch.nn.ReplicationPad2d([0, 1 , 0, 1])
        self.interpolate2 = PixelValueLayer(sigma_d, tao_r , Prowindow )
        self.interpolate2_1 = PixelWeightLayer(101*threshhold,sigma_d,tao_r, Prowindow)
        #self.interpolate_R2 = ReliableValueLayer(Nw, tao_r , Prowindow)
        self.interpolate_R2_1 = ReliableWeightLayer(101*threshhold, sigma_d,tao_r, Prowindow)

        self.training = training
        self.threshold = threshhold
        return
        #self.lambda_e = lambda_e
        #self.lambda_v = lambda_v
        #self.sigma_d = sigma_d
        #self.Nw = Nw
        #self.tao_r = tao_r #maybe not useable
        #self.Prowindow = Prowindow
        #    lambda_e = self.lambda_e , lambda_v = self.lambda_v,Nw = self.Nw
        #    sigma_d = self.sigma_d,  tao_r = self.tao_r , Prowindow = self.Prowindow 
        #self.sigma_d,    self.tao_r ,  self.Prowindow 


    # input1 ==> ref1 image
    # #input2 ==> ref2 image
    # input3 ==> ref1 flow
    # input4 ==> ref2 flow
    def forward(self, input1, input2, input3, input4):
        epsilon = 1e-6
        #flow1_grad = torch.sum(torch.sqrt(
        #                    (input3[:, :, :-1, :-1] - input3[:, :, 1:, :-1]) ** 2 +
        #                    (input3[:, :, :-1, :-1] - input3[:, :, :-1, 1:]) ** 2 + epsilon * epsilon
        #                ), dim = 1,keepdim =True)
        #flow1_grad = self.padder1(flow1_grad)
        # if input1.is_cuda:
        #     err = gradcheck(self.calc_weight1,(Variable(input1.data,requires_grad=True),
        #                                        Variable(input2 .data,requires_grad=True),
        #                                        Variable(input3.data,requires_grad= True),
        #                                         # Variable(flow1_grad.data,requires_grad=True)
        #                                        ), eps=1e-3)
        #     print(err)
            # pass
            #input1.requires_grad = True
            #input2.requires_grad = True

        flow_weight1 = self.calc_weight1(input1,input2,input3 )
        # if flow1_grad.is_cuda:
            # err = gradcheck(self.interpolate1,(Variable(input1.data,requires_grad=True),
            #                                    Variable(input3.data,requires_grad= True),
            #                                     Variable(flow_weight1.data,requires_grad=True)), eps=1e-3)
            # err = gradcheck(self.interpolate1_1, (Variable(input3.data,requires_grad=True),
            #                                       Variable(flow_weight1.data, requires_grad =True)),eps=1e-3)
            # err = gradcheck(self.interpolate_R1_1,(input3,),eps=1e-3)
            # print(err)
        # print(flow_weight1[0,:,50:100,50:100])
        p1 = self.interpolate1(input1, input3, flow_weight1)
        p1_r,p1_g,p1_b = torch.split(p1,1,dim=1)
        pw1 = self.interpolate1_1(input3, flow_weight1)
        i1_r,i1_g,i1_b = (p1_r)/(pw1+self.threshold),\
                         (p1_g)/(pw1+self.threshold), \
                         (p1_b)/(pw1+self.threshold)
        #if not self.training:
        #    i1_r[pw1<=10*self.threshold], i1_g[pw1<=10*self.threshold], i1_b[pw1<=10*self.threshold] = 0,0,0
        #i1 = torch.cat((i1_r,i1_g,i1_b),dim=1
        #r1 = self.interpolate_R1(input3, flow_weight1)
        r1 = pw1
        rw1 = self.interpolate_R1_1(input3)
        w1 = (r1)/(rw1+self.threshold)
        # if torch.sum(w1 <= 0).cpu().data.numpy()[0] > 0:
        #   pass
            # print("there are holes in i1 :" )
            # print(torch.sum(w1 <= 0))
        #if not self.training:
        #    w1[rw1 <=10*self.threshold] = 0

        # flow2_grad = torch.sum(torch.sqrt(
        #                     (input4[:, :, :-1, :-1] - input4[:, :, 1:, :-1]) ** 2 +
        #                     (input4[:, :, :-1, :-1] - input4[:, :, :-1, 1:]) ** 2 + epsilon * epsilon
        #                 ), dim = 1,keepdim=True)
        # flow2_grad = self.padder2(flow2_grad)

        flow_weight2 = self.calc_weight2(input2,input1,input4)
        p2 = self.interpolate2(input2, input4, flow_weight2)
        p2_r,p2_g,p2_b = torch.split(p2,1,dim=1)
        pw2 = self.interpolate2_1(input4, flow_weight2)
        i2_r,i2_g,i2_b = (p2_r)/(pw2+self.threshold),\
                         (p2_g)/(pw2+self.threshold), \
                         (p2_b)/(pw2+self.threshold)
        #if not self.training:
        #    i2_r[pw2<=10*self.threshold], i2_g[pw2<=10*self.threshold], i2_b[pw2<=10*self.threshold] = 0,0,0
        #i2 = torch.cat((p2[:,0,...] /pw2, p2[:,1,...] /pw2, p2[:,2,...]/pw2),dim=1)
        #r2 = self.interpolate_R2(input4, flow_weight2)
        r2 = pw2
        rw2 = self.interpolate_R2_1(input4)
        w2 = (r2)/(rw2+self.threshold)
        #if torch.sum(w2 <= 0).cpu().data.numpy()[0] > 0:
        #    pass
        #    print("there are holes in i2 :" )
        #    print(torch.sum(w2 <= 0))
        #if not self.training:
        #    w2[rw2 <= 10*self.threshold] = 0
        # i = (i1 * w1 + i2 * w2 )/ (w1 + w2)

        w = w1+w2
        i_r = (i1_r * w1 + i2_r * w2)/ (w + self.threshold) #(w1 + w2)
        i_g = (i1_g * w1 + i2_g * w2)/ (w + self.threshold) #(w1 + w2)
        i_b = (i1_b * w1 + i2_b * w2)/ (w + self.threshold) #(w1 + w2)
        #if torch.sum(w <= 0).cpu().data.numpy()[0] > 0:
        #    print("there are holes in i :")
        #    print(torch.sum(w <= 0))
        if not self.training:
            i_r[w<= 10*self.threshold], i_g[w<=10*self.threshold], i_b[w<=10*self.threshold] = 0,0,0
            w[w <= 10 *self.threshold] = 0
        i = torch.cat((i_r,i_g,i_b),dim=1)
        return i
