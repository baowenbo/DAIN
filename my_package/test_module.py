# main.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck

#from modules.InterpolationModule import InterpolationModule
#from modules.FilterInterpolationModule import FilterInterpolationModule
#from modules.FlowProjectionModule import FlowProjectionModule
from my_package.DepthFlowProjection import DepthFlowProjectionModule

#from modules.FilterInterpolationModule import AdaptiveWeightInterpolationModule
#from modules.SeparableConvModule import SeparableConvModule
import time
import numpy
#from modules.InterpolationChModule import InterpolationChModule
#from modules.WeigtedFlowProjectionModule import WeightedFlowProjectionModule
#from modules.SeparableConvFlowModule import SeparableConvFlowModule

def test_SeparableConvFlowModule(input1, input2, input3,filtersize):
    FilterInterpolate = SeparableConvFlowModule(filtersize)

    t1 = time.time()

    output = FilterInterpolate(input1, input2, input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    FilterInterpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = FilterInterpolate(input1_cuda, input2_cuda, input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())

    print("Check the forward path between CPU and GPU...", end='\t')
    x = (output_cuda - output.cuda()) *2 / (torch.abs(output_cuda) + torch.abs(output).cuda())
    x = torch.max(torch.abs(x))
    # print(x)

    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(output_cuda - output.cuda()))
    else:
        print("output pass", end='\n')

    # x = (flow_cuda - flow.cuda() ) * 2 / (torch.abs(flow_cuda) + torch.abs(flow).cuda() )
    # x = torch.max(torch.abs(x))
    # # print(x)
    #
    # if (x.cpu().data.numpy()[0] > 1e-6):
    #     print(x)
    # else:
    #     print("flow pass", end='\n')
    #
    print("Check the backward path between CPU and GPU...", end='\t')
    # x = (input1_cuda.grad - input1.grad.cuda()) * 2 /(torch.abs(input1_cuda.grad) + torch.abs(input1.grad).cuda())
    # # y = x.cpu().data.numpy()
    # x = torch.max(torch.abs(x))
    # # print(x)
    #
    # if (x.cpu().data.numpy()[0] > 1e-6):
    #     print(x)
    #     print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    # else:
    #     print("pass", end='\t')

    x = (input2_cuda.grad - input2.grad.cuda()) * 2 /(torch.abs(input2_cuda.grad) + torch.abs(input2.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = (input3_cuda.grad - input3.grad.cuda()) * 2 / (torch.abs(input3_cuda.grad) + torch.abs(input3.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])
    return t2 - t1, t3 - t2

def test_SeparableConvModule(input1, input2, input3,filtersize):
    FilterInterpolate = SeparableConvModule(filtersize)

    t1 = time.time()

    output = FilterInterpolate(input1, input2, input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    t1 = time.time()
    FilterInterpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = FilterInterpolate(input1_cuda, input2_cuda, input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())

    print("Check the forward path between CPU and GPU...", end='\t')
    x = (output_cuda - output.cuda()) *2 / (torch.abs(output_cuda) + torch.abs(output).cuda())
    x = torch.max(torch.abs(x))
    # print(x)

    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass", end='\n')

    print("Check the backward path between CPU and GPU...", end='\t')
    x = (input1_cuda.grad - input1.grad.cuda()) * 2 /(torch.abs(input1_cuda.grad) + torch.abs(input1.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    # print(x)

    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    else:
        print("pass", end='\t')
    x = (input2_cuda.grad - input2.grad.cuda()) * 2 /(torch.abs(input2_cuda.grad) + torch.abs(input2.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = (input3_cuda.grad - input3.grad.cuda()) * 2 / (torch.abs(input3_cuda.grad) + torch.abs(input3.grad).cuda())
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])
    return t2 - t1, t3 - t2


def test_FilterInterpolation(input1,input2,input3):
    FilterInterpolate = FilterInterpolationModule()

    t1 = time.time()

    output = FilterInterpolate(input1, input2, input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad = True)
    t1 = time.time()
    FilterInterpolate.zero_grad()# to clean up the gradient in the last backward

    output_cuda = FilterInterpolate(input1_cuda, input2_cuda ,input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())


    print("Check the forward path between CPU and GPU...", end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    # print(x)

    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass", end='\n')

    print("Check the backward path between CPU and GPU...", end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    # print(x)

    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = input3_cuda.grad - input3.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])
    return t2-t1,t3-t2


def test_InterpolationModule(input1,input2):
    # input1 = Variable(torch.zeros(12,3,64,64).type(torch.FloatTensor))
    # input2 = Variable(torch.rand(12,2,64,64).type(torch.FloatTensor))
    # input1 = Variable(torch.arange(0.0, 12*3*64*256).view(12,3,64,256), requires_grad=True)
    # input2 = Variable(torch.rand(12,2,64,256)*20, requires_grad= True)
    # input2 = Variable(torch.zeros(12,2,64,64))
    # input2 = Variable(torch.ones(12,2,64,64) * (-2.1))
    # input2 = Variable(torch.cat((torch.ones(12,1,64,64) *0.251, torch.zeros(12,1,64,64)),dim=1))
    # input1.data.uniform_()
    # input2.data.uniform_(-5,5)

    Interpolate = InterpolationModule()

    t1 = time.time()

    output = Interpolate(input1,input2)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()


    print("CPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad = True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad = True)
    t1 = time.time()
    output_cuda = Interpolate(input1_cuda,input2_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())


    print("Check the forward path between CPU and GPU...",end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')
    print("Check the backward path between CPU and GPU...",end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')


    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])
    return t2-t1,t3-t2

def test_InterpolationChModule(input1,input2):
    # input1 = Variable(torch.zeros(12,3,64,64).type(torch.FloatTensor))
    # input2 = Variable(torch.rand(12,2,64,64).type(torch.FloatTensor))
    # input1 = Variable(torch.arange(0.0, 12*3*64*256).view(12,3,64,256), requires_grad=True)
    # input2 = Variable(torch.rand(12,2,64,256)*20, requires_grad= True)
    # input2 = Variable(torch.zeros(12,2,64,64))
    # input2 = Variable(torch.ones(12,2,64,64) * (-2.1))
    # input2 = Variable(torch.cat((torch.ones(12,1,64,64) *0.251, torch.zeros(12,1,64,64)),dim=1))
    # input1.data.uniform_()
    # input2.data.uniform_(-5,5)

    Interpolate = InterpolationChModule(input1.size(1))

    t1 = time.time()

    output = Interpolate(input1,input2)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()


    print("CPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad = True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad = True)
    t1 = time.time()
    output_cuda = Interpolate(input1_cuda,input2_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())


    print("Check the forward path between CPU and GPU...",end='\t')
    x = output_cuda - output.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')
    print("Check the backward path between CPU and GPU...",end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')


    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])
    return t2-t1,t3-t2

def test_FlowProjectionModule(input1):
    # input1 = Variable(torch.zeros(12,3,64,64).type(torch.FloatTensor))
    # input2 = Variable(torch.rand(12,2,64,64).type(torch.FloatTensor))
    # input1 = Variable(torch.arange(0.0, 12*3*64*256).view(12,3,64,256), requires_grad=True)
    # input2 = Variable(torch.rand(12,2,64,256)*20, requires_grad= True)
    # input2 = Variable(torch.zeros(12,2,64,64))
    # input2 = Variable(torch.ones(12,2,64,64) * (-2.1))
    # input2 = Variable(torch.cat((torch.ones(12,1,64,64) *0.251, torch.zeros(12,1,64,64)),dim=1))
    # input1.data.uniform_()
    # input2.data.uniform_(-5,5)

    Project = FlowProjectionModule()

    t1 = time.time()

    output = Project(input1)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()


    print("CPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad = True)
    # input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad = True)
    Project = FlowProjectionModule() # regnenerate
    t1 = time.time()
    output_cuda = Project(input1_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())


    print("Check the forward path between CPU and GPU...",end='\t')
    x = output_cuda - output.cuda()
    # print(output_cuda[0, 0, :10, :10])
    # print(output[0, 0, :10, :10])
    # print(x[0, 0, :10, :10])
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')
    print("Check the backward path between CPU and GPU...",end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    # print(input1_cuda[0,0,:10,:10])
    # print(input1[0,0,:10,:10])
    # print(x[0,0,:10,:10])
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(torch.abs(input1_cuda.grad - input1.grad.cuda())))
        print(torch.mean((input1_cuda.grad - input1.grad.cuda())))
    else:
        print("pass",end='\t')
    # x = input2_cuda.grad - input2.grad.cuda()
    # x = torch.max(torch.abs(x))
    # if(x.cpu().data.numpy()[0] > 1e-6):
    #     print(x)
    # else:
    #     print("pass",end='\n')


    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])

    print("\n\n")
    return t2-t1,t3-t2

def test_DepthFlowProjectionModule(input1,input2):
    # input1 = Variable(torch.zeros(12,3,64,64).type(torch.FloatTensor))
    # input2 = Variable(torch.rand(12,2,64,64).type(torch.FloatTensor))
    # input1 = Variable(torch.arange(0.0, 12*3*64*256).view(12,3,64,256), requires_grad=True)
    # input2 = Variable(torch.rand(12,2,64,256)*20, requires_grad= True)
    # input2 = Variable(torch.zeros(12,2,64,64))
    # input2 = Variable(torch.ones(12,2,64,64) * (-2.1))
    # input2 = Variable(torch.cat((torch.ones(12,1,64,64) *0.251, torch.zeros(12,1,64,64)),dim=1))
    # input1.data.uniform_()
    # input2.data.uniform_(-5,5)

    # Project = DepthFlowProjectionModule()

    # t1 = time.time()

    # output = Project(input1,input2)
    # t2 = time.time()

    # output.backward(output.data)
    # t3 = time.time()


    # print("CPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    # temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad = True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad = True)
    Project = DepthFlowProjectionModule(input1_cuda.requires_grad) # regnenerate
    t1 = time.time()
    output_cuda = Project(input1_cuda,input2_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())


    print("Check the forward path between CPU and GPU...",end='\t')
    x = output_cuda - output.cuda()
    # print(output_cuda[0, 0, :10, :10])
    # print(output[0, 0, :10, :10])
    # print(x[0, 0, :10, :10])
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')
    print("Check the backward path between CPU and GPU...",end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    # print(input1_cuda[0,0,:10,:10])
    # print(input1[0,0,:10,:10])
    # print(x[0,0,:10,:10])
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(torch.abs(input1_cuda.grad - input1.grad.cuda())))
        print(torch.mean((input1_cuda.grad - input1.grad.cuda())))
    else:
        print("pass",end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')


    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])

    print("\n\n")
    return t2-t1,t3-t2

def test_WeightedFlowProjectionModule(input1 , input2, input3):
    # input1 = Variable(torch.zeros(12,3,64,64).type(torch.FloatTensor))
    # input2 = Variable(torch.rand(12,2,64,64).type(torch.FloatTensor))
    # input1 = Variable(torch.arange(0.0, 12*3*64*256).view(12,3,64,256), requires_grad=True)
    # input2 = Variable(torch.rand(12,2,64,256)*20, requires_grad= True)
    # input2 = Variable(torch.zeros(12,2,64,64))
    # input2 = Variable(torch.ones(12,2,64,64) * (-2.1))
    # input2 = Variable(torch.cat((torch.ones(12,1,64,64) *0.251, torch.zeros(12,1,64,64)),dim=1))
    # input1.data.uniform_()
    # input2.data.uniform_(-5,5)

    # Project = FlowProjectionModule()
    Project = WeightedFlowProjectionModule(threshold=20.0/255.0,requires_grad=True)

    t1 = time.time()

    output = Project(input1,input2,input3)
    t2 = time.time()

    output.backward(output.data)
    t3 = time.time()


    print("CPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad = True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad = True)
    input3_cuda = Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad = True)
    Project = WeightedFlowProjectionModule(threshold=20.0/255.0, requires_grad=True) # regnenerate
    t1 = time.time()
    output_cuda = Project(input1_cuda,input2_cuda,input3_cuda)
    t2 = time.time()
    output_cuda.backward(output_cuda.data)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2-t1) +"s\t" + str(t3-t2) +"s\t")
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())


    print("Check the forward path between CPU and GPU...",end='\t')
    x = output_cuda - output.cuda()
    # print(output_cuda[0, 0, :10, :10])
    # print(output[0, 0, :10, :10])
    # print(x[0, 0, :10, :10])
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
    else:
        print("pass",end='\n')
    print("Check the backward path between CPU and GPU...",end='\t')
    x = input1_cuda.grad - input1.grad.cuda()
    # print(input1_cuda[0,0,:10,:10])
    # print(input1[0,0,:10,:10])
    # print(x[0,0,:10,:10])
    x = torch.max(torch.abs(x))
    if(x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(torch.abs(input1_cuda.grad - input1.grad.cuda())))
        print(torch.mean((input1_cuda.grad - input1.grad.cuda())))
    else:
        print("pass",end='\t')
    # x = input2_cuda.grad - input2.grad.cuda()
    # x = torch.max(torch.abs(x))
    # if(x.cpu().data.numpy()[0] > 1e-6):
    #     print(x)
    # else:
    #     print("pass",end='\n')


    # print(x[0,0,...])
    # print(x[0,1,...])
    # print(x[0,2,...])
    #
    # print(torch.max(x))
    # print(x[11,2,...])

    print("\n\n")
    return t2-t1,t3-t2

def test_AdaptiveWeightInterpolationModule(input1, input2, input3, input4):
    training = True
    Interpolate = AdaptiveWeightInterpolationModule(training=training)
#gradcheck(Interpolate,)
    t1 = time.time()

    output = Interpolate(input1, input2, input3, input4)
    t2 = time.time()

    if training:
        #output.backward(output.data)
        grad = output.data
        # grad = grad.zero_()
        output.backward(grad)
        print(        input3.grad)
    t3 = time.time()

    print("CPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")

    #
    # print(output)
    # print(input1.grad.size())
    # print(input1.grad)
    # print(output[3,0,...])
    temp = input1.grad

    # input1 = input1.cuda()
    # input2 = input2.cuda()
    # input1_cuda = Variable(torch.arange(0.0, 12*3*64*64).view(12,3,64,64).type(torch.cuda.FloatTensor), requires_grad=True)
    # input2_cuda = Variable((torch.rand(12,2,64,64)*20).type(torch.cuda.FloatTensor), requires_grad= True)
    input1_cuda = Variable(input1.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input2_cuda = Variable(input2.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input3_cuda = Variable(input3.data.type(torch.cuda.FloatTensor), requires_grad=True)
    input4_cuda = Variable(input4.data.type(torch.cuda.FloatTensor), requires_grad=True )
    t1 = time.time()
    Interpolate.zero_grad()  # to clean up the gradient in the last backward

    output_cuda = Interpolate(input1_cuda, input2_cuda, input3_cuda,input4_cuda)
    t2 = time.time()
    if training :
#        output_cuda.backward(output_cuda.data)
        grad = output_cuda.data
#         grad = grad.zero_()
        output_cuda.backward(grad)
    t3 = time.time()
    print("GPU Forward and backward time is : " + str(t2 - t1) + "s\t" + str(t3 - t2) + "s\t")
    #    return
    # print(output_cuda)
    # print(input1_cuda.grad.size())
    # print(input1_cuda.grad)

    # print(output_cuda[3,0,...])
    # print(output[3,0,...]- output_cuda[3,0,...].cpu())

    # print(output_cuda - output.cuda())
    # print(input1_cuda.grad - input1.grad.cuda())

    print("Check the forward path between CPU and GPU...", end='\n')
    x = output_cuda - output.cuda()
    #print(x)
    #print(x>1e-6)
    print("==>total number of difference")
    print(torch.sum(torch.abs(x) > 1e-6))

    x = torch.max(torch.abs(x))
    print("==>max difference value is ")
    print(x)
    print(torch.sum(output_cuda > 1) )
    print(torch.sum(output.cuda() > 1))

    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)

    else:
        print("pass", end='\n')

    if not training:
        return t2 - t1, t3 - t2

    print("Check the backward path between CPU and GPU...", end='\t')
    y = input1_cuda.grad - input1.grad.cuda()
    x = y.cpu().data.numpy()
    #print(x>1e-6)
    x = torch.max(torch.abs(y))
    print(x)


    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input1_cuda.grad - input1.grad.cuda()))
    else:
        print("pass", end='\t')
    x = input2_cuda.grad - input2.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input2_cuda.grad - input2.grad.cuda()))

    else:
        print("pass", end='\t')
    x = input3_cuda.grad - input3.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input3_cuda.grad - input3.grad.cuda()))

    else:
        print("pass", end='\n')

    x = input4_cuda.grad - input4.grad.cuda()
    y = x.cpu().data.numpy()
    x = torch.max(torch.abs(x))
    if (x.cpu().data.numpy()[0] > 1e-6):
        print(x)
        print(torch.mean(input4_cuda.grad - input4.grad.cuda()))

    else:
        print("pass", end='\n')

    return t2 - t1, t3 - t2
#
#
# # input1 = Variable(torch.zeros(12,3,64,64).type(torch.FloatTensor))
# # input2 = Variable(torch.rand(12,2,64,64).type(torch.FloatTensor))
# # B,H,W = 1,16,16
# # B,C,H,W = 2,64,32,32
# # filtersize = 4
# # input1 = Variable(torch.arange(0.0, B * C * H * W).view(B, C ,H,W), requires_grad=True)
# # input2 = Variable(torch.rand(B, 2, H, W), requires_grad=True)
# # input3 = Variable(torch.rand(B, filtersize**2, H, W), requires_grad=True)
# #input2 = Variable(torch.arange(1, 1+ B * 3 * H * W).view(B , 3, H, W), requires_grad=True)
# # input3 = Variable(torch.rand(B, 2, H, W), requires_grad=True)
# # input4 = Variable(torch.rand(B, 2, H,W), requires_grad =True)
# B,C,H,W = 1,3,128,128
# filtersize = 51
# input1 = Variable(torch.arange(0.0, B * C * H * W).view(B, C ,H,W), requires_grad=True)
# input2 = Variable(torch.zeros(B,filtersize,H-filtersize+1,W-filtersize+1),requires_grad = True)
# input3 = Variable(torch.ones(B,filtersize,H-filtersize+1,W-filtersize+1),requires_grad = True)
#
# # input1 = Variable(torch.arange(0.0, B * 3 * H * W).view(B, 3,H,W), requires_grad=True)
# # input2 = Variable(torch.arange(1, 1+ B * 3 * H * W).view(B , 3, H, W), requires_grad=True)
# # input3 = Variable(torch.rand(B, 2, H, W), requires_grad=True)
# # input4 = Variable(torch.rand(B, 2, H,W), requires_grad =True)
# # input2 = Variable(torch.zeros(12,2,64,64),requires_grad = True)
# # input3 = Variable(torch.ones(12,16,64,64),requires_grad = True)
# # input2 = Variable(torch.ones(12,///2,64,64) * (-2.1))
# # input2 = Variable(torch.cat((torch.ones(12,1,64,64) *0.251, torch.zeros(12,1,64,64)),dim=1))
# input1.data.uniform_(0, 1)
# input2.data.uniform_(0, 1)
# input3.data.uniform_(0, 1) # not have to be normalized to 1.0
# # input4.data.uniform_(-1,1)
# #
# #
# # ftimes = []
# # btimes = []
# # for i in range(10):
# #     input1.data.uniform_(0, 1)
# #     input2.data.uniform_(-1, 1)
# #     input3.data.uniform_(0,1)
# #     input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in InterpolationModule
# #     input2 = Variable(input2.clone().data, requires_grad = True)
# #     input3 = Variable(input3.clone().data, requires_grad = True)
# #     ftime, btime = test_FilterInterpolation(input1,input2,input3)
# #     ftimes.append(ftime)
# #     btimes.append(btime)
# #
# # print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) +"s\t" + str(numpy.array(btimes).mean()) +"s\t\n\n\n\n")
# # # nn.LogSoftmax
# # exit(0)
# # ftimes = []
# # btimes = []
# # for i in range(10):
# #     input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in InterpolationModule
# #     input2 = Variable(input2.clone().data, requires_grad = True)
# #     ftime, btime = test_InterpolationModule(input1,input2)
# #     ftimes.append(ftime)
# #     btimes.append(btime)
# #
# # print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) +"s\t" + str(numpy.array(btimes).mean()) +"s\t\n\n\n\n")
# #
# # ftimes = []
# # btimes = []
# # for i in range(10):
# #     input1.data.uniform_(0, 1)
# #     input2.data.uniform_(-16, 17)
# #     input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in InterpolationModule
# #     input2 = Variable(input2.clone().data, requires_grad = True)
# #     ftime, btime = test_InterpolationChModule(input1,input2)
# #     ftimes.append(ftime)
# #     btimes.append(btime)
# #
# # print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) +"s\t" + str(numpy.array(btimes).mean()) +"s\t\n\n\n\n")
# # # nn.LogSoftmax
# # exit(0)
# #
# ftimes = []
# btimes = []
# for i in range(3):
#     input1.data.uniform_(0.0, 1)
#     input2.data.uniform_(1.0/filtersize, 1.1/filtersize)
#     input3.data.uniform_(1.0/filtersize, 1.1/filtersize)  # not have to be normalized to 1.0
#
#     input1 = Variable(input1.clone().data, requires_grad=True)  # to delete the graph in InterpolationModule
#     input2 = Variable(input2.clone().data, requires_grad=True)
#     input3 = Variable(input3.clone().data, requires_grad=True)
#     # ftime, btime = test_SeparableConvModule(input1, input2, input3,filtersize)
#     ftime, btime = test_SeparableConvFlowModule(input1, input2, input3,filtersize)
#     ftimes.append(ftime)
#     btimes.append(btime)
# print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) + "s\t" + str(
#     numpy.array(btimes).mean()) + "s\t")
# exit(0)
#
# #
# # for i in range(10):
# #     input1.data.uniform_(0.14, 0.405)
# #     input2.data.uniform_(0.14, 0.405)
# #     input3.data.uniform_(0.2, 0.501)  # not have to be normalized to 1.0
# #     input4.data.uniform_(0.2, 0.501)
# #
# #     input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in InterpolationModule
# #     input2 = Variable(input2.clone().data, requires_grad = True)
# #     input3 = Variable(input3.clone().data, requires_grad = True)
# #     input4 = Variable(input4.clone().data, requires_grad = True)
# #     ftime,btime = test_AdaptiveWeightInterpolationModule(input1,input2,input3,input4)
# #     ftimes.append(ftime)
# #     btimes.append(btime)
# # print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) +"s\t" + str(numpy.array(btimes).mean()) +"s\t")
#
#
# input1 = Variable(torch.arange(0.0, 12 * 2 * 64 * 64).view(12, 2, 64, 64), requires_grad=True)
# input1.data.uniform_(-1.0,1.0)
# # input1 = Variable( - 0.5 * torch.ones(12,2,64,64).type(torch.FloatTensor), requires_grad = True)
#
#
#

B,C,H,W = 1,2,512,704
input1 = Variable(torch.arange(0.0, B*C * H * W).view(B, C, H, W), requires_grad=True)
input3 = Variable(torch.arange(0.0, B* 3 * H * W).view(B,3, H,W), requires_grad = True)
# input2 = Variable(torch.arange(0.0, B * 3 * H * W).view(B, 3 ,H,W), requires_grad=True)
input2 = Variable(torch.arange(0.0, B * 1 * H * W).view(B, 1 ,H,W), requires_grad=True)



ftimes = []
btimes = []
for i in range(10):
    input1.data.uniform_(-1.0, 1.0)
    input2.data.uniform_(0.1, 1.0) # must be larger than zero
    # input3.data.uniform_(0.0, 1.0)
    input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in InterpolationModule
    input2 = Variable(input2.clone().data, requires_grad = True)
    # ftime, btime = test_FlowProjectionModule(input1)
    ftime,btime  =test_DepthFlowProjectionModule(input1,input2)
    ftimes.append(ftime)
    btimes.append(btime)

print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) +"s\t" + str(numpy.array(btimes).mean()) +"s\t\n\n\n\n")


exit(0)



ftimes = []
btimes = []
for i in range(10):
    input1 = Variable(input1.clone().data, requires_grad = True) # to delete the graph in InterpolationModule

    input2 = Variable(input2.clone().data, requires_grad = True)
    input3 = Variable(input3.clone().data, requires_grad = True)
    ftime, btime = test_WeightedFlowProjectionModule(input1,input2,input3)
    ftimes.append(ftime)
    btimes.append(btime)

print("GPU Forward and backward time is : " + str(numpy.array(ftimes).mean()) +"s\t" + str(numpy.array(btimes).mean()) +"s\t\n\n\n\n")
