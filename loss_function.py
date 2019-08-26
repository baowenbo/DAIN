import sys
import os

import sys
import  threading
import torch
from torch.autograd import Variable
from lr_scheduler import *
from torch.autograd import gradcheck

import numpy




def charbonier_loss(x,epsilon):
    loss = torch.mean(torch.sqrt(x * x + epsilon * epsilon))
    return loss
def negPSNR_loss(x,epsilon):
    loss = torch.mean(torch.mean(torch.mean(torch.sqrt(x * x + epsilon * epsilon),dim=1),dim=1),dim=1)
    return torch.mean(-torch.log(1.0/loss) /100.0)

def tv_loss(x,epsilon):
    loss = torch.mean( torch.sqrt(
        (x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 +
        (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2 + epsilon *epsilon
            )
        )
    return loss

    
def gra_adap_tv_loss(flow, image, epsilon):
    w = torch.exp( - torch.sum(	torch.abs(image[:,:,:-1, :-1] - image[:,:,1:, :-1]) + 
                            torch.abs(image[:,:,:-1, :-1] - image[:,:,:-1, 1:]), dim = 1))		
    tv = torch.sum(torch.sqrt((flow[:, :, :-1, :-1] - flow[:, :, 1:, :-1]) ** 2 + (flow[:, :, :-1, :-1] - flow[:, :, :-1, 1:]) ** 2 + epsilon *epsilon) ,dim=1)             
    loss = torch.mean( w * tv )
    return loss	
        
def smooth_loss(x,epsilon):
    loss = torch.mean(
        torch.sqrt(
            (x[:,:,:-1,:-1] - x[:,:,1:,:-1]) **2 +
            (x[:,:,:-1,:-1] - x[:,:,:-1,1:]) **2+ epsilon**2
        )
    )
    return loss
    
    
def motion_sym_loss(offset, epsilon, occlusion = None):
    if occlusion == None:
        # return torch.mean(torch.sqrt( (offset[:,:2,...] + offset[:,2:,...])**2 + epsilon **2))
        return torch.mean(torch.sqrt( (offset[0] + offset[1])**2 + epsilon **2))
    else:
        # TODO: how to design the occlusion aware offset symmetric loss?
        # return torch.mean(torch.sqrt((offset[:,:2,...] + offset[:,2:,...])**2 + epsilon **2))
        return torch.mean(torch.sqrt((offset[0] + offset[1])**2 + epsilon **2))



    
def part_loss(diffs, offsets, occlusions, images, epsilon, use_negPSNR=False):
    if use_negPSNR:
        pixel_loss = [negPSNR_loss(diff, epsilon) for diff in diffs]
    else:
        pixel_loss = [charbonier_loss(diff, epsilon) for diff in diffs]
    #offset_loss = [tv_loss(offset[0], epsilon) + tv_loss(offset[1], epsilon) for offset in
    #               offsets]

    if offsets[0][0] is not None:
        offset_loss = [gra_adap_tv_loss(offset[0],images[0], epsilon) + gra_adap_tv_loss(offset[1], images[1], epsilon) for offset in
                   offsets]
    else:
        offset_loss = [Variable(torch.zeros(1).cuda())]
    # print(torch.max(occlusions[0]))
    # print(torch.min(occlusions[0]))
    # print(torch.mean(occlusions[0]))

    # occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion - 0.5, epsilon) for occlusion in occlusions]
    # occlusion_loss = [smooth_loss(occlusion, epsilon) + charbonier_loss(occlusion[:, 0, ...] - occlusion[:, 1, ...], epsilon) for occlusion in occlusions]



    sym_loss = [motion_sym_loss(offset,epsilon=epsilon) for offset in offsets]
    # sym_loss = [ motion_sym_loss(offset,occlusion) for offset,occlusion in zip(offsets,occlusions)]
    return pixel_loss, offset_loss, sym_loss

