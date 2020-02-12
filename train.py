import sys
import os

import threading
import torch
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *

import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
from my_args import args



def train():
    torch.manual_seed(args.seed)

    model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=True)
    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()

    if not args.SAVED_MODEL==None:
        # args.SAVED_MODEL ='../model_weights/'+ args.SAVED_MODEL + "/best" + ".pth"
        args.SAVED_MODEL ='./model_weights/best.pth'
        print("Fine tuning on " +  args.SAVED_MODEL)
        if not  args.use_cuda:
            pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(args.SAVED_MODEL)
            # model.load_state_dict(torch.load(args.SAVED_MODEL))
        #print([k for k,v in      pretrained_dict.items()])

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        pretrained_dict = None

    if type(args.datasetName) == list:
        train_sets, test_sets = [],[]
        for ii, jj in zip(args.datasetName, args.datasetPath):
            tr_s, te_s = datasets.__dict__[ii](jj, split = args.dataset_split,single = args.single_output, task = args.task)
            train_sets.append(tr_s)
            test_sets.append(te_s)
        train_set = torch.utils.data.ConcatDataset(train_sets)
        test_set = torch.utils.data.ConcatDataset(test_sets)
    else:
        train_set, test_set = datasets.__dict__[args.datasetName](args.datasetPath)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = args.batch_size,
        sampler=balancedsampler.RandomBalancedSampler(train_set, int(len(train_set) / args.batch_size )),
        num_workers= args.workers, pin_memory=True if args.use_cuda else False)

    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                             num_workers=args.workers, pin_memory=True if args.use_cuda else False)
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))


    # if not args.lr == 0:
    print("train the interpolation net")
    optimizer = torch.optim.Adamax([
                {'params': model.initScaleNets_filter.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.initScaleNets_filter1.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.initScaleNets_filter2.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.ctxNet.parameters(), 'lr': args.ctx_lr_coe * args.lr},
                {'params': model.flownets.parameters(), 'lr': args.flow_lr_coe * args.lr},
                {'params': model.depthNet.parameters(), 'lr': args.depth_lr_coe * args.lr},
                {'params': model.rectifyNet.parameters(), 'lr': args.rectify_lr}
            ],
                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)


    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=args.factor, patience=args.patience,verbose=True)

    print("*********Start Training********")
    print("LR is: "+ str(float(optimizer.param_groups[0]['lr'])))
    print("EPOCH is: "+ str(int(len(train_set) / args.batch_size )))
    print("Num of EPOCH is: "+ str(args.numEpoch))
    def count_network_parameters(model):

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])

        return N
    print("Num. of model parameters is :" + str(count_network_parameters(model)))
    if hasattr(model,'flownets'):
        print("Num. of flow model parameters is :" +
              str(count_network_parameters(model.flownets)))
    if hasattr(model,'initScaleNets_occlusion'):
        print("Num. of initScaleNets_occlusion model parameters is :" +
              str(count_network_parameters(model.initScaleNets_occlusion) +
                  count_network_parameters(model.initScaleNets_occlusion1) +
        count_network_parameters(model.initScaleNets_occlusion2)))
    if hasattr(model,'initScaleNets_filter'):
        print("Num. of initScaleNets_filter model parameters is :" +
              str(count_network_parameters(model.initScaleNets_filter) +
                  count_network_parameters(model.initScaleNets_filter1) +
        count_network_parameters(model.initScaleNets_filter2)))
    if hasattr(model, 'ctxNet'):
        print("Num. of ctxNet model parameters is :" +
              str(count_network_parameters(model.ctxNet)))
    if hasattr(model, 'depthNet'):
        print("Num. of depthNet model parameters is :" +
              str(count_network_parameters(model.depthNet)))
    if hasattr(model,'rectifyNet'):
        print("Num. of rectifyNet model parameters is :" +
              str(count_network_parameters(model.rectifyNet)))

    training_losses = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    ikk = 0
    for kk in optimizer.param_groups:
        if kk['lr'] > 0:
            ikk = kk
            break

    for t in range(args.numEpoch):
        print("The id of this in-training network is " + str(args.uid))
        print(args)
        #Turn into training mode
        model = model.train()

        for i, (X0_half,X1_half, y_half) in enumerate(train_loader):

            if i >= int(len(train_set) / args.batch_size ):
                #(0 if t == 0 else EPOCH):#
                break

            X0_half = X0_half.cuda() if args.use_cuda else X0_half
            X1_half = X1_half.cuda() if args.use_cuda else X1_half
            y_half = y_half.cuda() if args.use_cuda else y_half

            X0 = Variable(X0_half, requires_grad= False)
            X1 = Variable(X1_half, requires_grad= False)
            y  = Variable(y_half,requires_grad= False)

            diffs, offsets,filters,occlusions = model(torch.stack((X0,y,X1),dim = 0))

            pixel_loss, offset_loss, sym_loss = part_loss(diffs,offsets,occlusions, [X0,X1],epsilon=args.epsilon)

            total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))

            training_losses.update(total_loss.item(), args.batch_size)
            if i % max(1, int(int(len(train_set) / args.batch_size )/500.0)) == 0:

                print("Ep [" + str(t) +"/" + str(i) +
                                    "]\tl.r.: " + str(round(float(ikk['lr']),7))+
                                    "\tPix: " + str([round(x.item(),5) for x in pixel_loss]) +
                                    "\tTV: " + str([round(x.item(),4)  for x in offset_loss]) +
                                    "\tSym: " + str([round(x.item(), 4) for x in sym_loss]) +
                                    "\tTotal: " + str([round(x.item(),5) for x in [total_loss]]) +
                                    "\tAvg. Loss: " + str([round(training_losses.avg, 5)]))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if t == 1:
            # delete the pre validation weights for cleaner workspace
            if os.path.exists(args.save_path + "/epoch" + str(0) +".pth" ):
                os.remove(args.save_path + "/epoch" + str(0) +".pth")

        if os.path.exists(args.save_path + "/epoch" + str(t-1) +".pth"):
            os.remove(args.save_path + "/epoch" + str(t-1) +".pth")
        torch.save(model.state_dict(), args.save_path + "/epoch" + str(t) +".pth")

        # print("\t\t**************Start Validation*****************")
        #Turn into evaluation mode

        val_total_losses = AverageMeter()
        val_total_pixel_loss = AverageMeter()
        val_total_PSNR_loss = AverageMeter()
        val_total_tv_loss = AverageMeter()
        val_total_pws_loss = AverageMeter()
        val_total_sym_loss = AverageMeter()

        for i, (X0,X1,y) in enumerate(val_loader):
            if i >=  int(len(test_set)/ args.batch_size):
                break

            with torch.no_grad():
                X0 = X0.cuda() if args.use_cuda else X0
                X1 = X1.cuda() if args.use_cuda else X1
                y = y.cuda() if args.use_cuda else y

                diffs, offsets,filters,occlusions = model(torch.stack((X0,y,X1),dim = 0))

                pixel_loss, offset_loss,sym_loss = part_loss(diffs, offsets, occlusions, [X0,X1],epsilon=args.epsilon)

                val_total_loss = sum(x * y for x, y in zip(args.alpha, pixel_loss))

                per_sample_pix_error = torch.mean(torch.mean(torch.mean(diffs[args.save_which] ** 2,
                                                                    dim=1),dim=1),dim=1)
                per_sample_pix_error = per_sample_pix_error.data # extract tensor
                psnr_loss = torch.mean(20 * torch.log(1.0/torch.sqrt(per_sample_pix_error)))/torch.log(torch.Tensor([10]))
                #

                val_total_losses.update(val_total_loss.item(),args.batch_size)
                val_total_pixel_loss.update(pixel_loss[args.save_which].item(), args.batch_size)
                val_total_tv_loss.update(offset_loss[0].item(), args.batch_size)
                val_total_sym_loss.update(sym_loss[0].item(), args.batch_size)
                val_total_PSNR_loss.update(psnr_loss[0],args.batch_size)
                print(".",end='',flush=True)

        print("\nEpoch " + str(int(t)) +
              "\tlearning rate: " + str(float(ikk['lr'])) +
              "\tAvg Training Loss: " + str(round(training_losses.avg,5)) +
              "\tValidate Loss: " + str([round(float(val_total_losses.avg), 5)]) +
              "\tValidate PSNR: " + str([round(float(val_total_PSNR_loss.avg), 5)]) +
              "\tPixel Loss: " + str([round(float(val_total_pixel_loss.avg), 5)]) +
              "\tTV Loss: " + str([round(float(val_total_tv_loss.avg), 4)]) +
              "\tPWS Loss: " + str([round(float(val_total_pws_loss.avg), 4)]) +
              "\tSym Loss: " + str([round(float(val_total_sym_loss.avg), 4)])
              )

        auxiliary_data.append([t, float(ikk['lr']),
                                   training_losses.avg, val_total_losses.avg, val_total_pixel_loss.avg,
                                   val_total_tv_loss.avg,val_total_pws_loss.avg,val_total_sym_loss.avg])

        numpy.savetxt(args.log, numpy.array(auxiliary_data), fmt='%.8f', delimiter=',')
        training_losses.reset()

        print("\t\tFinished an epoch, Check and Save the model weights")
            # we check the validation loss instead of training loss. OK~
        if saved_total_loss >= val_total_losses.avg:
            saved_total_loss = val_total_losses.avg
            torch.save(model.state_dict(), args.save_path + "/best"+".pth")
            print("\t\tBest Weights updated for decreased validation loss\n")

        else:
            print("\t\tWeights Not updated for undecreased validation loss\n")

        #schdule the learning rate
        scheduler.step(val_total_losses.avg)


    print("*********Finish Training********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
