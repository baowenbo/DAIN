import time
import os
from torch.autograd import Variable
import torch
import random
import numpy as np
import numpy
import networks
from my_args import  args
from scipy.misc import imread, imsave
from AverageMeter import  *
import shutil

torch.backends.cudnn.benchmark = True # to speed up the

DO_MiddleBurryOther = True
MB_Other_DATA = "./MiddleBurySet/other-data/"
MB_Other_RESULT = "./MiddleBurySet/other-result-author/"
MB_Other_GT = "./MiddleBurySet/other-gt-interp/"
if not os.path.exists(MB_Other_RESULT):
    os.mkdir(MB_Other_RESULT)



model = networks.__dict__[args.netName](    channel=args.channels,
                                    filter_size = args.filter_size ,
                                    timestep=args.time_step,
                                    training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval() # deploy mode

use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()
if DO_MiddleBurryOther:
    subdir = os.listdir(MB_Other_DATA)
    gen_dir = os.path.join(MB_Other_RESULT, unique_id)
    os.mkdir(gen_dir)

    tot_timer = AverageMeter()
    proc_timer = AverageMeter()
    end = time.time()
    for dir in subdir: 
        print(dir)
        os.mkdir(os.path.join(gen_dir, dir))
        arguments_strFirst = os.path.join(MB_Other_DATA, dir, "frame10.png")
        arguments_strSecond = os.path.join(MB_Other_DATA, dir, "frame11.png")
        gt_path = os.path.join(MB_Other_GT, dir, "frame10i11.png")

        X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
        X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)


        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)
        if not channel == 3:
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)
        X1 = pader(X1)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        proc_end = time.time()
        y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
        y_ = y_s[save_which]

        proc_timer.update(time.time() -proc_end)
        tot_timer.update(time.time() - end)
        end  = time.time()
        print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
        if use_cuda:
            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = y_.data.cpu().numpy()
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
        else:
            X0 = X0.data.numpy()
            if not isinstance(y_, list):
                y_ = y_.data.numpy()
            else:
                y_ = [item.data.numpy() for item in y_]
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()



        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                  intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter]  if filter is not None else None
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

        timestep = args.time_step
        numFrames = int(1.0 / timestep) - 1
        time_offsets = [kk * timestep for kk in range(1, 1 + numFrames, 1)]
        # for item, time_offset  in zip(y_,time_offsets):
        #     arguments_strOut = os.path.join(gen_dir, dir, "frame10_i{:.3f}_11.png".format(time_offset))
        #
        #     imsave(arguments_strOut, np.round(item).astype(numpy.uint8))
        #
        # # copy the first and second reference frame
        # shutil.copy(arguments_strFirst, os.path.join(gen_dir, dir,  "frame10_i{:.3f}_11.png".format(0)))
        # shutil.copy(arguments_strSecond, os.path.join(gen_dir, dir,  "frame11_i{:.3f}_11.png".format(1)))

        count = 0
        shutil.copy(arguments_strFirst, os.path.join(gen_dir, dir, "{:0>4d}.png".format(count)))
        count  = count+1
        for item, time_offset in zip(y_, time_offsets):
            arguments_strOut = os.path.join(gen_dir, dir, "{:0>4d}.png".format(count))
            count = count + 1
            imsave(arguments_strOut, np.round(item).astype(numpy.uint8))
        shutil.copy(arguments_strSecond, os.path.join(gen_dir, dir, "{:0>4d}.png".format(count)))
        count = count + 1


         