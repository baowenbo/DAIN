import time
import os
from torch.autograd import Variable
import torch
import numpy as np
import numpy
import networks
from my_args import  args
from scipy.misc import imread, imsave
from AverageMeter import  *
import shutil
import datetime
torch.backends.cudnn.benchmark = True

model = networks.__dict__[args.netName](
                                    channel=args.channels,
                                    filter_size = args.filter_size,
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
    print("**** We couldn't load any trained weights ***********************")
    print("*****************************************************************")
    exit(1)

model = model.eval() # deploy mode

use_cuda = args.use_cuda
save_which = args.save_which
dtype = args.dtype

frames_dir = '/content/DAIN/input_frames'
output_dir = '/content/DAIN/output_frames'

timestep = args.time_step
time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]

output_frame_count = 1
input_frame = 0
loop_timer = AverageMeter()

# TODO: Read amount of frames from the size of files available in `frames_dir`
final_frame = 100 

while input_frame < final_frame:
    input_frame += 1

    start_time = time.time()

    #input file names
    frame_1_filename = f"{input_frame:0>5}.png" # frame00010.png
    frame_1_path = os.path.join(frames_dir, frame_1_filename)

    frame_2_filename = f"{input_frame + 1:0>5}.png" # frame00011.png
    frame_2_path = os.path.join(frames_dir, frame_1_filename)

    X0 =  torch.from_numpy( np.transpose(imread(frame_1_path), (2,0,1)).astype("float32")/ 255.0).type(dtype)
    X1 =  torch.from_numpy( np.transpose(imread(frame_2_path), (2,0,1)).astype("float32")/ 255.0).type(dtype)

    y_ = torch.FloatTensor()

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channel = X0.size(0)
    if not channel == 3:
        print(f"Skipping {frame_1_filename}-{frame_2_filename} -- expected 3 color channels but found {channel}.")
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

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

    torch.set_grad_enabled(False)
    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)

    if use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()

    y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[save_which]

    frames_left = output_frame_count - input_frame
    estimated_seconds_left = frames_left * loop_timer.avg
    estimated_time_left = datetime.timedelta(seconds=estimated_seconds_left)
    print(f"******Processed image {input_frame} | Time per image (avg): {loop_timer.avg:2.2f}s | Time left: {estimated_time_left} ******************" )

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
                                intPaddingLeft:intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter]  if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

    shutil.copy(frame_1_path, os.path.join(output_dir, f"{output_frame_count:0>5d}.png"))
    output_frame_count += 1
    for item, time_offset in zip(y_, time_offsets):
        output_frame_file_path = os.path.join(output_dir, f"{output_frame_count:0>5d}.png")
        imsave(output_frame_file_path, np.round(item).astype(numpy.uint8))
        output_frame_count += 1

    end_time = time.time()
    loop_timer.update(end_time - start_time)

print("Finished processing images.")