import time
import os
import sys
from torch.autograd import Variable
import torch
import numpy as np
import numpy
import networks
from my_args import args
from imageio import imread, imsave
import cv2
from PIL import ImageFont, ImageDraw, Image
from urllib.request import urlopen
from tqdm import tqdm
import shutil
torch.backends.cudnn.benchmark = True

model = networks.__dict__[args.netName](
                                    channel = args.channels,
                                    filter_size = args.filter_size,
                                    timestep = args.time_step,
                                    training = False)

if args.use_cuda:
    model = model.cuda()

model_path = './model_weights/best.pth'
if not os.path.exists(model_path):
    print("*****************************************************************")
    print("**** We couldn't load any trained weights ***********************")
    print("*****************************************************************")
    exit(1)

if args.use_cuda:
    pretrained_dict = torch.load(model_path)
else:
    pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

model_dict = model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
# 4. release the pretrained dict for saving memory
pretrained_dict = []

model = model.eval() # deploy mode

frames_dir = args.frame_input_dir
output_dir = args.frame_output_dir

timestep = args.time_step
time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]

input_frame = args.start_frame - 1

final_frame = args.end_frame

torch.set_grad_enabled(False)

font_url = 'https://github.com/googlefonts/roboto/blob/master/src/hinted/Roboto-Bold.ttf?raw=true'
big_font = ImageFont.truetype(urlopen(font_url), size=24)
smol_font = ImageFont.truetype(urlopen(font_url), size=14)

interpolated_frame_number = 0


def debug(f):
    """ This decorator enables or disables debugging, such as and plastering 
    debug information on the output images. Helps to reason/tune algorithm behaviour. """
    def wrapper(*pargs, **kwargs):
        if args.debug:
            return f(*pargs, **kwargs)
    return wrapper


class SkipFrame(Exception):
    pass


def load_scene_frames():
    """ Scene frames are the last frames of scenes, their successors introduce a new scene """
    frames = set()
    with open('scene_frames.log') as f:
        for line in f:
            if line:
                frames.add(int(line))
    return frames


def is_jumping_scene(from_frame, to_frame):
    """ It's jumping unless both frames and all between are of the same scene """
    for f in range(from_frame, to_frame):
        if f in scene_frames:
            return True
    return False


def input_filename(input_frame):
    return os.path.join(frames_dir, f'{input_frame:0>5d}.png')


def generate_output_filename(input_frame):
    global interpolated_frame_number
    interpolated_frame_number += 1
    return os.path.join(output_dir, f"{input_frame:0>5d}{interpolated_frame_number:0>3d}.png")


def copy(input_frame):
    shutil.copy(input_filename(input_frame), generate_output_filename(input_frame))


def subframes(image_1, image_2):
    X0 = torch.from_numpy(np.transpose(image_1, (2,0,1)).astype("float32") / 255.0).type(args.dtype)
    X1 = torch.from_numpy(np.transpose(image_2, (2,0,1)).astype("float32") / 255.0).type(args.dtype)

    assert (X0.size(1) == X1.size(1))
    assert (X0.size(2) == X1.size(2))

    intWidth = X0.size(2)
    intHeight = X0.size(1)
    channels = X0.size(0)
    assert channels == 3
    if not channels == 3:
        print(f"Skipping frame -- expected 3 color channels but found {channels}.")
        raise SkipFrame()

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft = int((intWidth_pad - intWidth) / 2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

    X0 = Variable(torch.unsqueeze(X0,0))
    X1 = Variable(torch.unsqueeze(X1,0))
    X0 = pader(X0)
    X1 = pader(X1)

    if args.use_cuda:
        X0 = X0.cuda()
        X1 = X1.cuda()

    y_s, offset, filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[args.save_which]

    if args.use_cuda:
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

    for item, time_offset in zip(y_, time_offsets):
        image = np.round(item).astype(numpy.uint8)
        if args.resize_hotfix:
            dimension = (image.shape[1]+2, image.shape[0]+2)
            resized = cv2.resize(image, dimension, interpolation=cv2.INTER_LANCZOS4)
            image = resized[1:(dimension[1]-1), 1:(dimension[0]-1)]
        yield image


def midframe(image_1, image_2):
    assert timestep == 0.5
    return next(subframes(image_1, image_2))


def interpolate(frame_1, frame_2):
    image_1 = imread(input_filename(input_frame))
    image_2 = imread(input_filename(input_frame+1))
    for frame, time_offset in zip(subframes(image_1, image_2), time_offsets):
        imsave(generate_output_filename(input_frame), frame)


def normal_interpolate(input_frame):
    """ The normal interpolation between input_frame and input_frame+1 """
    copy(input_frame)
    if not is_jumping_scene(input_frame, input_frame+1):
        interpolate(input_frame, input_frame+1)
    else:
        for _ in time_offsets:
            copy(input_frame)


def greased_interpolate(input_frame):
    """ Smooth out from further frames than the directly adjacent ones. Only works with 2x frame rate """
    assert timestep == 0.5
    if input_frame - 1 < 1 or input_frame + 2 > final_frame or is_jumping_scene(input_frame-1, input_frame+2):
        normal_interpolate(input_frame)
    else:
        interpolate(input_frame-1, input_frame+1)
        if input_frame == final_frame - 2:
            interpolate(input_frame, input_frame+1)
        else:
            interpolate(input_frame-1, input_frame+2)


@debug
def debug_text_on_image(image, s, position='left'):
    canvas = np.full((image.shape), (0,0,0), dtype=np.uint8)
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    coords = (15,15) if position == 'left' else (200,15)
    font = big_font if position == 'left' else smol_font
    draw.text(coords, s, font=font, fill=(255,255,255,0))
    drawing = np.array(canvas_pil)
    x,y,w,h = cv2.boundingRect(drawing[:,:,2])
    m = 15
    x,y,w,h = x-m,y-m,w+2*m,h+2*m
    image[y:y+h, x:x+w] = drawing[y:y+h, x:x+w]


def distorted_pixel_ratio(frame_1, frame_2, distortion_threshold):
    diff = cv2.absdiff(frame_1, frame_2)
    _,bw = cv2.threshold(diff, distortion_threshold, 255, cv2.THRESH_BINARY)
    ratio = cv2.countNonZero(cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)) / bw.size
    return ratio


@debug
def describe_diffs(frame_1, frame_2):
    ratios = {}
    for t in [10,15,20,25,30]:
        ratio = distorted_pixel_ratio(frame_1, frame_2, t)
        ratios[t] = ratio
    return '; '.join(f'{t}={ratio:.4f}' for t, ratio in ratios.items())


def contains_double_frames(frames):
    # Double frames are usually delibirately not completely equal, to make
    # the animation feel alive. The goal of double frames is to save animation 
    # effort, so the induced distortion is done by adding noise to brightness
    # or color. The pixel to pixel difference is relatively small. Large
    # differences are false positives or animation. Even tiny areas of
    # of the canvas may be animated, and ideally shouldn't be treated as double 
    # frames.
    for i in range(len(frames)-1):
        ratio = distorted_pixel_ratio(frames[i], frames[i+1], args.doubleframe_distortion_threshold)
        if ratio < args.doubleframe_distorted_pixel_ratio:
            return True
    return False


def scene_limits(input_frame):
    floor_frame = input_frame-2
    for f in range(input_frame, input_frame-2, -1):
        if f-1 in scene_frames:
            floor_frame = f
            break
    ceil_frame = input_frame+2
    for f in range(input_frame, input_frame+2, +1):
        if f in scene_frames:
            ceil_frame = f
            break

    description = ''
    if floor_frame == input_frame-1:
        description += 'floor=b '
    if floor_frame == input_frame:
        description += 'floor=c '
    if ceil_frame == input_frame:
        description += 'ceil=c '
    if ceil_frame == input_frame+1:
        description += 'ceil=d '

    return floor_frame, ceil_frame, description


def claymation_interpolate(input_frame):
    """ Extreme interpolate for 2x frame rate. Smooth transition even when the source animation has
    models moving only every second frame. Clay animation has such behaviour, and the same scene
    may even have models moving every frame while others only every second frame. """
    assert timestep == 0.5
    if input_frame - 2 < 1 or input_frame + 2 > final_frame:
        normal_interpolate(input_frame)
    else:
        # Using avg of 4 frame pointers and moving forward 2 of them. The moved
        # pointers must be 3 frames apart. This smoothes out duplicates, sacrificing
        # accuracy for slick animation. Even the input frame pointer needs to be
        # replaced to ensure pointers always move forward.
        # A B CxD E
        # C) (A^C)^D
        # x) (B^C)^(D^E)
        # 112233
        # 1.5 1.75 2 ...
        # However such approach smears out fast movement a lot. So detect if
        # double frames are actually happening, choose a more local interpolation
        # if not.
        # 0 0 1 1 2 2 3 4 5 6 7 8 8 9 9
        #           oxxxxxxxxxmo
        # 2 2.25 2.5 ... 6.5 7 7.25
        # x = must interpolate from close neighbors, the premise
        # o = can use the normal smooth_c calculation, all pointers more forward
        # m = (B^C)^(C^D)

        floor_frame, ceil_frame, s_limits = scene_limits(input_frame)
        image_a = imread(input_filename(max(input_frame-2, floor_frame)))
        image_b = imread(input_filename(max(input_frame-1, floor_frame)))
        image_c = imread(input_filename(input_frame))
        image_d = imread(input_filename(min(input_frame+1, ceil_frame)))
        image_e = imread(input_filename(min(input_frame+2, ceil_frame)))

        s_diffs = describe_diffs(image_c, image_d)

        @debug
        def d(image, algo):
            debug_text_on_image(image, f'{s_limits}{algo}')
            debug_text_on_image(image, f'{s_diffs}', position='right')

        # Copy or generate a new input_frame
        if contains_double_frames([image_b, image_c, image_d]):
            image_ac = midframe(image_a, image_c)
            image_acd = midframe(image_ac, image_d)
            image_smooth_c = image_acd
            d(image_smooth_c, 'acd')
            imsave(generate_output_filename(input_frame), image_smooth_c)
        else:
            if contains_double_frames([image_d, image_e]):
                # Transitioning into double frames
                image_bc = midframe(image_b, image_c)
                image_cd = midframe(image_c, image_d)
                image_smooth_c = midframe(image_bc, image_cd)
                d(image_smooth_c, 'bcd')
                imsave(generate_output_filename(input_frame), image_smooth_c)
            else:
                copy(input_frame)

        # Generate the interpolated frame
        if contains_double_frames([image_c, image_d, image_e]):
            image_bc = midframe(image_b, image_c)
            image_de = midframe(image_d, image_e)
            image_smooth_cd = midframe(image_bc, image_de)
            d(image_smooth_cd, 'bcde')
            imsave(generate_output_filename(input_frame), image_smooth_cd)
        else:
            image_cd = midframe(image_c, image_d)
            d(image_cd, 'cd')
            imsave(generate_output_filename(input_frame), image_cd)


mixers = {
    'normal': normal_interpolate,
    'greased': greased_interpolate,
    'claymation': claymation_interpolate,
}
interpolate_mixer = mixers.get(args.mixer)
assert interpolate_mixer is not None, f'Mixer "{args.mixer}" not found'

scene_frames = load_scene_frames()

# we want to have input_frame between (start_frame-1) and (end_frame-2)
# this is because at each step we read (frame) and (frame+1)
# so the last iteration will actuall be (end_frame-1) and (end_frame)
total = final_frame - input_frame - 1
with tqdm(total=total, file=sys.stdout, smoothing=0) as pbar:
    while input_frame < final_frame - 1:
        pbar.update(1)
        input_frame += 1
        interpolated_frame_number = 0
        try:
            interpolate_mixer(input_frame)
        except SkipFrame:
            pass

# Copying last frame
last_frame_filename = os.path.join(frames_dir, str(str(final_frame).zfill(5))+'.png')
shutil.copy(last_frame_filename, os.path.join(output_dir, f"{final_frame:0>5d}{0:0>3d}.png"))

print("Finished processing images.")
