import os
from pathlib import Path
import shutil
import subprocess as sp

def dain_slowmotion(
    input_filepath: Path,
    output_dir: Path,
    time_step: float,
    seamless: bool = False,
    dain_exec_path: Path = Path("/usr/local/dain/colab_interpolate.py")
):
    # Make the output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make input frames directory
    input_frames_dir = output_dir / "input_frames"
    shutil.rmtree(input_frames_dir, ignore_errors=True)
    input_frames_dir.mkdir()
    print(f"Created directory for input frames: {input_frames_dir}")

    # Make output frames directory
    output_frames_dir = output_dir / "output_frames"
    shutil.rmtree(output_frames_dir, ignore_errors=True)
    output_frames_dir.mkdir()
    print(f"Created directory for output frames: {output_frames_dir}")

    # Use ffmpeg to extract input frames
    print("Extracting input frames...")
    os.system(f"ffmpeg -i '{input_filepath}' '{input_frames_dir}/%05d.png'")
    print("Input frames have been extracted.")

    # Assign properties for input frames
    input_frames = [name for name in os.listdir(input_frames_dir) if os.path.isfile(input_frames_dir / name)]
    num_input_frames = len(input_frames)
    first_input_frame = input_frames_dir / f"{1:05d}.png"
    
    # Detect and remove alpha channel if necessary
    img_channels = sp.getoutput('identify -format %[channels] 00001.png')
    
    if "a" in img_channels:
        print("Detected alpha channel in input frames.  Removing.")
        print(sp.getoutput(f"find '{input_frames_dir}' -name '*.png' -exec convert '{{}}' -alpha off PNG24:'{{}}' \\;"))
        print("Each image has had its alpha channel removed.")

    # Use first frame as last if this is a looping video
    if seamless:
        loop_input_frame = input_frames_dir / f"{(num_input_frames + 1):05d}.png"
        shutil.copy(first_input_frame, loop_input_frame)
        print("Using first frame as last frame.")

dain_slowmotion(
    input_filepath=Path("/usr/local/dain/content/test1/books.mp4"),
    output_dir=Path("/usr/local/dain/content/test1-out"),
    time_step=0.25,
    seamless=True
)