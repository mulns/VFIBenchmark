import argparse
import os
import os.path as osp
import tempfile
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from calc_metric import Metrics
from tools import IOBuffer, Tools

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

############ Arguments ############

parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", type=str, default="edsc")
parser.add_argument(
    "--save", "-s", type=str, default="compare_video", help="directory for video saving"
)
parser.add_argument(
    "--scale", "-scale", type=int, default=2, help="X 2/4/8 interpolation"
)
parser.add_argument(
    "--dataset", "-dst", type=str, default="davis480P", help="dataset name"
)
parser.add_argument("--fps", "-fps", type=int, default=10, help="output frame rate")

args = parser.parse_args()

############ Preliminary ############

# tmpDir = "tmp"
tmpDir = tempfile.TemporaryDirectory().name
os.makedirs(osp.join(tmpDir, "disImages"), exist_ok=True)
os.makedirs(osp.join(tmpDir, "refImages"), exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
SCALE = args.scale
print("Testing on Dataset: ", args.dataset)
print("Running VFI method : ", args.method)
print("TMP (temporary) Dir: ", tmpDir)

visDir = args.save
os.makedirs(visDir, exist_ok=True)
print("VIS (visualize) Dir: ", visDir)

############ build Dataset ############
dataset = args.dataset

if dataset.lower() == "custom":
    dstDir = "data/custom"
    RESCALE = None
    videos = sorted(os.listdir(dstDir))


############ build VFI model ############
from build_models import build_model

print("Building VFI model...")
model, infer = build_model(args.method, device=DEVICE)
print("Done")
if args.method.lower() == "stmfnet":
    input_frames = 4
else:
    input_frames = 2


def inferRGB(*inputs):
    inputs = [x.to(DEVICE) for x in inputs]
    outputs = []
    for time in range(SCALE - 1):
        t = (time + 1) / SCALE
        tenOut = infer(*inputs, time=t)
        outputs.append(tenOut.cpu())
    return outputs


############ load videos, interpolate, calc metric ############
print(len(videos))
scores = {}
for vid_name in tqdm(videos):
    sequences = [
        x for x in os.listdir(osp.join(dstDir, vid_name)) if ".jpg" in x or ".png" in x
    ]
    sequences.sort(key=lambda x: int(x[:-4]))
    sequences = [osp.join(dstDir, vid_name, x) for x in sequences]

    ############ build buffer with multi-threads ############
    inputSeq = sequences
    IO = IOBuffer(RESCALE, inp_num=input_frames)
    IO.start(inputSeq, osp.join(tmpDir, "disImages"))

    ############ interpolation & write distorted frames ############
    inps = IO.reader.get()  # e.g., [I1 I3]
    IO.writer.put(Tools.toArray(inps[0]))
    while True:
        outs = inferRGB(*inps)  # e.g., [I2]
        for o in Tools.toArray(outs + [inps[-input_frames // 2]]):
            IO.writer.put(o)
        inps = IO.reader.get()
        if inps is None:
            break
    IO.stop()

    ############ save .mp4 files for visualize ############

    print("SAVING to .mp4")
    os.makedirs(osp.join(visDir, vid_name), exist_ok=True)
    disPth = osp.join(visDir, vid_name, f"{vid_name}-{args.method}.mp4")

    Tools.frames2mp4(osp.join(tmpDir, "disImages", "*.png"), disPth, args.fps)
