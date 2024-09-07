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
    "--save", "-s", type=str, default=None, help="directory for video saving"
)
parser.add_argument(
    "--scale", "-scale", type=int, default=2, help="X 2/4/8 interpolation"
)
parser.add_argument("--eval", "-e", action="store_true", help="evaluate scores or not")
parser.add_argument(
    "--dataset", "-dst", type=str, default="davis480P", help="dataset name"
)
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
if args.save:
    visDir = args.save
    os.makedirs(visDir, exist_ok=True)
    print("VIS (visualize) Dir: ", visDir)

############ build Dataset ############
dataset = args.dataset

if dataset.lower() == "adobe240":
    dstDir = "data/Adobe240"
    RESCALE = None
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "xiph2k":
    dstDir = "data/XiPH"
    RESCALE = "2K"
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "xiph4k":
    dstDir = "data/XiPH"
    RESCALE = "4K"
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "davis480p":
    dstDir = "data/DAVIS/JPEGImages/480p"
    RESCALE = None
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "davis1080p":
    dstDir = "data/DAVIS/JPEGImages/Full-Resolution"
    RESCALE = "1080P"
    videos = sorted(os.listdir(dstDir))
elif dataset.lower() == "custom":
    dstDir = "data/custom"
    RESCALE = None
    videos = sorted(os.listdir(dstDir))
else:
    raise NotImplementedError("Unsupported Dataset")


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


############ build SCORE model ############
if args.eval:
    metrics = ["psnr", "ssim", "lpips", "dists", "flolpips", "vfips"]
    print("Building SCORE models...", metrics)
    metric = Metrics(metrics, skip_ref_frames=SCALE)
    print("Done")


############ load videos, interpolate, calc metric ############
print(len(videos))
scores = {}
for vid_name in tqdm(videos):
    sequences = [
        x for x in os.listdir(osp.join(dstDir, vid_name)) if ".jpg" in x or ".png" in x
    ]
    sequences.sort(key=lambda x: int(x[:-4]))
    sequences = [osp.join(dstDir, vid_name, x) for x in sequences]
    targetSeq = sequences[: (len(sequences) - 1) // SCALE * SCALE + 1]

    ############ write reference video frames ############
    out_dir = osp.join(tmpDir, "refImages")
    for cnt, f in enumerate(targetSeq):
        frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        frame = Tools.resize(frame, RESCALE)
        cv2.imwrite(f"{out_dir}/{cnt:0>7d}.png", frame)
    height, width, _ = frame.shape
    tot_frames = len(targetSeq)
    print("VIDEO: ", vid_name, " (%d x %d x %d)" % (tot_frames, height, width))

    ############ build buffer with multi-threads ############
    inputSeq = Tools.sample_sequence(targetSeq, interval=SCALE)
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
    if args.save:
        print("SAVING to .mp4")
        os.makedirs(osp.join(visDir, vid_name), exist_ok=True)
        disPth = osp.join(visDir, vid_name, f"{vid_name}-{args.method}.mp4")
        refPth = osp.join(visDir, vid_name, f"{vid_name}-GT.mp4")

        Tools.frames2mp4(osp.join(tmpDir, "disImages", "*.png"), disPth)
        if not osp.isfile(refPth):
            Tools.frames2mp4(osp.join(tmpDir, "refImages", "*.png"), refPth)

    ############ save .yuv files for calc metric ############
    disPth = osp.join(tmpDir, "dis.yuv")
    refPth = osp.join(tmpDir, "ref.yuv")
    Tools.frames2rawvideo(osp.join(tmpDir, "disImages", "*.png"), disPth)
    Tools.frames2rawvideo(osp.join(tmpDir, "refImages", "*.png"), refPth)

    ############ calc metric ############
    if args.eval:
        meta = dict(
            tmpDir=tmpDir,
            disImgs=osp.join(tmpDir, "disImages"),
            refImgs=osp.join(tmpDir, "refImages"),
            disMP4=disPth,
            refMP4=refPth,
            scale=SCALE,
            hwt=(height, width, tot_frames),
        )
        print("Calculating metrics")
        s = metric.eval(meta)
        scores[vid_name] = s
        print({k: f"{v:.2f}" for k, v in s.items()})

    ############ delete tmp files ############
    os.system("rm -rf %s/*/*.png" % tmpDir)
    os.system("rm -rf %s/*.mp4" % tmpDir)

# save result to txt
if args.eval:
    avg_score = {k: np.mean([v[k] for v in scores.values()]) for k in metrics}
    print("AVG Score of %s".center(41, "=") % args.method)
    for k, v in avg_score.items():
        print("{:<10} {:<10.3f}".format(k, v))

    os.makedirs("scores", exist_ok=True)
    need_head = False if osp.isfile(f"scores/{dataset}X{SCALE}.txt") else True
    result_file = f"scores/{dataset}X{SCALE}.txt"
    with open(result_file, "a+") as f:
        if need_head:
            head = "{:<10} " + " {:<10} " * len(metrics) + "\n"
            f.write(head.format("methods", *metrics))
        avg_message = "{:<10} " + " {:<10.3f} " * len(metrics) + "\n"
        f.write(avg_message.format(args.method, *[avg_score[x] for x in metrics]))
