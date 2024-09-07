import argparse
import os
import os.path as osp
import tempfile
import warnings

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

import tools
from calc_metric import Metrics

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

############ Arguments ############

parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", type=str, default="edsc")
parser.add_argument("--dataset", "-dst", type=str, default="vimeo")
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

print("Testing on Dataset: ", args.dataset)
print("Running VFI method : ", args.method)
print("TMP (temporary) Dir: ", tmpDir)


############ build Dataset ############
dataset = args.dataset
if dataset.lower() == "middlebury":
    dst = tools.MiddelBury("data/Middlebury")

elif dataset.lower() == "vimeo":
    dst = tools.Vimeo90K_test("data/Vimeo-Test")

elif dataset.lower() == "ucf101":
    dst = tools.UCF101_test("data/UCF101-test")

else:
    raise NotImplementedError("Unsupported Dataset")


############ build VFI model ############
from build_models import build_model

print("Building VFI model...")
model, infer = build_model(args.method, device=DEVICE)
print("Done")


def inferRGB(*inputs):
    inputs = [x.to(DEVICE) for x in inputs]
    tenOut = infer(*inputs, time=0.5)
    return tenOut.cpu()


############ build SCORE model ############
metrics = ["psnr", "ssim", "lpips"]
# metrics = ["psnr", "ssim", "lpips", "ie", "dists"]
print("Building SCORE models...", metrics)
metric = Metrics(metrics, skip_ref_frames=None, batch_size=1)
print("Done")


############ load videos, interpolate, calc metric ############
for i, (I0, I1, I2) in enumerate(tqdm(dst)):
    if "stmfnet" in args.method.lower():
        out = inferRGB(I0, I0, I2, I2)
    else:
        out = inferRGB(I0, I2)
    disPth = f"{tmpDir}/disImages/{i:08d}.png"
    refPth = f"{tmpDir}/refImages/{i:08d}.png"
    TF.to_pil_image(out[0]).save(disPth)
    TF.to_pil_image(I1[0]).save(refPth)

############ calc metric ############
print("Calculating metrics")
meta = dict(disImgs=f"{tmpDir}/disImages", refImgs=f"{tmpDir}/refImages")
avg_score = metric.eval(meta)

############ delete tmp files ############
os.system("rm -rf %s/*/*.png" % tmpDir)

############ save results ############
print("AVG Score of %s".center(41, "=") % args.method)
for k, v in avg_score.items():
    print("{:<10} {:<10.3f}".format(k, v))

os.makedirs("scores", exist_ok=True)
need_head = False if osp.isfile(f"scores/{dataset}.txt") else True
result_file = f"scores/{dataset}.txt"
with open(result_file, "a+") as f:
    if need_head:
        head = "{:<10} " + " {:<10} " * len(metrics) + "\n"
        f.write(head.format("methods", *metrics))
    avg_message = "{:<10} " + " {:<10.3f} " * len(metrics) + "\n"
    f.write(avg_message.format(args.method, *[avg_score[x] for x in metrics]))
