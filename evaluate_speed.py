import argparse
import time

import torch
from tqdm import tqdm

from build_models import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", type=str, default="stmfnet")
args = parser.parse_args()
model_name = args.method
shape = (512, 512)


model, infer = build_model(model_name)

I1 = torch.randn(1, 3, *shape).to(device)
I2 = torch.randn(1, 3, *shape).to(device)
inputs = [I1, I2]
if model_name == "stmfnet":
    I3 = torch.randn(1, 3, *shape).to(device)
    I4 = torch.randn(1, 3, *shape).to(device)
    inputs = [I1, I2, I3, I4]
num_iterations = 100  # Number of iterations to run for more accurate timing
start_time = time.time()

for _ in tqdm(range(num_iterations)):
    output = infer(*inputs)

end_time = time.time()
elapsed_time = end_time - start_time
runtime = elapsed_time / num_iterations
runtime = "%.3f" % runtime + "s"
import os

from thop import clever_format, profile

macs, params = profile(model, infer=infer, inputs=inputs)
macs, params = clever_format([macs, params], "%.3f")
print(f"macs: {macs}, params: {params}, runtime: {runtime}")
print(f"time: {runtime}")


os.makedirs("scores", exist_ok=True)
result_file = "scores/profile.txt"
with open(result_file, "r") as f:
    need_head = True if f.readlines() == [] else False
with open(result_file, "a+") as f:
    if need_head:
        head = "{:<20} " + " {:<10} " * 3 + "\n"
        f.write(head.format("methods", *["runtime", "MACs", "#Params"]))
    message = "{:<20} " + " {:<10} " * 3 + "\n"
    f.write(message.format(model_name, *[runtime, macs, params]))
