import functools
import importlib
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .archs.VFIformer_arch import VFIformer
from .utils.util import scandir


def define_network(args):
    net = VFIformer(args)
    return net


def init_weights(net, init_type="kaiming", gain=0.02):
    def init_func(net):
        for name, m in net.named_modules():
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(
    net, gpu_ids=[], device=None, dist=False, init_type="normal", init_gain=0.02
):
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            raise AssertionError
        net.to(device)
        if dist:
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        else:
            net = torch.nn.DataParallel(net, gpu_ids)
    # init_weights(net, init_type, gain=init_gain)
    return net


def define_G(
    args,
    init_type="xavier",
    init_gain=0.02,
):
    gpu_ids = args.gpu_ids
    device = args.device
    dist = args.dist

    net = define_network(args)
    return init_net(net, gpu_ids, device, dist, init_type, init_gain)
