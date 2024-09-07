import argparse
import itertools
import os
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from .modules.bi_flownet import BiFlowNet
from .modules.fusionnet import FusionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pipeline:

    def __init__(
        self,
        module_cfg_dict,  # To check the data format, refer to the config files in `./configs`.
        optimizer_cfg_dict=None,  # To check the data formart, refer to the config files in `./configs`.
        local_rank=-1,  # `local_rank=-1` means test mode. We init the pipeline with test mode by default.
        training=False,  # Init the pipeline for testing by default.
        resume=False  # If set `True`, we will restart the experiment from previous checkpoint.
    ):
        self.module_cfg_dict = module_cfg_dict
        self.optimizer_cfg_dict = optimizer_cfg_dict

        self.init_ppl_modules()
        self.device()
        self.training = training

        # We note that in practical, the `lr` of AdamW is reset from the outside,
        # using cosine annealing during the while training process.
        if training:
            self.optimG = AdamW(
                itertools.chain(
                    filter(lambda p: p.requires_grad,
                           self.bi_flownet.parameters()),
                    filter(lambda p: p.requires_grad,
                           self.fusionnet.parameters())),
                lr=optimizer_cfg_dict["init_lr"],
                weight_decay=optimizer_cfg_dict["weight_decay"])

        # `local_rank == -1` is used for testing, which does not need DDP
        if local_rank != -1:
            if not vars(module_cfg_dict["bi_flownet"]).get(
                    "fix_pretrain", False):
                self.bi_flownet = DDP(self.bi_flownet,
                                      device_ids=[local_rank],
                                      output_device=local_rank,
                                      find_unused_parameters=False)
            if not vars(module_cfg_dict["fusionnet"]).get(
                    "fix_pretrain", False):
                self.fusionnet = DDP(self.fusionnet,
                                     device_ids=[local_rank],
                                     output_device=local_rank,
                                     find_unused_parameters=False)

        # Restart the experiment from last saved model, by loading the state of the optimizer
        if resume:
            assert training, "To restart the training, please init the pipeline with training mode!"
            print("Load optimizer state to restart the experiment")
            ckpt_dict = torch.load(optimizer_cfg_dict["ckpt_file"])
            self.optimG.load_state_dict(ckpt_dict["optimizer"])

    def train(self):
        self.bi_flownet.train()
        self.fusionnet.train()

    def eval(self):
        self.bi_flownet.eval()
        self.fusionnet.eval()

    def device(self):
        self.bi_flownet.to(DEVICE)
        self.fusionnet.to(DEVICE)

    @staticmethod
    def convert_state_dict(rand_state_dict, pretrained_state_dict):
        param = {
            k.replace("module.", ""): v
            for k, v in pretrained_state_dict.items()
        }
        param = {
            k: v
            for k, v in param.items() if ((k in rand_state_dict) and (
                rand_state_dict[k].shape == param[k].shape))
        }
        rand_state_dict.update(param)
        return rand_state_dict

    def init_ppl_modules(self, load_pretrain=True):

        def load_pretrained_state_dict(module, module_name, module_args):
            load_pretrain = module_args.load_pretrain \
                    if "load_pretrain" in module_args else True
            if not load_pretrain:
                print("Train %s from random initialization." % module_name)
                return False

            model_file = module_args.model_file \
                    if "model_file" in module_args else ""
            if (model_file == "") or (not os.path.exists(model_file)):
                raise ValueError(
                    "Please set the correct path for pretrained %s!" %
                    module_name)

            rand_state_dict = module.state_dict()
            pretrained_state_dict = torch.load(model_file)

            return Pipeline.convert_state_dict(rand_state_dict,
                                               pretrained_state_dict)

        # init instance for bi_flownet and fusionnet
        bi_flownet_args = self.module_cfg_dict["bi_flownet"]
        self.bi_flownet = BiFlowNet(bi_flownet_args)
        fusionnet_args = self.module_cfg_dict["fusionnet"]
        self.fusionnet = FusionNet(fusionnet_args)

        # load pretrained model by default
        state_dict = load_pretrained_state_dict(self.bi_flownet, "bi_flownet",
                                                bi_flownet_args)
        if state_dict:
            self.bi_flownet.load_state_dict(state_dict)
        state_dict = load_pretrained_state_dict(self.fusionnet, "fusionnet",
                                                fusionnet_args)
        if state_dict:
            self.fusionnet.load_state_dict(state_dict)

    def save_optimizer_state(self, path, rank, step):
        if rank == 0:
            optimizer_ckpt = {
                "optimizer": self.optimG.state_dict(),
                "step": step
            }
            torch.save(optimizer_ckpt, "{}/optimizer-ckpt.pth".format(path))

    def save_model(self, path, rank, save_step=None):
        if (rank == 0) and (save_step is None):
            torch.save(self.bi_flownet.state_dict(),
                       '{}/bi-flownet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(),
                       '{}/fusionnet.pkl'.format(path))
        if (rank == 0) and (save_step is not None):
            torch.save(self.bi_flownet.state_dict(),
                       '{}/bi-flownet-{}.pkl'.format(path, save_step))
            torch.save(self.fusionnet.state_dict(),
                       '{}/fusionnet-{}.pkl'.format(path, save_step))

    def inference(self, img0, img1, time_period=0.5):
        bi_flow = self.bi_flownet(img0, img1)
        interp_img = self.fusionnet(img0,
                                    img1,
                                    bi_flow,
                                    time_period=time_period)
        return interp_img, bi_flow

    def train_one_iter(self,
                       img0,
                       img1,
                       gt,
                       learning_rate=0,
                       bi_flow_gt=None,
                       time_period=0.5,
                       loss_type="l2+0.1census"):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()

        bi_flow = self.bi_flownet(img0, img1)
        interp_img = self.fusionnet(img0,
                                    img1,
                                    bi_flow,
                                    time_period=time_period)

        if loss_type == "l2":
            loss_l2 = (((interp_img - gt)**2 + 1e-6)**0.5).mean()
            loss_G = loss_l2
        elif loss_type == "l2+census":
            loss_l2 = (((interp_img - gt)**2 + 1e-6)**0.5).mean()
            loss_ter = self.ter(interp_img, gt).mean()
            loss_G = loss_l2 + loss_ter
        elif loss_type == "l2+0.1census":
            loss_l2 = (((interp_img - gt)**2 + 1e-6)**0.5).mean()
            loss_ter = 0.1 * self.ter(interp_img, gt).mean()
            loss_G = loss_l2 + loss_ter
        elif loss_type == "l1":
            loss_l1 = (interp_img - gt).abs().mean()
            loss_G = loss_l1
            with torch.no_grad():
                loss_l2 = (((interp_img - gt)**2 + 1e-6)**0.5).mean()
        elif loss_type == "lap":
            loss_interp_lapl1 = (self.laploss(interp_img, gt)).mean()
            loss_G = loss_interp_lapl1
            with torch.no_grad():
                loss_l2 = (((interp_img - gt)**2 + 1e-6)**0.5).mean()
        elif loss_type == "l2+lap":
            loss_l2 = (((interp_img - gt)**2 + 1e-6)**0.5).mean()
            loss_interp_lapl1 = (self.laploss(interp_img, gt)).mean()
            loss_G = loss_l2 + loss_interp_lapl1
        else:
            raise ValueError("Unsupported loss type!")

        self.optimG.zero_grad()
        loss_G.backward()
        self.optimG.step()

        return interp_img, bi_flow, loss_l2


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass
