import argparse
import warnings

import torch
from torch.nn import functional as F

warnings.simplefilter("ignore", UserWarning)


def build_model(name, device="cuda"):
    if name.lower() == "edsc":
        from models.edsc.EDSC import Network

        model = Network()
        model.load_state_dict(
            torch.load("./checkpoints/edsc/EDSC_s_l1.ckpt", map_location="cuda:0")[
                "model_state"
            ]
        )
        model.to(device)
        model.eval()

        def infer(I1, I2):
            divide = 32
            _, _, H, W = I1.size()
            H_padding = (divide - H % divide) % divide
            W_padding = (divide - W % divide) % divide
            inps = [
                F.pad(x, (0, W_padding, 0, H_padding), "constant") for x in [I1, I2]
            ]
            out = model(inps)
            out = torch.clamp(out, 0.0, 1.0)
            out = out[..., :H, :W]
            return out

    elif "xvfi" in name.lower():
        from models.xvfi.XVFINet import XVFInet

        class xvfiArgs:
            S_trn = 3
            S_tst = 5
            module_scale_factor = 4
            gpu = 0
            img_ch = 3
            nf = 64

        args = xvfiArgs()
        if "vimeo" in name.lower():
            args.S_trn = 1
            args.S_tst = 1
            args.module_scale_factor = 2
            checkpoint = torch.load(
                "./checkpoints/XVFI/XVFInet_Vimeo_exp1_latest.pt", map_location="cuda:0"
            )
        else:
            args.S_trn = 3
            args.S_tst = 5
            args.module_scale_factor = 4
            checkpoint = torch.load(
                "./checkpoints/XVFI/XVFInet_X4K1000FPS_exp1_latest.pt",
                map_location="cuda:0",
            )
        model = XVFInet(args).to(device)
        model.load_state_dict(checkpoint["state_dict_Model"])
        model.eval()

        def infer(I1, I2, time):
            divide = 2 ** (args.S_tst) * args.module_scale_factor * 4
            inp_frames = torch.cat([x[:, :, None, ...] for x in [I1, I2]], dim=2)
            _, _, _, H, W = inp_frames.size()
            H_padding = (divide - H % divide) % divide
            W_padding = (divide - W % divide) % divide
            if H_padding != 0 or W_padding != 0:
                inp_frames = F.pad(inp_frames, (0, W_padding, 0, H_padding), "constant")
            inp_frames = (inp_frames - 0.5) * 2
            time = torch.Tensor([[time]]).to(device)
            out = model(inp_frames, time, is_training=False)
            out = torch.clamp((out + 1) * 0.5, 0.0, 1.0)
            if H_padding != 0 or W_padding != 0:
                out = out[:, :, :H, :W]
            return out

    elif "rife" in name.lower():
        from models.rife.RIFE_HDv3 import Model

        ppl = Model()
        ppl.load_model("checkpoints/RIFE.pkl", -1)
        ppl.device()
        ppl.eval()
        model = ppl.flownet

        def infer(I1, I2):
            _, _, h, w = I1.size()
            tmp = 32
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)
            inps = [F.pad(x, padding, "constant") for x in [I1, I2]]
            out = ppl.inference(inps[0], inps[1])
            out = torch.clamp(out, 0.0, 1.0)
            out = out[..., :h, :w]
            return out

    elif "ebme" in name.lower():
        from models.ebme.core.unified_ppl import Pipeline

        bi_flownet_args = argparse.Namespace()
        bi_flownet_args.pyr_level = 3
        bi_flownet_args.load_pretrain = True
        bi_flownet_args.model_file = "checkpoints/EBME/bi-flownet.pkl"

        fusionnet_args = argparse.Namespace()
        fusionnet_args.high_synthesis = False
        fusionnet_args.load_pretrain = True
        fusionnet_args.model_file = "checkpoints/EBME/fusionnet.pkl"

        module_cfg_dict = dict(bi_flownet=bi_flownet_args, fusionnet=fusionnet_args)

        ppl = Pipeline(module_cfg_dict)
        ppl.eval()
        model = torch.nn.Sequential(ppl.bi_flownet, ppl.fusionnet)

        def infer(I1, I2, time):
            divide = 16
            _, _, H, W = I1.size()
            H_padding = (divide - H % divide) % divide
            W_padding = (divide - W % divide) % divide
            inps = [
                F.pad(x, (0, W_padding, 0, H_padding), "constant", 0.5)
                for x in [I1, I2]
            ]
            out, _ = ppl.inference(inps[0], inps[1], time)
            out = torch.clamp(out, 0.0, 1.0)
            out = out[..., :H, :W]
            return out

    elif "stmfnet" in name.lower():
        from models.stmfnet import STMFNet

        args = argparse.Namespace(
            featc=[64, 128, 256, 512],
            featnet="UMultiScaleResNext",
            featnorm="batch",
            kernel_size=5,
            dilation=1,
            finetune_pwc=False,
        )
        model = STMFNet(args)
        checkpoint = torch.load("checkpoints/stmfnet/stmfnet.pth")
        model.load_state_dict(checkpoint["state_dict"])
        model.cuda().eval()

        def infer(I1, I2, I3, I4):
            out = model(I1, I2, I3, I4)
            return torch.clamp(out, 0.0, 1.0)

    elif "vfiformer" in name.lower():
        from models.vfiformer.modules import define_G

        args = argparse.Namespace(
            gpu_ids=[0],
            dist=False,
            phase="test",
            crop_size=192,
            resume_flownet=False,
        )
        args.device = torch.device("cuda" if len(args.gpu_ids) != 0 else "cpu")
        model = define_G(args).module.eval()
        model.load_state_dict(torch.load("checkpoints/VFIFormer.pth"))

        def infer(I1, I2, time):
            divide = 64
            _, _, H, W = I1.size()
            H_padding = (divide - H % divide) % divide
            W_padding = (divide - W % divide) % divide
            I1, I2 = [
                F.pad(x, (0, W_padding, 0, H_padding), "constant", 0.0)
                for x in [I1, I2]
            ]
            pred, _ = model(I1, I2, None)
            pred = pred[..., :H, :W]
            return pred

    elif "amt" in name.lower():
        from omegaconf import OmegaConf

        from models.AMT.utils.build_utils import build_from_cfg

        if "-s" in name.lower():
            cfg_path = "models/AMT/cfgs/AMT-S.yaml"
            ckpt_path = "checkpoints/AMT/amt-s.pth"
        elif "-g" in name.lower():
            cfg_path = "models/AMT/cfgs/AMT-G.yaml"
            ckpt_path = "checkpoints/AMT/amt-g.pth"
        else:
            print("Use AMT-S by default.")
            cfg_path = "models/AMT/cfgs/AMT-S.yaml"
            ckpt_path = "checkpoints/AMT/amt-s.pth"

        network_cfg = OmegaConf.load(cfg_path).network
        model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()

        def infer(I0, I1, time):
            divide = 16
            _, _, H, W = I0.shape
            H_padding = (((H // divide) + 1) * divide - H) % divide
            W_padding = (((W // divide) + 1) * divide - W) % divide
            _pad = [
                W_padding // 2,
                W_padding - W_padding // 2,
                H_padding // 2,
                H_padding - H_padding // 2,
            ]
            I0 = F.pad(I0, _pad, mode="replicate")
            I1 = F.pad(I1, _pad, mode="replicate")
            embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(device)

            pred = model(I0, I1, embt, scale_factor=1, eval=True)["imgt_pred"]
            pred = pred[..., _pad[2] : H + _pad[2], _pad[0] : W + _pad[0]]
            return torch.clamp(pred, 0.0, 1.0)

    elif "ema" in name.lower():
        from models.ema_vfi import Model

        model = Model(-1)
        model.load_model()
        model.eval()
        model.device()

        def infer(I0, I1, time):
            divide = 32
            _, _, H, W = I0.shape
            H_padding = (((H // divide) + 1) * divide - H) % divide
            W_padding = (((W // divide) + 1) * divide - W) % divide
            _pad = [
                W_padding // 2,
                W_padding - W_padding // 2,
                H_padding // 2,
                H_padding - H_padding // 2,
            ]
            I0 = F.pad(I0, _pad, mode="replicate")
            I1 = F.pad(I1, _pad, mode="replicate")
            pred = model.multi_inference(
                I0, I1, TTA=True, time_list=[time], fast_TTA=True
            )[0][None]
            pred = pred[..., _pad[2] : H + _pad[2], _pad[0] : W + _pad[0]]
            return torch.clamp(pred, 0.0, 1.0)

    elif "pervfi" in name.lower():
        from models.pervfi.pipeline import Pipeline_infer

        ckpt = "checkpoints/PerVFI/v00.pth"
        # i.e., RAFT+PerVFI
        ofnet = name.split("+")[0]
        ofnet = None if ofnet == "none" else ofnet

        model = Pipeline_infer(ofnet, "v00", model_file=ckpt)

        def infer(I1, I2, time=0.5):
            divide = 8
            _, _, H, W = I1.size()
            H_padding = (divide - H % divide) % divide
            W_padding = (divide - W % divide) % divide
            I1, I2 = [
                F.pad(x, (0, W_padding, 0, H_padding), "constant", 0.0)
                for x in [I1, I2]
            ]
            pred = model.inference_rand_noise(I1, I2, heat=0.3, time=time)
            return pred[..., :H, :W]

    elif "copy" in name.lower():
        model = None

        def infer(I1, I2, time):
            return I1

    return model, infer
