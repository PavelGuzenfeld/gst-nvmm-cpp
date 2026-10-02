#!/usr/bin/env python3
"""Export XFeat and LighterGlue to the xfeat.onnx / lightglue.onnx contracts gst/common/xfeat_matcher.cpp binds.

LightGlue stops before sigmoid_log_double_softmax: the C++ filter_matches takes raw sim and z logits.
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from kornia.feature.lightglue import LightGlue
from kornia.feature.lightglue import filter_matches, sigmoid_log_double_softmax

XH, XW = 256, 480
LIGHTERGLUE_CONF = {
    "name": "xfeat", "input_dim": 64, "descriptor_dim": 96, "add_scale_ori": False,
    "add_laf": False, "scale_coef": 1.0, "n_layers": 6, "num_heads": 1, "flash": False,
    "mp": False, "depth_confidence": -1, "width_confidence": -1, "filter_threshold": 0.1,
    "weights": None,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xfeat-src", required=True, help="accelerated_features checkout")
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def _reshape_unflatten(self, dim, sizes):
    """Export shim: the TorchScript exporter loses the rank of unflatten on a dynamic axis, and the next transpose fails."""
    dim = dim % self.dim()
    return self.reshape(*self.shape[:dim], *sizes, *self.shape[dim + 1:])


def _expand_repeat_interleave(self, repeats, dim):
    d = dim % self.dim()
    x = self.unsqueeze(d + 1).expand(*self.shape[:d + 1], repeats, *self.shape[d + 1:])
    return x.reshape(*self.shape[:d], self.shape[d] * repeats, *self.shape[d + 1:])


def load_lighterglue(path):
    LightGlue.default_conf = LIGHTERGLUE_CONF
    lg = LightGlue(None).eval()
    sd = torch.load(path, map_location="cpu")
    for i in range(lg.conf.n_layers):
        sd = {k.replace(f"self_attn.{i}", f"transformers.{i}.self_attn"): v for k, v in sd.items()}
        sd = {k.replace(f"cross_attn.{i}", f"transformers.{i}.cross_attn"): v for k, v in sd.items()}
    sd = {k.replace("matcher.", ""): v for k, v in sd.items() if not k.startswith("extractor.")}
    missing, unexpected = lg.load_state_dict(sd, strict=False)
    if set(missing) - {"confidence_thresholds"} or unexpected:
        sys.exit(f"lighterglue weights do not fit kornia LightGlue: missing={missing} unexpected={unexpected}")
    return lg


class LighterGlueLogits(nn.Module):
    def __init__(self, lg):
        super().__init__()
        self.lg = lg

    def forward(self, desc0, desc1, nkpts0, nkpts1):
        lg = self.lg
        d0, d1 = lg.input_proj(desc0), lg.input_proj(desc1)
        e0, e1 = lg.posenc(nkpts0), lg.posenc(nkpts1)
        for layer in lg.transformers:
            d0, d1 = layer(d0, d1, e0, e1)
        la = lg.log_assignment[-1]
        m0, m1 = la.final_proj(d0), la.final_proj(d1)
        scale = m0.shape[-1] ** 0.25
        sim = torch.einsum("bmd,bnd->bmn", m0 / scale, m1 / scale)
        return sim, la.matchability(d0).squeeze(-1), la.matchability(d1).squeeze(-1)


def export(model, args, path, names_in, names_out, opset, dynamic=None):
    with torch.no_grad():
        torch.onnx.export(model, args, path, input_names=names_in, output_names=names_out,
                          opset_version=opset, dynamic_axes=dynamic, do_constant_folding=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: a.numpy() for n, a in zip(names_in, args)})
    with torch.no_grad():
        want = model(*args)
    for n, g, w in zip(names_out, got, want):
        err = float(np.abs(g - w.numpy()).max() / np.abs(w.numpy()).max())
        print(f"  {os.path.basename(path)} {n} {tuple(g.shape)} max|ort-torch|/max|torch|={err:.2e}", flush=True)
        if err > 1e-4:
            sys.exit(f"{path}: {n} diverges from torch by {err}")


def check_logits_match_stock(lg, wrapper):
    torch.manual_seed(1)
    n0, n1 = 300, 280
    k0, k1 = torch.rand(1, n0, 2) * 2 - 1, torch.rand(1, n1, 2) * 2 - 1
    d0 = nn.functional.normalize(torch.randn(1, n0, 64), dim=-1)
    d1 = nn.functional.normalize(torch.randn(1, n1, 64), dim=-1)
    d1[:, :200] = d0[:, :200] + 0.01 * torch.randn(1, 200, 64)
    k1[:, :200] = k0[:, :200]
    with torch.no_grad():
        sim, z0, z1 = wrapper(d0, d1, k0, k1)
        ours = sigmoid_log_double_softmax(sim, z0.unsqueeze(-1), z1.unsqueeze(-1))
        size = torch.tensor([[2.0, 2.0]])
        stock = lg({"image0": {"keypoints": k0 + 1, "descriptors": d0, "image_size": size},
                    "image1": {"keypoints": k1 + 1, "descriptors": d1, "image_size": size}})
    if not torch.equal(stock["matches0"], filter_matches(ours, LIGHTERGLUE_CONF["filter_threshold"])[0]):
        sys.exit("wrapper argmax disagrees with stock LightGlue matches0")
    print(f"  stock LightGlue agrees on {(stock['matches0'] >= 0).sum().item()} matches", flush=True)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, a.xfeat_src)
    from modules.model import XFeatModel

    xf = XFeatModel().eval()
    xf.load_state_dict(torch.load(os.path.join(a.xfeat_src, "weights", "xfeat.pt"), map_location="cpu"))
    export(xf, (torch.rand(1, 3, XH, XW),), os.path.join(a.out, "xfeat.onnx"),
           ["image"], ["feats", "keypoints", "heatmap"], a.opset)

    lg = load_lighterglue(os.path.join(a.xfeat_src, "weights", "xfeat-lighterglue.pt"))
    wrapper = LighterGlueLogits(lg).eval()
    check_logits_match_stock(lg, wrapper)
    torch.Tensor.unflatten = _reshape_unflatten
    torch.Tensor.repeat_interleave = _expand_repeat_interleave
    n0, n1 = 200, 150
    args = (nn.functional.normalize(torch.randn(1, n0, 64), dim=-1),
            nn.functional.normalize(torch.randn(1, n1, 64), dim=-1),
            torch.rand(1, n0, 2) * 2 - 1, torch.rand(1, n1, 2) * 2 - 1)
    export(wrapper, args, os.path.join(a.out, "lightglue.onnx"),
           ["desc0", "desc1", "nkpts0", "nkpts1"], ["sim", "z0", "z1"], a.opset,
           dynamic={"desc0": {1: "N0"}, "nkpts0": {1: "N0"}, "desc1": {1: "N1"},
                    "nkpts1": {1: "N1"}, "sim": {1: "N0", 2: "N1"}, "z0": {1: "N0"},
                    "z1": {1: "N1"}})
    print("XFEAT_EXPORT_DONE", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
