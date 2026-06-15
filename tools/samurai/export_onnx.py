#!/usr/bin/env python3
"""Export the SAM2.1 / SAMURAI sub-models to ONNX for the nvmmsamurai element.

Produces five ONNX graphs whose I/O contracts match what gst/nvmmsamurai loads:
  image_encoder, prompt_encoder, mask_decoder, memory_encoder, memory_attention.

Everything is built from the **public** SAM2.1 base_plus checkpoint + the public
SAMURAI repo (see docs/building-engines.md). No captured/internal data is used:
trace inputs are synthetic tensors of the exact shapes the runtime expects (ONNX
export records the graph, not the values).

The encoder runs at a 512x512 crop (not SAM2's native 1024), so image_size is
overridden to 512 -> image_embed 32x32, feat_s0 128x128, feat_s1 64x64.

Usage (inside a PyTorch container, with the public samurai repo on PYTHONPATH):
  python3 export_onnx.py \
      --ckpt /work/ckpt/sam2.1_hiera_base_plus.pt \
      --out  /work/onnx \
      [--config configs/samurai/sam2.1_hiera_b+.yaml] [--image-size 512] [--device cuda]
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Public SAM2 (vendored under the SAMURAI repo as the `sam2` package).
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam import transformer as TR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="public sam2.1_hiera_base_plus.pt")
    p.add_argument("--out", required=True, help="output dir for the .onnx files")
    p.add_argument("--config", default="configs/samurai/sam2.1_hiera_b+.yaml",
                   help="hydra config name (resolved from the sam2 package)")
    p.add_argument("--image-size", type=int, default=512,
                   help="encoder crop size (the runtime uses 512, not SAM2's 1024)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


# ONNX-export shim: torch.repeat_interleave builds its index on CPU during trace,
# clashing with cuda inputs. B=1 here so repeat-by-1 is a no-op; build the index
# on-device otherwise.
_orig_ri = torch.repeat_interleave


def _safe_ri(x, repeats, dim=None, output_size=None):
    if dim == 0:
        return x
    if isinstance(repeats, int):
        idx = torch.arange(x.shape[dim], device=x.device).repeat_interleave(repeats)
        return x.index_select(dim, idx)
    return _orig_ri(x, repeats, dim=dim, output_size=output_size)


# --- image_encoder: stock forward returns a dict; flatten to the out1..out6 list
# the C++ binds (out3=pos 32x32, out4=feat_s0 128x128, out5=feat_s1 64x64,
# out6=image_embed 32x32). ----------------------------------------------------
class ImageEnc(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        o = self.enc(x)
        pos = o["vision_pos_enc"]
        fpn = o["backbone_fpn"]
        return pos[0], pos[1], pos[2], fpn[0], fpn[1], fpn[2]


# --- prompt_encoder (BOX seed: 2 corners labels 2,3 + 1 pad label -1). Rewritten
# without boolean-mask scatter (TRT cannot parse it): slice + concat instead. ---
class PEBox(nn.Module):
    def __init__(self, pe):
        super().__init__()
        self.pe = pe

    def forward(self, coords):                     # coords [1,2,2] box corners
        bs = coords.shape[0]
        pts = torch.cat([coords + 0.5, torch.zeros((bs, 1, 2), device=coords.device)], dim=1)
        emb = self.pe.pe_layer.forward_with_coords(pts, self.pe.input_image_size)
        e0 = emb[:, 0:1, :] + self.pe.point_embeddings[2].weight
        e1 = emb[:, 1:2, :] + self.pe.point_embeddings[3].weight
        e2 = emb[:, 2:3, :] * 0.0 + self.pe.not_a_point_embed.weight     # label -1
        sparse = torch.cat([e0, e1, e2], dim=1)
        dense = self.pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, *self.pe.image_embedding_size)
        return sparse, dense


# --- mask_decoder (deploy): raw predict_masks, 4 candidates, batched hypernetwork
# (the stock per-token MLP loop will not export), selection done in C++. ---------
class BatchedHyper(nn.Module):
    def __init__(self, mlps):
        super().__init__()
        self.nl = mlps[0].num_layers
        self.W = nn.ParameterList()
        self.B = nn.ParameterList()
        for li in range(self.nl):
            self.W.append(nn.Parameter(torch.stack([m.layers[li].weight for m in mlps], 0),
                                       requires_grad=False))
            self.B.append(nn.Parameter(torch.stack([m.layers[li].bias for m in mlps], 0),
                                       requires_grad=False))

    def forward(self, mto):
        x = mto
        for li in range(self.nl):
            x = torch.einsum("bni,noi->bno", x, self.W[li]) + self.B[li]
            if li < self.nl - 1:
                x = torch.relu(x)
        return x


class MDdeploy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.hyper = BatchedHyper(d.output_hypernetworks_mlps)

    def forward(self, ie, pe, sp, de, f0, f1):
        d = self.d
        f0r = d.conv_s0(f0)
        f1r = d.conv_s1(f1)
        ot = torch.cat([d.obj_score_token.weight, d.iou_token.weight, d.mask_tokens.weight],
                       0).unsqueeze(0).expand(sp.size(0), -1, -1)
        tokens = torch.cat((ot, sp), 1)
        src = ie + de
        b, c, h, w = src.shape
        hs, src2 = d.transformer(src, pe, tokens)
        iou_tok = hs[:, 1, :]
        mto = hs[:, 2:2 + d.num_mask_tokens, :]
        src2 = src2.transpose(1, 2).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = d.output_upscaling
        up = act2(dc2(act1(ln1(dc1(src2) + f1r))) + f0r)
        hyper = self.hyper(mto)
        bb, cc, hh, ww = up.shape
        masks = (hyper @ up.view(bb, cc, hh * ww)).view(bb, -1, hh, ww)
        ious = d.iou_prediction_head(iou_tok)
        obj = d.pred_obj_score_head(hs[:, 0, :])
        return masks, ious, mto, obj


# --- memory_encoder (skip_mask_sigmoid=True). -----------------------------------
class ME(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.e = e

    def forward(self, pix, mask):
        o = self.e(pix, mask, skip_mask_sigmoid=True)
        return o["vision_features"], o["vision_pos_enc"][0]


# --- memory_attention: replace the complex view_as_complex RoPE with a real-valued
# path TRT can parse, baking cos/sin as constant buffers. ------------------------
def _rope_real(x, cos, sin):                      # x:[B,H,L,D]  cos/sin:[L,D/2]
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x0 = xr[..., 0]
    x1 = xr[..., 1]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.stack([o0, o1], dim=-1).reshape(*x.shape)


def _rope_forward(self, q, k, v, num_k_exclude_rope=0):
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)
    num_k_rope = k.size(-2) - num_k_exclude_rope
    cos, sin = self._rope_cos, self._rope_sin
    q = _rope_real(q, cos, sin)
    k_rope = k[:, :, :num_k_rope]
    if self.rope_k_repeat:
        r = k_rope.shape[-2] // cos.shape[0]
        ck = cos.repeat(r, 1)
        sk = sin.repeat(r, 1)
    else:
        ck, sk = cos, sin
    k = torch.cat([_rope_real(k_rope, ck, sk), k[:, :, num_k_rope:]], dim=2)
    out = F.scaled_dot_product_attention(q, k, v)
    return self.out_proj(self._recombine_heads(out))


class MA(nn.Module):
    def __init__(self, ma, n_optr):
        super().__init__()
        self.ma = ma
        self.n = n_optr

    def forward(self, curr, memory, curr_pos, memory_pos):
        return self.ma(curr=curr, memory=memory, curr_pos=curr_pos,
                       memory_pos=memory_pos, num_obj_ptr_tokens=self.n)


def save_onnx(out_dir, fn, wrapped, args, names_in, names_out, opset, dynamic=None):
    wrapped.eval()
    with torch.no_grad():
        _ = wrapped(*args)
    torch.onnx.export(wrapped, args, os.path.join(out_dir, f"{fn}.onnx"),
                      input_names=names_in, output_names=names_out,
                      opset_version=opset, dynamic_axes=dynamic, do_constant_folding=True)
    print(f"  exported {fn}.onnx", flush=True)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    torch.manual_seed(0)
    torch.repeat_interleave = _safe_ri

    m = build_sam2_video_predictor(
        a.config, ckpt_path=a.ckpt, device=dev, mode="eval",
        hydra_overrides_extra=[f"++model.image_size={a.image_size}"])
    S = a.image_size
    g = S // 16            # embed grid (32 at 512)
    print(f"model built. image_size={S} embed_grid={g}", flush=True)

    # image_encoder
    x = torch.randn(1, 3, S, S, device=dev)
    save_onnx(a.out, "image_encoder", ImageEnc(m.image_encoder), (x,),
              ["input"], ["out1", "out2", "out3", "out4", "out5", "out6"], a.opset)

    # prompt_encoder (box-seed)
    coords = torch.tensor([[[100., 100.], [200., 200.]]], device=dev)
    save_onnx(a.out, "prompt_encoder", PEBox(m.sam_prompt_encoder), (coords,),
              ["coords"], ["sparse", "dense"], a.opset)

    # mask_decoder (deploy: dynamic sparse Np axis)
    ie = torch.randn(1, 256, g, g, device=dev)
    pe = m.sam_prompt_encoder.get_dense_pe().to(dev)
    sp = torch.randn(1, 3, 256, device=dev)
    de = torch.randn(1, 256, g, g, device=dev)
    f0 = torch.randn(1, 256, g * 4, g * 4, device=dev)
    f1 = torch.randn(1, 256, g * 2, g * 2, device=dev)
    save_onnx(a.out, "mask_decoder", MDdeploy(m.sam_mask_decoder),
              (ie, pe, sp, de, f0, f1),
              ["image_embed", "image_pe", "sparse", "dense", "feat_s0", "feat_s1"],
              ["masks", "ious", "tokens", "obj_score"], a.opset,
              dynamic={"sparse": {1: "Np"}})

    # memory_encoder
    pix = torch.randn(1, 256, g, g, device=dev)
    mask = torch.randn(1, 1, S, S, device=dev)
    save_onnx(a.out, "memory_encoder", ME(m.memory_encoder), (pix, mask),
              ["pix_feat", "mask"], ["maskmem_feat", "maskmem_pos"], a.opset)

    # memory_attention (real-RoPE)
    for mod in m.memory_attention.modules():
        if isinstance(mod, TR.RoPEAttention):
            fc = mod.freqs_cis
            mod.register_buffer("_rope_cos", fc.real.contiguous().to(dev), persistent=False)
            mod.register_buffer("_rope_sin", fc.imag.contiguous().to(dev), persistent=False)
    TR.RoPEAttention.forward = _rope_forward
    n_optr = 64                                    # kObjTok: trailing obj_ptr tokens
    hw = g * g                                     # 1024 at 512
    ktotal = 7 * hw + n_optr                        # kMask*kTok + kObjTok = 7232
    curr = torch.randn(hw, 1, 256, device=dev)
    memory = torch.randn(ktotal, 1, 64, device=dev)
    curr_pos = torch.randn(hw, 1, 256, device=dev)
    memory_pos = torch.randn(ktotal, 1, 64, device=dev)
    save_onnx(a.out, "memory_attention", MA(m.memory_attention, n_optr),
              (curr, memory, curr_pos, memory_pos),
              ["curr", "memory", "curr_pos", "memory_pos"], ["attn"], a.opset)

    print("EXPORT_DONE", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
