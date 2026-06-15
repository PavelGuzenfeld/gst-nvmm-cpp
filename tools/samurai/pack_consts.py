#!/usr/bin/env python3
"""Pack the SAM2.1 / SAMURAI out-of-engine constants into samurai_consts.bin.

The nvmmsamurai element loads a handful of learned tensors that live outside the
five TRT engines (temporal pos-enc, no-mem/no-obj embeddings, the obj-ptr
projections, the image positional encoding, and the empty-prompt sparse/dense
embeddings). This gathers them from the **public** SAM2.1 base_plus checkpoint
and writes the self-describing binary the C++ loads via the `consts-file`
property. No model weights are bundled in the repo — you produce the .bin here.

.bin format (little-endian):
  u32 magic 0x5341434E ('SACN'), u32 count
  per tensor: u32 name_len, name bytes, u32 ndim, ndim*u32 dims, f32 data

Usage (same PyTorch container as export_onnx.py):
  python3 pack_consts.py --ckpt /work/ckpt/sam2.1_hiera_base_plus.pt \
                         --out  /work/onnx/samurai_consts.bin [--image-size 512]
"""
import argparse
import os
import struct

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/samurai/sam2.1_hiera_b+.yaml")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    a = parse_args()
    m = build_sam2_video_predictor(
        a.config, ckpt_path=a.ckpt, device=a.device, mode="eval",
        hydra_overrides_extra=[f"++model.image_size={a.image_size}"])
    dev = next(m.parameters()).device

    out = {}

    def add(name, t):
        if t is None:
            print("MISSING", name)
            return
        out[name] = np.ascontiguousarray(t.detach().float().cpu().numpy())

    for nm in ["maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc",
               "no_obj_ptr", "no_obj_embed_spatial"]:
        add(nm, getattr(m, nm, None))
    add("image_pe", m.sam_prompt_encoder.get_dense_pe())
    for mod_name in ["obj_ptr_proj", "obj_ptr_tpos_proj"]:
        mod = getattr(m, mod_name, None)
        if mod is None:
            print("MISSING module", mod_name)
            continue
        for pn, p in mod.named_parameters():
            add(f"{mod_name}.{pn}", p)

    # Empty-prompt sparse + dense for tracking frames: 1 empty point (label -1) + pad.
    with torch.no_grad():
        coords = torch.zeros(1, 1, 2, device=dev)
        labels = -torch.ones(1, 1, dtype=torch.int32, device=dev)
        sparse, dense = m.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    add("empty_sparse", sparse)      # (1,2,256)
    add("dense_no_mask", dense)      # (1,256,g,g)  -> 32x32 at image_size 512

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<II", 0x5341434E, len(out)))
        for name, arr in out.items():
            arr = arr.astype(np.float32)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            f.write(struct.pack("<%dI" % arr.ndim, *arr.shape))
            f.write(arr.tobytes())
    print("PACKED", len(out), "tensors ->", a.out, os.path.getsize(a.out), "bytes")
    for k, v in out.items():
        print("  %-28s %s" % (k, tuple(v.shape)))
    print("PACK_CONSTS_DONE")


if __name__ == "__main__":
    main()
