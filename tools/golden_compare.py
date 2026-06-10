#!/usr/bin/env python3
"""Compare nvmminfer's detections against the independent ONNX reference.

Greedy-matches each reference box to an actual box of the SAME class with the
best IoU. PASS iff, for all detections at/above --conf: counts match, every
reference box has a match with IoU >= --iou, and each matched pair's confidence
agrees within --conf-tol (fp16-TRT vs fp32-ONNX numeric slack).

    golden_compare.py --reference ref.json --actual-log nvmminfer.log
                      [--conf 0.3] [--iou 0.5] [--conf-tol 0.15]

Exit 0 on PASS, 1 on FAIL.
"""
import argparse
import json
import re
import sys

# nvmminfer GST_LOG line: "  [0] bus 0.92  box=(20,230 480x520)"
LINE = re.compile(
    r"\[\d+\]\s+(?P<label>.+?)\s+(?P<conf>[01]?\.\d+)\s+box=\("
    r"(?P<x>-?[\d.]+),(?P<y>-?[\d.]+)\s+(?P<w>[\d.]+)x(?P<h>[\d.]+)\)"
)


def parse_log(path):
    dets = []
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                dets.append({
                    "label": m["label"].strip(),
                    "conf": float(m["conf"]),
                    "x": float(m["x"]), "y": float(m["y"]),
                    "w": float(m["w"]), "h": float(m["h"]),
                })
    return dets


def iou(a, b):
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--actual-log", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-tol", type=float, default=0.15)
    a = ap.parse_args()

    with open(a.reference) as f:
        ref = [d for d in json.load(f)["detections"] if d["conf"] >= a.conf]
    act = [d for d in parse_log(a.actual_log) if d["conf"] >= a.conf]

    print(f"reference: {len(ref)} det(s) >= {a.conf}; actual: {len(act)} det(s) >= {a.conf}")

    # Floor: a reference with detections but an empty parse means the log format
    # changed or GST_DEBUG=nvmminfer:6 was missing — fail loudly, don't pass on 0.
    if ref and not act:
        print("GOLDEN FAIL: reference has detections but none parsed from the "
              "actual log (format change or missing GST_DEBUG=nvmminfer:6?)")
        return 1

    used = [False] * len(act)
    failures = []
    for r in sorted(ref, key=lambda d: -d["conf"]):
        best, best_iou = -1, 0.0
        for j, c in enumerate(act):
            if used[j] or c["label"] != r["label"]:
                continue
            v = iou(r, c)
            if v > best_iou:
                best, best_iou = j, v
        if best < 0 or best_iou < a.iou:
            failures.append(f"MISS {r['label']} {r['conf']:.2f} "
                            f"box=({r['x']:.0f},{r['y']:.0f} {r['w']:.0f}x{r['h']:.0f}) "
                            f"best IoU={best_iou:.2f}")
            continue
        used[best] = True
        c = act[best]
        dconf = abs(r["conf"] - c["conf"])
        status = "OK  " if dconf <= a.conf_tol else "CONF"
        print(f"  {status} {r['label']:<12} IoU={best_iou:.3f} "
              f"conf ref={r['conf']:.2f} act={c['conf']:.2f} d={dconf:.2f}")
        if dconf > a.conf_tol:
            failures.append(f"CONF {r['label']} delta {dconf:.2f} > {a.conf_tol}")

    for j, c in enumerate(act):
        if not used[j]:
            failures.append(f"EXTRA {c['label']} {c['conf']:.2f} "
                            f"box=({c['x']:.0f},{c['y']:.0f} {c['w']:.0f}x{c['h']:.0f})")

    if failures:
        print("GOLDEN FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"GOLDEN PASS: {len(ref)} detection(s) matched within "
          f"IoU>={a.iou}, conf-tol={a.conf_tol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
