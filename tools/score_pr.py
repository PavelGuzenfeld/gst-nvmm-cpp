#!/usr/bin/env python3
"""Precision / recall of per-frame boxes against ground truth.

Both --pred and --gt are JSONL, one object per frame:

    {"frame": 12, "boxes": [[x, y, w, h, score], ...]}   # score optional

Boxes are xywh in pixel coordinates. For each frame the predicted boxes are
greedily matched to GT boxes best-IoU-first; a pair is a true positive when
IoU >= --iou. Unmatched predictions are false positives, unmatched GT are false
negatives. Predictions with a score below --conf are dropped before matching.

    score_pr.py --pred pred.jsonl --gt gt.jsonl [--iou 0.5] [--conf 0.3] [--json out.json]

Prints precision, recall, F1 and mean matched IoU. Exit 0 always -- this scores,
it does not gate.
"""
import argparse
import json
import sys


def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load(path):
    frames = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            frames[int(obj["frame"])] = obj.get("boxes", [])
    return frames


def match_frame(preds, gts, thr):
    """Greedy best-IoU matching. Returns (tp, fp, fn, [matched_ious])."""
    candidates = []
    for pi, p in enumerate(preds):
        for gj, g in enumerate(gts):
            v = iou(p, g)
            if v >= thr:
                candidates.append((v, pi, gj))
    candidates.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for v, pi, gj in candidates:
        if pi in used_p or gj in used_g:
            continue
        used_p.add(pi)
        used_g.add(gj)
        matched.append(v)
    tp = len(matched)
    return tp, len(preds) - tp, len(gts) - tp, matched


def main():
    ap = argparse.ArgumentParser(description="Per-frame box precision/recall vs GT.")
    ap.add_argument("--pred", required=True, help="predictions JSONL")
    ap.add_argument("--gt", required=True, help="ground-truth JSONL")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--json", help="write metrics JSON to this path")
    args = ap.parse_args()

    pred, gt = load(args.pred), load(args.gt)
    tp = fp = fn = 0
    all_iou = []
    for frame in sorted(set(pred) | set(gt)):
        preds = [b for b in pred.get(frame, []) if len(b) < 5 or b[4] >= args.conf]
        f_tp, f_fp, f_fn, ious = match_frame(preds, gt.get(frame, []), args.iou)
        tp += f_tp
        fp += f_fp
        fn += f_fn
        all_iou += ious

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(all_iou) / len(all_iou) if all_iou else 0.0
    result = {"iou_thr": args.iou, "conf": args.conf,
              "tp": tp, "fp": fp, "fn": fn,
              "precision": round(precision, 4), "recall": round(recall, 4),
              "f1": round(f1, 4), "mean_iou": round(mean_iou, 4)}
    for k in ("tp", "fp", "fn", "precision", "recall", "f1", "mean_iou"):
        print(f"{k:>10}: {result[k]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
