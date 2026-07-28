#!/usr/bin/env python3
"""Compare two nvmmfusekf per-frame box dumps -- a tracked path against a baseline.

Reads the CSV that nvmmfusekf writes when $NVMMFUSEKF_CSV is set
(frame,valid,left,top,width,height,score,yolo_fused) from a baseline run and a
test run of the SAME clip, and judges whether the test run tracks the same path.

This is the behavioural-parity gate for changes that alter the math rather than
just the placement of work -- detector decimation (nvmminfer infer-interval),
speculative cropping, precision changes. It reads *fusekf's emitted box*, which
is what downstream consumers actually receive: nvmmfusekf takes the best YOLO
detection as a per-frame gated measurement and its flush-BB path can publish the
tight detection box, so anything that removes detections on some frames moves
this signal on those frames -- not merely at reseed.

Default thresholds are the agreed bar: per-frame IoU >= 0.99 median, no frame
below 0.9, and no valid-flag flips.

    trajectory_compare.py --baseline base.csv --test n3.csv
                          [--median-iou 0.99] [--min-iou 0.9]
                          [--allow-flips 0] [--verbose]

Frames present in one run but not the other are reported as missing and fail the
comparison -- a differing frame count means the runs are not comparable.
Frames where BOTH runs are invalid (no track) are excluded from the IoU stats
but still checked for flag agreement: IoU is undefined with no box.

Exit 0 on PASS, 1 on FAIL.
"""
import argparse
import csv
import statistics as st
import sys


def read_dump(path):
    """frame -> dict. Tolerates the header and any trailing blank line."""
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("frame"):
                continue
            rows[int(r["frame"])] = {
                "valid": int(r["valid"]),
                "x": float(r["left"]), "y": float(r["top"]),
                "w": float(r["width"]), "h": float(r["height"]),
                "score": float(r["score"]),
                "yolo_fused": int(r["yolo_fused"]),
            }
    return rows


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
    ap.add_argument("--baseline", required=True, help="reference run's NVMMFUSEKF_CSV")
    ap.add_argument("--test", required=True, help="candidate run's NVMMFUSEKF_CSV")
    ap.add_argument("--median-iou", type=float, default=0.99)
    ap.add_argument("--min-iou", type=float, default=0.9)
    ap.add_argument("--allow-flips", type=int, default=0,
                    help="tolerated valid-flag disagreements")
    ap.add_argument("--verbose", action="store_true",
                    help="list the worst frames and every flag flip")
    a = ap.parse_args()

    base, test = read_dump(a.baseline), read_dump(a.test)
    if not base:
        print(f"FAIL: baseline {a.baseline} has no rows")
        return 1

    missing = sorted(set(base) ^ set(test))
    common = sorted(set(base) & set(test))

    ious, flips, both_invalid = [], [], 0
    for n in common:
        b, t = base[n], test[n]
        if b["valid"] != t["valid"]:
            flips.append((n, b["valid"], t["valid"]))
            continue
        if not b["valid"]:
            both_invalid += 1
            continue
        ious.append((n, iou(b, t)))

    # YOLO's actual contribution rate, to show the mechanism rather than infer it.
    fused_base = sum(v["yolo_fused"] for v in base.values())
    fused_test = sum(v["yolo_fused"] for v in test.values())

    print(f"frames: baseline={len(base)} test={len(test)} common={len(common)}"
          f" missing={len(missing)}")
    print(f"yolo-fused frames: baseline={fused_base}"
          f" ({100.0 * fused_base / max(1, len(base)):.1f}%)"
          f"  test={fused_test} ({100.0 * fused_test / max(1, len(test)):.1f}%)")
    print(f"both-invalid frames excluded from IoU: {both_invalid}")

    fails = []
    if missing:
        fails.append(f"{len(missing)} frame(s) in only one run (first: {missing[:5]})")
    if len(flips) > a.allow_flips:
        fails.append(f"{len(flips)} valid-flag flip(s) > allowed {a.allow_flips}")

    if ious:
        vals = [v for _, v in ious]
        med = st.median(vals)
        worst = min(ious, key=lambda p: p[1])
        below = [(n, v) for n, v in ious if v < a.min_iou]
        vs = sorted(vals)
        print(f"IoU over {len(vals)} compared frames: median={med:.4f}"
              f" mean={st.fmean(vals):.4f} p05={vs[len(vs) // 20]:.4f}"
              f" min={worst[1]:.4f} @frame {worst[0]}")
        if med < a.median_iou:
            fails.append(f"median IoU {med:.4f} < {a.median_iou}")
        if below:
            fails.append(f"{len(below)} frame(s) below min-IoU {a.min_iou}"
                         f" (worst: frame {worst[0]} @ {worst[1]:.4f})")
        if a.verbose:
            for n, v in sorted(ious, key=lambda p: p[1])[:10]:
                print(f"  worst frame {n}: IoU={v:.4f}")
    else:
        fails.append("no frames were comparable (all invalid or flipped)")

    if a.verbose and flips:
        for n, bv, tv in flips[:20]:
            print(f"  flag flip frame {n}: baseline valid={bv} test valid={tv}")

    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
