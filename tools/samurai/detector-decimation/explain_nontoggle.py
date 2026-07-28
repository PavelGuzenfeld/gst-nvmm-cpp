#!/usr/bin/env python3
"""Why do frames where yolo_fused AGREES still fall below the IoU floor?"""
import csv
import statistics as st
import sys


def rd(p):
    out = {}
    for r in csv.DictReader(open(p)):
        if r.get("frame"):
            r["valid"] = int(r["valid"])
            out[int(r["frame"])] = r
    return out


def box(r):
    return (float(r["left"]), float(r["top"]), float(r["width"]), float(r["height"]))


def iou(a, c):
    ax, ay, aw, ah = box(a)
    cx, cy, cw, ch = box(c)
    ix = max(0.0, min(ax + aw, cx + cw) - max(ax, cx))
    iy = max(0.0, min(ay + ah, cy + ch) - max(ay, cy))
    inter = ix * iy
    u = aw * ah + cw * ch - inter
    return inter / u if u > 0 else 0.0


b, t = rd(sys.argv[1]), rd(sys.argv[2])
# Skip frames invalid in BOTH runs -- with seed-delay these are the pre-seed frames,
# which have no box at all. Scoring them gives IoU 0.0 and swamps the statistics
# (they are not disagreements, they are "tracking has not started").
common = [f for f in sorted(b)
          if f in t and not (b[f]["valid"] == 0 and t[f]["valid"] == 0)]
print(f"comparable frames: {len(common)} "
      f"(excluded {sum(1 for f in b if f in t) - len(common)} invalid-in-both)")
tog = {f for f in common if b[f]["yolo_fused"] != t[f]["yolo_fused"]}
bad = [(f, iou(b[f], t[f])) for f in common
       if f not in tog and iou(b[f], t[f]) < 0.9]

print(f"non-toggle frames below 0.9: {len(bad)}")
print("their fused flag values:", sorted({b[f]['yolo_fused'] for f, _ in bad}))

runs, prev = [], None
for f, _ in bad:
    if prev is None or f - prev > 3:
        runs.append([f, f])
    else:
        runs[-1][1] = f
    prev = f
print(f"clustered into {len(runs)} runs; first 8: {runs[:8]}")

near = sum(1 for f, _ in bad if any((f + d) in tog for d in range(-5, 6)))
print(f"within 5 frames of a toggle: {near}/{len(bad)}")

# Centre agreement vs scale agreement, to separate drift from size mismatch.
cds, srs = [], []
for f, _ in bad:
    ax, ay, aw, ah = box(b[f])
    cx, cy, cw, ch = box(t[f])
    cds.append((((ax + aw / 2) - (cx + cw / 2)) ** 2 + ((ay + ah / 2) - (cy + ch / 2)) ** 2) ** 0.5)
    srs.append((cw * ch) / (aw * ah) if aw * ah > 0 else 0.0)
if bad:
    print(f"centre distance px: median={st.median(cds):.1f} max={max(cds):.1f}")
    print(f"area ratio test/base: median={st.median(srs):.2f} min={min(srs):.2f} max={max(srs):.2f}")
for f, v in bad[:6]:
    print(f"  f={f} IoU={v:.3f} fused={b[f]['yolo_fused']} "
          f"base={box(b[f])} test={box(t[f])}")
