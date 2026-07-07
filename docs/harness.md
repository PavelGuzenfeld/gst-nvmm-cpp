# Validation harness

Two small, dependency-light tools for judging a pipeline objectively, on any
box, without a bespoke rig. They are element-agnostic: they take a plain
`gst-launch` string and (optionally) ground-truth boxes, so they work for a
detector, a tracker, a converter -- anything you can express as a pipeline.

## `tools/pipeline_bench.py` — honest frame throughput

Counts the buffers crossing a named element's src pad and times the window from
the first such buffer to the last, so the rate reflects steady-state processing
and excludes pipeline preroll and one-time engine/CUDA warmup (which would
otherwise depress the number by a different amount for every clip length). It
also deliberately ignores any in-band FPS HUD (for example an overlay's
exponential-moving-average counter): an EMA of inter-frame gaps over-reports
2-3x when a queue drains in a burst, so it is the wrong number to judge
throughput by. Count buffers at a pad instead.

Pure-software smoke (no hardware, no engine):

```
tools/pipeline_bench.py --probe t \
  --pipeline "videotestsrc num-buffers=300 ! video/x-raw,width=1920,height=1080 \
              ! identity name=t ! fakesink"
```

Detector throughput on Jetson (average over 3 runs):

```
tools/pipeline_bench.py --probe infer --iterations 3 --json infer.json \
  --pipeline "filesrc location=clip.mp4 ! parsebin ! nvv4l2decoder \
              ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
              ! nvmminfer name=infer engine-file=y.engine ! fakesink"
```

`--probe` is the `name=` of the element whose src pad is counted. Exit 0 if
every iteration reached EOS, 1 on a pipeline error. For an unattended sweep pass
`--timeout <seconds>` so a stalled or non-prerolling pipeline is abandoned and
reported as an error instead of hanging the batch.

## `tools/score_pr.py` — precision / recall vs ground truth

Both inputs are JSONL, one object per frame, boxes in `xywh` pixels:

```
{"frame": 12, "boxes": [[x, y, w, h, score], ...]}   # score optional
```

Per frame it greedily matches predicted to GT boxes best-IoU-first; a pair is a
true positive at `IoU >= --iou`. Predictions below `--conf` are dropped first.

```
tools/score_pr.py --pred pred.jsonl --gt gt.jsonl --iou 0.5 --conf 0.3 --json pr.json
```

Prints precision, recall, F1 and mean matched IoU. It scores; it does not gate
(always exit 0). If the two files barely share frame indices it warns on stderr:
that usually means they number frames differently (an off-by-one), which would
otherwise show up as a silent precision/recall of zero.

## Recording an overlay clip alongside a run

Overlay recording is just part of the pipeline string — put `nvmmdrawdet` and an
encoder tee in your launch, and the same run that `pipeline_bench` measures also
writes the annotated MP4:

```
... ! nvmmdrawdet draw-det=false ! nvvidconv ! nvv4l2h264enc ! h264parse \
    ! qtmux ! filesink location=overlay.mp4
```

`draw-det=false` draws only the fused track box + HUD (the tracker's result),
not the raw per-frame detections. See `scripts/nvmminfer_overlay_e2e.sh` for a
complete detector + overlay example.
