#!/usr/bin/env python3
"""Measure the real frame throughput of a GStreamer pipeline.

Counts the buffers crossing a named element's src pad and divides by the
wall-clock run time -- the honest end-to-end rate. This deliberately ignores
any in-band FPS HUD (for example an overlay's exponential-moving-average
counter): an EMA of inter-frame gaps over-reports 2-3x when a queue drains in a
burst, so it is the wrong number to judge throughput by. To know how fast a
pipeline actually processes frames, count buffers at a pad.

    pipeline_bench.py --probe infer --iterations 3 \\
        --pipeline "filesrc location=clip.mp4 ! parsebin ! nvv4l2decoder
                    ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12'
                    ! nvmminfer name=infer engine-file=y.engine ! fakesink"

--pipeline is any gst-launch string. --probe is the name= of the element whose
src pad is counted. Exit 0 if every iteration reached EOS; 1 on pipeline error.
"""
import argparse
import json
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


def run_once(pipeline_str, probe_name):
    """Run the pipeline to EOS, return (frames, seconds, fps)."""
    pipeline = Gst.parse_launch(pipeline_str)
    elem = pipeline.get_by_name(probe_name)
    if elem is None:
        raise SystemExit(f"no element named '{probe_name}' in pipeline")
    pad = elem.get_static_pad("src")
    if pad is None:
        raise SystemExit(f"element '{probe_name}' has no static src pad")

    count = {"n": 0}

    def on_buffer(_pad, _info):
        count["n"] += 1
        return Gst.PadProbeReturn.OK

    pad.add_probe(Gst.PadProbeType.BUFFER, on_buffer)

    loop = GLib.MainLoop()
    state = {"error": None}
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_msg(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            state["error"] = f"{err.message} ({dbg})"
            loop.quit()

    bus.connect("message", on_msg)

    pipeline.set_state(Gst.State.PLAYING)
    t0 = time.monotonic()
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    elapsed = time.monotonic() - t0
    if state["error"]:
        raise RuntimeError(state["error"])
    fps = count["n"] / elapsed if elapsed > 0 else 0.0
    return count["n"], elapsed, fps


def main():
    ap = argparse.ArgumentParser(description="Frame-counted GStreamer throughput.")
    ap.add_argument("--pipeline", required=True, help="gst-launch pipeline string")
    ap.add_argument("--probe", required=True,
                    help="name= of the element whose src pad to count")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--json", help="write metrics JSON to this path")
    args = ap.parse_args()

    Gst.init(None)
    runs = []
    for i in range(args.iterations):
        frames, elapsed, fps = run_once(args.pipeline, args.probe)
        runs.append({"iter": i, "frames": frames,
                     "seconds": round(elapsed, 3), "fps": round(fps, 2)})
        print(f"iter {i}: {frames} frames  {elapsed:.2f}s  {fps:.2f} fps")

    mean_fps = sum(r["fps"] for r in runs) / len(runs)
    print(f"mean: {mean_fps:.2f} fps over {args.iterations} iteration(s)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"probe": args.probe, "iterations": args.iterations,
                       "mean_fps": round(mean_fps, 2), "runs": runs}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
