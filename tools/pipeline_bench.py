#!/usr/bin/env python3
"""Measure the real frame throughput of a GStreamer pipeline.

Counts the buffers crossing a named element's src pad and times the window from
the first such buffer to the last -- the honest steady-state rate. Timing from
the first buffer (rather than from the set_state(PLAYING) call) excludes
pipeline preroll and one-time engine/CUDA warmup, which would otherwise be
folded into the wall-clock and depress the rate by a different amount for every
clip length. This also deliberately ignores any in-band FPS HUD (for example an
overlay's exponential-moving-average counter): an EMA of inter-frame gaps
over-reports 2-3x when a queue drains in a burst, so it is the wrong number to
judge throughput by. To know how fast a pipeline actually processes frames,
count buffers at a pad.

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


def run_once(pipeline_str, probe_name, timeout_s=0.0):
    """Run the pipeline to EOS, return (frames, seconds, fps).

    ``seconds`` is the window from the first buffer at the probe pad to the
    last, and ``fps`` is measured across it as ``(frames - 1) / seconds`` --
    the count of inter-frame intervals over their span. Warmup before the first
    buffer (preroll, engine deserialization, CUDA context) is excluded by
    construction, so short and long clips are directly comparable.

    ``timeout_s`` > 0 bounds a single run: if neither EOS nor an error arrives
    within that wall-clock budget the run is abandoned and raised as an error,
    so a stalled or non-prerolling pipeline cannot hang an unattended sweep.
    """
    pipeline = Gst.parse_launch(pipeline_str)
    elem = pipeline.get_by_name(probe_name)
    if elem is None:
        raise SystemExit(f"no element named '{probe_name}' in pipeline")
    pad = elem.get_static_pad("src")
    if pad is None:
        raise SystemExit(f"element '{probe_name}' has no static src pad")

    stats = {"n": 0, "first": None, "last": None}

    def on_buffer(_pad, _info):
        now = time.monotonic()
        if stats["first"] is None:
            stats["first"] = now
        stats["last"] = now
        stats["n"] += 1
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

    if timeout_s > 0:
        def on_timeout():
            state["error"] = (
                f"timed out after {timeout_s:g}s "
                f"(reached {stats['n']} frames, no EOS)"
            )
            loop.quit()
            return False  # one-shot
        GLib.timeout_add(int(timeout_s * 1000), on_timeout)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("pipeline failed to start (state change to PLAYING failed)")
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    if state["error"]:
        raise RuntimeError(state["error"])
    n = stats["n"]
    # First-to-last buffer span: n buffers bound n-1 inter-frame intervals.
    elapsed = (stats["last"] - stats["first"]) if n >= 2 else 0.0
    fps = (n - 1) / elapsed if elapsed > 0 else 0.0
    return n, elapsed, fps


def main():
    ap = argparse.ArgumentParser(description="Frame-counted GStreamer throughput.")
    ap.add_argument("--pipeline", required=True, help="gst-launch pipeline string")
    ap.add_argument("--probe", required=True,
                    help="name= of the element whose src pad to count")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per-iteration wall-clock budget in seconds; "
                         "0 (default) waits indefinitely for EOS")
    ap.add_argument("--json", help="write metrics JSON to this path")
    args = ap.parse_args()

    Gst.init(None)
    runs = []
    for i in range(args.iterations):
        frames, elapsed, fps = run_once(args.pipeline, args.probe, args.timeout)
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
