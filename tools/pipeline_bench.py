#!/usr/bin/env python3
"""Steady-state fps from buffers counted at one element's src pad, timed first buffer to last.
That excludes preroll and engine/CUDA warmup, and ignores in-band FPS HUDs, whose EMA of
inter-frame gaps over-reports 2-3x when a queue drains in a burst."""
import argparse
import json
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


def run_once(pipeline_str, probe_name, timeout_s=0.0):
    """Return (frames, seconds, fps), fps = (frames - 1) / seconds: n buffers bound n - 1 intervals.

    timeout_s > 0 raises after that many wall-clock seconds, so a stalled pipeline cannot hang a sweep."""
    pipeline = Gst.parse_launch(pipeline_str)
    elem = pipeline.get_by_name(probe_name)
    if elem is None:
        raise RuntimeError(f"no element named '{probe_name}' in pipeline")
    pad = elem.get_static_pad("src")
    if pad is None:
        raise RuntimeError(f"element '{probe_name}' has no static src pad")

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
            return GLib.SOURCE_REMOVE
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
    error = None
    for i in range(args.iterations):
        try:
            frames, elapsed, fps = run_once(args.pipeline, args.probe, args.timeout)
        except RuntimeError as exc:
            error = f"iter {i}: {exc}"
            print(f"error: {error}", file=sys.stderr)
            break
        runs.append({"iter": i, "frames": frames,
                     "seconds": round(elapsed, 3), "fps": round(fps, 2)})
        print(f"iter {i}: {frames} frames  {elapsed:.2f}s  {fps:.2f} fps")

    mean_fps = round(sum(r["fps"] for r in runs) / len(runs), 2) if runs else 0.0
    if runs:
        print(f"mean: {mean_fps:.2f} fps over {len(runs)} iteration(s)")
    if args.json:
        result = {"probe": args.probe, "iterations": args.iterations,
                  "completed": len(runs), "mean_fps": mean_fps, "runs": runs}
        if error:
            result["error"] = error
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
