#!/usr/bin/env python3
"""Port nvmminfer infer-interval into the divergent deploy-checkout checkout.

That tree is NOT a git repo, so a .pre-interval.bak is taken before this runs.
Idempotent: exits 0 with a note if already patched.
"""
import sys

import os

# Deploy checkout root, e.g. DEPLOY_SRC=/path/to/deploy-checkout
P = os.path.join(os.environ["DEPLOY_SRC"], "gst/nvmminfer/gstnvmminfer.cpp")
s = open(P).read()

if "infer_interval" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    if old not in s:
        print("ANCHOR NOT FOUND:\n" + old); sys.exit(2)
    if s.count(old) != 1:
        print("ANCHOR NOT UNIQUE (%d):\n%s" % (s.count(old), old)); sys.exit(3)
    s = s.replace(old, new, 1)


# 1. struct fields
rep("""    guint64 frame_no;
""",
    """    guint64 frame_no;
    guint    infer_interval; /* run the network every Nth frame (1 = every frame) */
    guint64  seen_frames;    /* every frame reaching transform_ip; drives infer-interval.
                                Separate from frame_no, which counts only inferred
                                frames (it is the det-meta frame_number). */
""")

# 2. property enum
rep("PROP_MEASURE_LATENCY, PROP_LABELS, PROP_MERGE };",
    "PROP_MEASURE_LATENCY, PROP_LABELS, PROP_MERGE,\n       PROP_INFER_INTERVAL };")

# 3. the skip, first thing in transform_ip
rep("""    auto *self = GST_NVMM_INFER(bt);

    NvBufSurface *surf = surface_of(buf);""",
    """    auto *self = GST_NVMM_INFER(bt);

    /* infer-interval: run the network on every Nth frame only. A skipped frame passes
       through with NO detection meta -- downstream already handles a frame the detector
       found nothing in (nvmmfusekf gates on has_yolo), so no new code path is needed.
       Deliberately never re-attaches the previous frame's dets: stale meta would assert
       a box for a frame that was never inferred. In merge mode a skipped frame simply
       contributes nothing, leaving any upstream detector's dets untouched. */
    if (self->infer_interval > 1 && (self->seen_frames++ % self->infer_interval) != 0)
        return GST_FLOW_OK;

    NvBufSurface *surf = surface_of(buf);""")

# 4. set/get property
rep("        case PROP_MERGE:            self->merge = g_value_get_boolean(v); break;",
    "        case PROP_MERGE:            self->merge = g_value_get_boolean(v); break;\n"
    "        case PROP_INFER_INTERVAL:   self->infer_interval = g_value_get_uint(v); break;")
rep("        case PROP_MERGE:            g_value_set_boolean(v, self->merge); break;",
    "        case PROP_MERGE:            g_value_set_boolean(v, self->merge); break;\n"
    "        case PROP_INFER_INTERVAL:   g_value_set_uint(v, self->infer_interval); break;")

# 5. init defaults
rep("    self->frame_no = 0;",
    "    self->frame_no = 0;\n"
    "    self->infer_interval = 1;\n"
    "    self->seen_frames = 0;")

# 6. property registration, appended after the merge block
rep("""            "with cross-class NMS (instead of writing a separate meta). Off by default.",
            FALSE, flags));
""",
    """            "with cross-class NMS (instead of writing a separate meta). Off by default.",
            FALSE, flags));
    g_object_class_install_property(go, PROP_INFER_INTERVAL,
        g_param_spec_uint("infer-interval", "Inference interval",
            "Run the detector on every Nth frame (1 = every frame). Skipped frames pass "
            "through with no detection meta. Trades detection freshness for throughput: "
            "measured on Orin NX, a 1088x1920 YOLO is ~50%% of the frame budget when the "
            "tracker coasts (nvmmsamurai max-kf>0), so N=3 buys ~+50%% fps at ~8%% GT "
            "success -- and that cost is mostly slower ACQUISITION (seed latency 2.65x), "
            "not worse steady-state tracking",
            1, 1000, 1, flags));
""")

open(P, "w").write(s)
print("patched: struct, enum, transform_ip skip, set/get, init, property registration")
