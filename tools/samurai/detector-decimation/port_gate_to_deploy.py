#!/usr/bin/env python3
"""Port nvmminfer infer-gate-frames (acquisition gating) into deploy-checkout.

Companion to port_interval_to_deploy.py; that must run first. The deploy tree is NOT
a git repo -- a .pre-gate.bak is taken before writing. Idempotent.
"""
import shutil
import sys

import os

# Deploy checkout root, e.g. DEPLOY_SRC=/path/to/deploy-checkout
P = os.path.join(os.environ["DEPLOY_SRC"], "gst/nvmminfer/gstnvmminfer.cpp")
s = open(P).read()

if "infer_gate_frames" in s:
    print("already patched"); sys.exit(0)
if "infer_interval" not in s:
    print("run port_interval_to_deploy.py first"); sys.exit(2)

shutil.copyfile(P, P + ".pre-gate.bak")


def rep(old, new):
    global s
    if old not in s:
        print("ANCHOR NOT FOUND:\n" + old); sys.exit(3)
    if s.count(old) != 1:
        print("ANCHOR NOT UNIQUE (%d):\n%s" % (s.count(old), old)); sys.exit(4)
    s = s.replace(old, new, 1)


rep("    guint    infer_interval; /* run the network every Nth frame (1 = every frame) */",
    "    guint    infer_interval; /* run the network every Nth frame (1 = every frame) */\n"
    "    guint    infer_gate_frames; /* consecutive inferred frames WITH a detection\n"
    "                                  required before decimating; 0 = ungated */")

rep("       frames (it is the det-meta frame_number). */",
    "       frames (it is the det-meta frame_number). */\n"
    "    guint    acq_run;        /* consecutive inferred frames that produced >=1 det */")

rep("PROP_MEASURE_LATENCY, PROP_LABELS, PROP_MERGE,\n       PROP_INFER_INTERVAL };",
    "PROP_MEASURE_LATENCY, PROP_LABELS, PROP_MERGE,\n"
    "       PROP_INFER_INTERVAL, PROP_INFER_GATE_FRAMES };")

# Gate the skip. Anchor on the exact line the interval patch installed.
rep("    if (self->infer_interval > 1 && (self->seen_frames++ % self->infer_interval) != 0)\n"
    "        return GST_FLOW_OK;",
    "       infer-gate-frames holds the interval OFF until detections flow steadily; no\n"
    "       detections is exactly when the tracker is acquiring or recovering, which is\n"
    "       where ungated decimation did its damage (3 of 12 GT sequences never acquired\n"
    "       at N=3). Gates on ANY detection -- nvmminfer does not know target_class. */\n"
    "    const gboolean gate_open = self->infer_gate_frames == 0 ||\n"
    "                               self->acq_run >= self->infer_gate_frames;\n"
    "    if (self->infer_interval > 1 && gate_open &&\n"
    "        (self->seen_frames++ % self->infer_interval) != 0)\n"
    "        return GST_FLOW_OK;")
# The interval patch's comment ended with "*/" just above; reopen it so the added
# prose sits inside one block comment instead of becoming bare code.
rep("       contributes nothing, leaving any upstream detector's dets untouched. */\n"
    "       infer-gate-frames holds",
    "       contributes nothing, leaving any upstream detector's dets untouched.\n\n"
    "       infer-gate-frames holds")

rep("    gst_buffer_add_nvmm_det_meta(buf, &fm);",
    "    /* Acquisition-gate state: reset on the first empty inferred frame so losing the\n"
    "       target immediately restores every-frame inference for fast reacquisition. */\n"
    "    if (fm.num_objects > 0) {\n"
    "        if (self->acq_run < G_MAXUINT) self->acq_run++;\n"
    "    } else {\n"
    "        self->acq_run = 0;\n"
    "    }\n\n"
    "    gst_buffer_add_nvmm_det_meta(buf, &fm);")

rep("        case PROP_INFER_INTERVAL:   self->infer_interval = g_value_get_uint(v); break;",
    "        case PROP_INFER_INTERVAL:   self->infer_interval = g_value_get_uint(v); break;\n"
    "        case PROP_INFER_GATE_FRAMES: self->infer_gate_frames = g_value_get_uint(v); break;")
rep("        case PROP_INFER_INTERVAL:   g_value_set_uint(v, self->infer_interval); break;",
    "        case PROP_INFER_INTERVAL:   g_value_set_uint(v, self->infer_interval); break;\n"
    "        case PROP_INFER_GATE_FRAMES: g_value_set_uint(v, self->infer_gate_frames); break;")

rep("            1, 1000, 1, flags));",
    "            1, 1000, 1, flags));\n"
    "    g_object_class_install_property(go, PROP_INFER_GATE_FRAMES,\n"
    "        g_param_spec_uint(\"infer-gate-frames\", \"Acquisition gate (frames)\",\n"
    "            \"Hold infer-interval OFF until this many consecutive inferred frames have \"\n"
    "            \"produced a detection, re-arming the hold as soon as one produces none; \"\n"
    "            \"0 = decimate unconditionally. Gates on ANY detection, not target-class\",\n"
    "            0, 10000, 0, flags));")

rep("    self->infer_interval = 1;\n    self->seen_frames = 0;",
    "    self->infer_interval = 1;\n    self->infer_gate_frames = 0;\n"
    "    self->seen_frames = 0;\n    self->acq_run = 0;")

open(P, "w").write(s)
print("patched: struct, enum, gated skip, acq_run update, set/get, property, init")
