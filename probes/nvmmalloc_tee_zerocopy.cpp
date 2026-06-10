// E2E proof that share-capable nvmmalloc keeps tee fan-out + make_writable
// zero-copy. Run ON the Jetson (real NvBufSurface). Pipeline:
//
//   videotestsrc ! nvvidconv ! NVMM NV12 ! nvmmconvert ! tee
//        ├─ queue ! fakesink   (branch A: simulates per-branch meta attach)
//        └─ queue ! fakesink   (branch B: read-only consumer)
//
// nvmmconvert emits buffers from OUR allocator; tee refs the same buffer into
// both branches. Branch A's probe does gst_buffer_make_writable (what attaching
// a per-branch GstMeta triggers). With share-capable memory the surface pointer
// is unchanged (shallow copy, same NvBufSurface); with the old NO_SHARE flag it
// would deep-copy to a different surface (and, for opaque NVMM, garbage).
//
// The NvBufSurface* is read via the standard NVMM convention: mapped data ptr.
#include <gst/gst.h>

#include <cstdio>

static int frames = 0, zero_copy = 0, deep_copied = 0;

static void* surface_of(GstBuffer* buf) {
    GstMemory* m = gst_buffer_peek_memory(buf, 0);
    GstMapInfo mi;
    if (!m || !gst_memory_map(m, &mi, GST_MAP_READ)) return nullptr;
    void* p = mi.data;  // NVMM convention: mapped data == NvBufSurface*
    gst_memory_unmap(m, &mi);
    return p;
}

// Branch A: make the buffer writable (as a per-branch meta attach would) and
// check the surface survives unchanged — i.e. the copy was shallow (shared).
static GstPadProbeReturn branch_a(GstPad*, GstPadProbeInfo* info, gpointer) {
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (frames == 0) {
        GstMemory* m = gst_buffer_peek_memory(buf, 0);
        g_print("  (allocator: %s)\n",
                m && m->allocator ? m->allocator->mem_type : "?");
    }
    void* before = surface_of(buf);
    buf = gst_buffer_make_writable(buf);
    void* after = surface_of(buf);
    GST_PAD_PROBE_INFO_DATA(info) = buf;  // hand back the (possibly new) buffer

    frames++;
    const bool same = before && before == after;
    if (same) zero_copy++; else deep_copied++;
    g_print("  frame %2d  branchA surface before=%p after make_writable=%p  %s\n",
            frames, before, after, same ? "SAME -> zero-copy" : "DIFFERENT -> DEEP COPY");
    return GST_PAD_PROBE_OK;
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    GError* err = nullptr;
    GstElement* pipe = gst_parse_launch(
        "videotestsrc num-buffers=8 ! video/x-raw,width=640,height=480,format=I420 "
        "! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
        // Force a real transform (NV12->RGBA) so nvmmconvert produces output
        // from OUR pool, rather than passing the upstream buffer through.
        "! nvmmconvert ! video/x-raw(memory:NVMM),format=RGBA,width=640,height=480 "
        "! tee name=t "
        "t. ! queue ! fakesink name=a sync=false "
        "t. ! queue ! fakesink name=b sync=false", &err);
    if (!pipe) {
        g_printerr("pipeline build failed: %s\n", err ? err->message : "?");
        return 2;
    }

    GstElement* a = gst_bin_get_by_name(GST_BIN(pipe), "a");
    GstPad* apad = gst_element_get_static_pad(a, "sink");
    gst_pad_add_probe(apad, GST_PAD_PROBE_TYPE_BUFFER, branch_a, nullptr, nullptr);

    g_print("== tee zero-copy E2E (share-capable nvmmalloc) ==\n");
    gst_element_set_state(pipe, GST_STATE_PLAYING);

    GstBus* bus = gst_element_get_bus(pipe);
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus, 10 * GST_SECOND,
        (GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    int rc = 0;
    if (msg && GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* e = nullptr; gchar* dbg = nullptr;
        gst_message_parse_error(msg, &e, &dbg);
        g_printerr("PIPELINE ERROR: %s\n", e ? e->message : "?");
        rc = 1;
    }
    if (msg) gst_message_unref(msg);

    gst_element_set_state(pipe, GST_STATE_NULL);
    gst_object_unref(apad); gst_object_unref(a);
    gst_object_unref(bus); gst_object_unref(pipe);

    g_print("\n%d frames: %d zero-copy, %d deep-copied\n", frames, zero_copy, deep_copied);
    if (rc == 0 && frames > 0 && deep_copied == 0)
        g_print("E2E PASS: tee fan-out + make_writable stayed zero-copy "
                "(same NvBufSurface on every branch-A frame)\n");
    else { g_print("E2E FAIL\n"); rc = rc ? rc : 1; }
    gst_deinit();
    return rc;
}
