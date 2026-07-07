/// programmatic_pipeline — build and run a GStreamer graph in code, not from a
/// gst-launch string.
///
/// A gst-launch string is convenient for experiments, but an application that
/// embeds these elements assembles the graph itself: make each element by its
/// registered factory name, set properties with g_object_set, add them to a
/// GstBin, link with gst_element_link_many, then drive the bus. The elements
/// behave identically however they are instantiated, so this is exactly the
/// construction you would use to embed nvmminfer / nvmmtracker in your own
/// program on a Jetson — only the factory names and properties change.
///
/// The demo graph is core-GStreamer only
/// (videotestsrc -> capsfilter -> queue -> identity -> fakesink) so it builds
/// and runs anywhere, including CI, with no CUDA/TensorRT/NVMM. A buffer probe
/// on the identity src pad counts frames; the bus watch turns EOS/ERROR into a
/// clean exit. To make it a real inference graph, swap videotestsrc for your
/// source (+ nvvidconv into NVMM NV12), identity for nvmminfer, and fakesink for
/// your sink — the surrounding code is unchanged.
///
/// Usage: programmatic_pipeline [num-buffers]   (default 300)

#include <gst/gst.h>

#include <cstdlib>

namespace {
guint frame_count = 0;

// Runs on the streaming thread for every buffer crossing the probed pad.
GstPadProbeReturn count_buffers(GstPad *, GstPadProbeInfo *, gpointer) {
  ++frame_count;
  return GST_PAD_PROBE_OK;
}

// EOS ends the run; ERROR reports and ends it. Anything else is ignored.
gboolean on_bus(GstBus *, GstMessage *msg, gpointer user) {
  auto *loop = static_cast<GMainLoop *>(user);
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *err = nullptr;
      gchar *dbg = nullptr;
      gst_message_parse_error(msg, &err, &dbg);
      g_printerr("error: %s (%s)\n", err->message, dbg ? dbg : "");
      g_clear_error(&err);
      g_free(dbg);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}
}  // namespace

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  int num_buffers = (argc > 1) ? std::atoi(argv[1]) : 300;
  if (num_buffers <= 0) num_buffers = 300;

  // 1. Make each element by its registered factory name. A real graph resolves
  //    "nvmminfer" / "nvmmtracker" the same way, once the plugin is on
  //    GST_PLUGIN_PATH.
  GstElement *pipeline = gst_pipeline_new("demo");
  GstElement *src = gst_element_factory_make("videotestsrc", "src");
  GstElement *caps = gst_element_factory_make("capsfilter", "caps");
  GstElement *queue = gst_element_factory_make("queue", "q");
  GstElement *probe = gst_element_factory_make("identity", "probe");
  GstElement *sink = gst_element_factory_make("fakesink", "sink");
  if (!pipeline || !src || !caps || !queue || !probe || !sink) {
    g_printerr("failed to create an element (is the plugin installed?)\n");
    return 1;
  }

  // 2. Configure with g_object_set. On a real graph this is where engine-file,
  //    conf-threshold, leaky, etc. go — set by property name, exactly as a
  //    launch string spells them.
  g_object_set(src, "num-buffers", num_buffers, nullptr);
  GstCaps *nv12 =
      gst_caps_from_string("video/x-raw,format=NV12,width=1280,height=720");
  g_object_set(caps, "caps", nv12, nullptr);
  gst_caps_unref(nv12);
  g_object_set(sink, "sync", FALSE, nullptr);

  // 3. Add every element to the bin, then link in order. link_many stops at the
  //    first pair that fails to negotiate, so check its return.
  gst_bin_add_many(GST_BIN(pipeline), src, caps, queue, probe, sink, nullptr);
  if (!gst_element_link_many(src, caps, queue, probe, sink, nullptr)) {
    g_printerr("failed to link the pipeline\n");
    gst_object_unref(pipeline);
    return 1;
  }

  // 4. Read results off a pad — here, count frames at the probe element's src.
  //    A real graph reads detection/track metadata off the same kind of probe.
  GstPad *srcpad = gst_element_get_static_pad(probe, "src");
  gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_BUFFER, count_buffers, nullptr,
                    nullptr);
  gst_object_unref(srcpad);

  // 5. Watch the bus so EOS/ERROR ends the run instead of blocking forever.
  GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
  GstBus *bus = gst_element_get_bus(pipeline);
  gst_bus_add_watch(bus, on_bus, loop);
  gst_object_unref(bus);

  // 6. Run to EOS, then tear down.
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);
  gst_element_set_state(pipeline, GST_STATE_NULL);

  g_print("processed %u frames\n", frame_count);

  gst_object_unref(pipeline);
  g_main_loop_unref(loop);

  // Clean run == every produced frame reached the probe.
  return (frame_count == static_cast<guint>(num_buffers)) ? 0 : 1;
}
