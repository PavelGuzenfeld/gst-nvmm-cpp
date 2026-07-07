# Embedding a pipeline (programmatic construction)

A `gst-launch-1.0` string is the fastest way to try a pipeline, but an
application that embeds these elements builds the graph in code instead — so it
can set properties from config, read metadata off a pad, react to bus messages,
and change state at runtime. The construction is plain core-GStreamer API and is
identical whether the elements are host-only (`nvmmtracker`) or Jetson
(`nvmminfer`); only the factory names and properties differ from the demo below.

The worked example is [`examples/programmatic_pipeline.cpp`](https://github.com/PavelGuzenfeld/gst-nvmm-cpp/blob/main/examples/programmatic_pipeline.cpp).
It is core-GStreamer only (`videotestsrc → capsfilter → queue → identity →
fakesink`), so it builds and runs everywhere and is exercised in CI as a `meson
test`. The six steps are the whole pattern:

1. **Make** each element by its registered factory name with
   `gst_element_factory_make` (the same names you would type in a launch
   string), and a `gst_pipeline_new` to hold them.
2. **Configure** with `g_object_set` — this is where `engine-file`,
   `conf-threshold`, `leaky`, and the rest go, by property name.
3. **Add** every element to the bin (`gst_bin_add_many`) and **link** them in
   order (`gst_element_link_many`); check the link return, it stops at the first
   pair that fails to negotiate caps.
4. **Read results** off a pad with `gst_pad_add_probe` — the example counts
   buffers; a real graph reads `GstNvmmDetMeta` / `GstNvmmTrackMeta` off the
   same kind of probe.
5. **Watch the bus** (`gst_bus_add_watch`) so EOS and ERROR end the run instead
   of blocking forever.
6. **Run** — `gst_element_set_state(PLAYING)`, drive a `GMainLoop`, then tear
   down to `NULL`.

## Turning it into an inference graph

Swap the core elements for the real ones — the surrounding code (steps 3–6) does
not change:

- `videotestsrc` → your source, plus `nvvidconv` into `video/x-raw(memory:NVMM),
  format=NV12` (the caps `nvmminfer` expects).
- `identity` → `nvmminfer` (set `engine-file`), optionally followed by
  `nvmmtracker` / `nvmmfusekf`.
- `fakesink` → your sink (encoder + `filesink`/`udpsink`, or an `appsink` you
  pull metadata from).
- Put a `queue` between the heavy stages for pipeline parallelism, and a
  `queue leaky=2 max-size-buffers=2` at a live source for freshest-frame
  behaviour — see [the tracker pipeline](tracker-pipeline.md#queues-throughput-and-freshest-frame).

The element must be on `GST_PLUGIN_PATH` for `gst_element_factory_make` to
resolve it, exactly as for `gst-launch-1.0`.
