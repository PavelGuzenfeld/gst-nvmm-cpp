# Worked example: assemble and run a GStreamer graph programmatically (in code,
# not a gst-launch string) — the pattern for embedding these elements in an
# application. Core-GStreamer only, so it builds and runs everywhere (CI-covered
# as a meson test on a core-element pipeline). Not installed; it exists to stay
# compilable and runnable as documentation for docs/embedding.md.
programmatic_pipeline = executable('programmatic_pipeline',
  files('programmatic_pipeline.cpp'),
  dependencies : [gst_dep],
  install : false,
)
test('programmatic_pipeline', programmatic_pipeline,
  args : ['30'], protocol : 'exitcode', timeout : 30)
