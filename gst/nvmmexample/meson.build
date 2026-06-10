# Worked example element for docs/extending.md. Pure host (no CUDA/VIC), so it
# builds and is CI-covered everywhere, but it is NOT installed — it exists to
# stay compilable as documentation.
gstnvmmdetlog = library('gstnvmmdetlog',
  files('gstnvmmdetlog.cpp'),
  include_directories : [config_inc, common_inc],
  dependencies : [nvmm_common_dep, gst_base_dep],
  install : false,
)
