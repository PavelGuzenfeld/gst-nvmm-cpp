/// Unit tests for the ViT patch-grid token-count helper (gst/common/vit_grid.hpp).
/// Pure host math — no CUDA/GStreamer — so it runs in the x86 CI build.
#include "vit_grid.hpp"

#include <cassert>
#include <cstdio>

int main() {
  using nvmm::vit_grid_side;
  using nvmm::vit_grid_tokens;

  // SAM2.1 encoder at stride 16: the three crop sizes the tracker supports.
  assert(vit_grid_side(512, 16) == 32);
  assert(vit_grid_tokens(512, 16) == 1024);  // the historical hard-coded value
  assert(vit_grid_side(384, 16) == 24);
  assert(vit_grid_tokens(384, 16) == 576);
  assert(vit_grid_side(256, 16) == 16);
  assert(vit_grid_tokens(256, 16) == 256);

  // General patch strides.
  assert(vit_grid_tokens(224, 14) == 256);  // 16 x 16
  assert(vit_grid_tokens(224, 16) == 196);  // 14 x 14 (classic ViT-B/16)

  // Truncating division when input is not an exact multiple of stride.
  assert(vit_grid_side(520, 16) == 32);
  assert(vit_grid_tokens(520, 16) == 1024);

  // Degenerate inputs must not divide by zero.
  assert(vit_grid_side(512, 0) == 0);
  assert(vit_grid_tokens(512, 0) == 0);

  std::puts("test_vit_grid: OK");
  return 0;
}
