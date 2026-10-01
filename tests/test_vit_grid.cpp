#include "vit_grid.hpp"

#include <cassert>
#include <cstdio>

int main() {
  using nvmm::vit_grid_side;
  using nvmm::vit_grid_tokens;

  assert(vit_grid_side(512, 16) == 32);
  assert(vit_grid_tokens(512, 16) == 1024);
  assert(vit_grid_side(384, 16) == 24);
  assert(vit_grid_tokens(384, 16) == 576);
  assert(vit_grid_side(256, 16) == 16);
  assert(vit_grid_tokens(256, 16) == 256);

  assert(vit_grid_tokens(224, 14) == 256);
  assert(vit_grid_tokens(224, 16) == 196);

  assert(vit_grid_side(520, 16) == 32);
  assert(vit_grid_tokens(520, 16) == 1024);

  assert(vit_grid_side(512, 0) == 0);
  assert(vit_grid_tokens(512, 0) == 0);

  std::puts("test_vit_grid: OK");
  return 0;
}
