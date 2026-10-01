#pragma once

namespace nvmm {

inline int vit_grid_side(int input, int stride) {
  return (stride > 0) ? input / stride : 0;
}

inline int vit_grid_tokens(int input, int stride) {
  const int side = vit_grid_side(input, stride);
  return side * side;
}

}
