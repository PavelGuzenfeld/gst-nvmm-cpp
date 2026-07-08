/// vit_grid.hpp — patch-grid token count for a vision-transformer encoder.
///
/// A ViT-style encoder that takes a square SxS input at patch stride P produces
/// an (S/P) x (S/P) grid of tokens, and the buffers that carry those tokens (and
/// anything sized per token, e.g. a memory bank of N frames) scale with that
/// count. Deriving the count from the configured input size — instead of baking
/// in the number for one size — is what lets the same code drive a 512, 384, or
/// 256 crop.
///
/// Pure integer math, no model/framework/GStreamer dependency, so it is
/// header-only and unit-tested off-target (tests/test_vit_grid.cpp).
#pragma once

namespace nvmm {

/// Side length (tokens per row) of the grid for an `input`x`input` square at
/// patch `stride`. Integer division: `input` should be a multiple of `stride`.
inline int vit_grid_side(int input, int stride) {
  return (stride > 0) ? input / stride : 0;
}

/// Token count (grid area) for an `input`x`input` square at patch `stride`.
inline int vit_grid_tokens(int input, int stride) {
  const int side = vit_grid_side(input, stride);
  return side * side;
}

}  // namespace nvmm
