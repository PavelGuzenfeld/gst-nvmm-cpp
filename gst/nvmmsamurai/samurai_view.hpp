/// samurai_view.hpp — dependency-free crop geometry, header-only so it can be
/// unit-tested off-target (no CUDA/GStreamer). Exact port of
/// samurai/tracker.py:get_view_around_bbox (the crop view == a fixed square
/// centered on the box, shifted to stay inside the frame).
#pragma once

namespace nvmm {

/// A fixed-size crop view (frame pixel coords); width/height == crop_size.
struct SamuraiView {
    float x = 0.f, y = 0.f, width = 0.f, height = 0.f;
};

/// Fixed-size `crop`-square view centered on box (bx,by,bw,bh top-left+size),
/// shifted to remain inside a frame_w x frame_h frame.
inline SamuraiView get_view_around_bbox(float bx, float by, float bw, float bh,
                                        int crop, int frame_w, int frame_h)
{
    float h = (float)crop, w = (float)crop;
    if (h > frame_h) h = (float)frame_h;
    if (w > frame_w) w = (float)frame_w;
    // BoundingBox.get_center(as_int=True) truncates toward zero.
    const int cx = (int)(bx + bw / 2.f);
    const int cy = (int)(by + bh / 2.f);
    float xs = cx - w / 2.f, xe = cx + w / 2.f;
    float ys = cy - h / 2.f, ye = cy + h / 2.f;
    float dx = 0.f, dy = 0.f;
    if (xs < 0.f)      dx = -xs;
    if (xe >= frame_w) dx = frame_w - xe;
    if (ys < 0.f)      dy = -ys;
    if (ye >= frame_h) dy = frame_h - ye;
    return SamuraiView{xs + dx, ys + dy, w, h};
}

}  // namespace nvmm
