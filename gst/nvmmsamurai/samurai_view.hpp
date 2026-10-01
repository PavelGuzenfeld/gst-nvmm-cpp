#pragma once

namespace nvmm {

struct SamuraiView {
    float x = 0.f, y = 0.f, width = 0.f, height = 0.f;
};

/// Port of samurai tracker.py get_view_around_bbox: a crop square centered on the box,
/// shifted to stay inside the frame. The center truncates toward zero, as get_center(as_int=True).
inline SamuraiView get_view_around_bbox(float bx, float by, float bw, float bh,
                                        int crop, int frame_w, int frame_h)
{
    float h = (float)crop, w = (float)crop;
    if (h > frame_h) h = (float)frame_h;
    if (w > frame_w) w = (float)frame_w;
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

}
