/// samurai_enc_probe — drive the real SamuraiTracker front-end (RoiCropper +
/// image_encoder) on a controlled NV12 frame so its crop + encoder outputs can
/// be parity-checked against the PyTorch reference fed the *same* NV12 bytes.
///
/// Usage:
///   samurai_enc_probe <nv12_file> <W> <H> <bx> <by> <bw> <bh> <engine_dir> <dump_dir>
///
/// Fills an NV12 NvBufSurface from the raw file (Y plane + interleaved UV plane,
/// tightly packed W x H), seeds the tracker at box (bx,by,bw,bh) in frame coords,
/// which runs get_view -> VIC crop -> SAM2 normalize -> image_encoder and (with
/// SAMURAI_DUMP_DIR set) dumps crop_input.bin + out1..out6.bin into dump_dir.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gst/gst.h>
#include <nvbufsurface.h>

#include "nvmm_buffer.hpp"
#include "nvmm_types.hpp"
#include "samurai_tracker.hpp"

// The tracker translation unit expects this category to exist (it is normally
// defined by the GStreamer element). Provide + init it for the standalone probe.
GST_DEBUG_CATEGORY(gst_nvmm_samurai_debug);

static bool fill_plane(nvmm::NvmmBuffer &buf, uint32_t plane,
                       const uint8_t *src, uint32_t rows, uint32_t row_bytes)
{
    auto m = buf.map_write(plane);
    if (!m.has_value()) { std::fprintf(stderr, "map_write(%u) failed\n", plane); return false; }
    const nvmm::PlaneInfo pi = buf.plane_info(plane);
    uint8_t *dst = m.value().data();
    for (uint32_t r = 0; r < rows; r++)
        std::memcpy(dst + (size_t)r * pi.pitch, src + (size_t)r * row_bytes, row_bytes);
    buf.unmap();
    return true;
}

int main(int argc, char **argv)
{
    gst_init(&argc, &argv);
    GST_DEBUG_CATEGORY_INIT(gst_nvmm_samurai_debug, "nvmmsamurai", 0, "probe");
    if (argc != 10) {
        std::fprintf(stderr, "usage: %s nv12 W H bx by bw bh engine_dir dump_dir\n", argv[0]);
        return 2;
    }
    const char *nv12 = argv[1];
    const uint32_t W = (uint32_t)std::atoi(argv[2]);
    const uint32_t H = (uint32_t)std::atoi(argv[3]);
    nvmm::TrackBox box;
    box.left = (float)std::atof(argv[4]); box.top = (float)std::atof(argv[5]);
    box.width = (float)std::atof(argv[6]); box.height = (float)std::atof(argv[7]);
    box.valid = true;
    const char *engine_dir = argv[8];
    const char *dump_dir = argv[9];

    // Read tightly-packed NV12: Y (H*W) then UV (H/2*W).
    const size_t y_bytes = (size_t)W * H;
    const size_t uv_bytes = (size_t)W * (H / 2);
    std::vector<uint8_t> raw(y_bytes + uv_bytes);
    FILE *f = std::fopen(nv12, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", nv12); return 2; }
    size_t got = std::fread(raw.data(), 1, raw.size(), f);
    std::fclose(f);
    if (got != raw.size()) {
        std::fprintf(stderr, "short read: %zu != %zu\n", got, raw.size());
        return 2;
    }

    nvmm::SurfaceParams sp;
    sp.width = W; sp.height = H; sp.color_format = nvmm::ColorFormat::kNV12;
    auto buf = nvmm::NvmmBuffer::create(sp);
    if (!buf.has_value()) { std::fprintf(stderr, "NvmmBuffer::create failed\n"); return 2; }
    if (!fill_plane(buf.value(), 0, raw.data(), H, W)) return 2;
    if (!fill_plane(buf.value(), 1, raw.data() + y_bytes, H / 2, W)) return 2;

    setenv("SAMURAI_DUMP_DIR", dump_dir, 1);
    nvmm::SamuraiConfig cfg;
    cfg.engine_dir = engine_dir;
    cfg.consts_file = std::string(engine_dir) + "/samurai_consts.bin";
    cfg.crop_size = 512;
    nvmm::SamuraiTracker tracker;
    std::string err;
    if (!tracker.init(cfg, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    if (!tracker.seed(buf.value().raw(), box, err)) {
        std::fprintf(stderr, "seed: %s\n", err.c_str());
        return 1;
    }
    std::printf("PROBE_OK dumped to %s\n", dump_dir);
    return 0;
}
