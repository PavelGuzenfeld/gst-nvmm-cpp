#include <gst/gst.h>
#include <gst/check/gstharness.h>
#include <npp.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "nvmm_class_meta.h"
#include "nvmm_det_meta.h"
#include "nvmm_frame.h"
#include "trt_test_engine.h"
#include "test_harness.h"

namespace {

struct GstInit { GstInit() { gst_init(nullptr, nullptr); } } _gst_init;

using nvinfer1::Dims2;
using nvinfer1::Dims3;
using nvinfer1::Dims4;

/// [1,5,8] YOLO head: proposal 0 is a fixed 8x4 box at net (16,16) whose class-0
/// score is the max of input channel 0; the other seven score 0.
std::unique_ptr<EngineFile> build_detector()
{
    NetBuilder b;
    nvinfer1::ITensor *in = b.input(Dims4{1, 3, 32, 32});
    std::vector<float> coords(4 * 8, 0.f);
    coords[0 * 8] = 16; coords[1 * 8] = 16; coords[2 * 8] = 8; coords[3 * 8] = 4;
    nvinfer1::ITensor *ch0 = b.net().addSlice(*in, Dims4{0, 0, 0, 0}, Dims4{1, 1, 32, 32},
                                              Dims4{1, 1, 1, 1})->getOutput(0);
    nvinfer1::ITensor *score = b.binary(
        b.reshape(b.reduce_max(ch0, 0xEu, true), Dims3{1, 1, 1}),
        b.constant(Dims3{1, 1, 8}, {1, 0, 0, 0, 0, 0, 0, 0}),
        nvinfer1::ElementWiseOperation::kPROD);
    nvinfer1::ITensor *parts[] = {b.constant(Dims3{1, 4, 8}, coords), score};
    nvinfer1::IConcatenationLayer *cat = b.net().addConcatenation(parts, 2);
    cat->setAxis(1);
    b.output(cat->getOutput(0), "output0");
    return b.save();
}

/// [1,3] head: per-channel max over the ROI.
std::unique_ptr<EngineFile> build_classifier()
{
    NetBuilder b;
    nvinfer1::ITensor *in = b.input(Dims4{1, 3, 16, 16});
    b.output(b.reduce_max(in, 0xCu, false), "scores");
    return b.save();
}

const EngineFile *detector_engine()
{
    static const auto e = build_detector();
    return e.get();
}

const EngineFile *classifier_engine()
{
    static const auto e = build_classifier();
    return e.get();
}

std::unique_ptr<EngineFile> engine_with(nvinfer1::Dims in_dims,
                                        std::vector<nvinfer1::Dims> outs)
{
    NetBuilder b;
    nvinfer1::ITensor *in = b.input(in_dims);
    int i = 0;
    for (const auto &d : outs) b.output(b.constant_fed_by(in, d), i++ ? "extra" : "output0");
    return b.save();
}

GstElement *make(const char *factory)
{
    GstElement *e = gst_element_factory_make(factory, nullptr);
    if (!e) throw std::runtime_error(std::string(factory) + " not found");
    return e;
}

/// Resets the NPP stream before teardown: Preprocessor leaves the process-wide NPP
/// stream on its own, which stop() destroys, and the next nvmminfer to configure
/// crashes in nppSetStream (#117). Delete the reset with that fix.
std::vector<GstBuffer *> run(GstElement *e, int w, int h, std::vector<GstBuffer *> in)
{
    GstHarness *hn = gst_harness_new_with_element(e, "sink", "src");
    const std::string caps = "video/x-raw(memory:NVMM),format=NV12,width=" +
        std::to_string(w) + ",height=" + std::to_string(h) + ",framerate=30/1";
    gst_harness_set_src_caps_str(hn, caps.c_str());
    std::vector<GstBuffer *> out;
    for (GstBuffer *b : in) {
        if (gst_harness_push(hn, b) != GST_FLOW_OK) throw std::runtime_error("push failed");
        out.push_back(gst_harness_pull(hn));
    }
    nppSetStream(nullptr);
    gst_harness_teardown(hn);
    gst_object_unref(e);
    return out;
}

void unref_all(std::vector<GstBuffer *> &v) { for (GstBuffer *b : v) gst_buffer_unref(b); }

GstNvmmDetMeta *dets_of(GstBuffer *b) { return gst_buffer_get_nvmm_det_meta(b); }

float scaled(uint8_t v, double scale) { return (float)v * (float)scale; }

TEST(red_frame_yields_one_person_box_mapped_back_through_the_letterbox) {
    auto eng = detector_engine();
    GstElement *e = make("nvmminfer");
    g_object_set(e, "engine-file", eng->path.c_str(), "conf-threshold", 0.5, NULL);
    auto out = run(e, 64, 32, {nvmm_nv12_buffer(64, 32, solid(kRed))});
    const Rgba red = vic_rgba_at(solid(kRed), 64, 32, 0, 0);
    GstNvmmDetMeta *m = dets_of(out[0]);
    ASSERT_NOT_NULL(m);
    ASSERT_EQ(m->frame_number, 0u);
    ASSERT_EQ(m->infer_width, 64u);
    ASSERT_EQ(m->infer_height, 32u);
    ASSERT_EQ(m->flags, 0u);
    ASSERT_EQ(m->num_objects, 1u);
    const NvmmDetObject &o = m->objects[0];
    ASSERT_EQ(o.left, 24.f);
    ASSERT_EQ(o.top, 12.f);
    ASSERT_EQ(o.width, 16.f);
    ASSERT_EQ(o.height, 8.f);
    ASSERT_EQ(o.class_id, 0);
    ASSERT_EQ(o.confidence, scaled(red[0], 1.0 / 255.0));
    ASSERT_EQ(std::string(o.label), std::string("person"));
    unref_all(out);
}

TEST(net_scale_factor_scales_the_pixels_the_engine_sees) {
    auto eng = detector_engine();
    GstElement *e = make("nvmminfer");
    g_object_set(e, "engine-file", eng->path.c_str(), "net-scale-factor", 1.0 / 510.0, NULL);
    auto out = run(e, 64, 32, {nvmm_nv12_buffer(64, 32, solid(kRed))});
    const Rgba red = vic_rgba_at(solid(kRed), 64, 32, 0, 0);
    ASSERT_EQ(dets_of(out[0])->num_objects, 1u);
    ASSERT_EQ(dets_of(out[0])->objects[0].confidence, scaled(red[0], 1.0 / 510.0));
    unref_all(out);
}

/// The 114 letterbox pad is 0.447 after scaling, under the 0.5 threshold.
TEST(bgr_colour_order_feeds_blue_to_channel_zero) {
    auto eng = detector_engine();
    GstElement *e = make("nvmminfer");
    gst_util_set_object_arg(G_OBJECT(e), "color-order", "bgr");
    g_object_set(e, "engine-file", eng->path.c_str(), "conf-threshold", 0.5, NULL);
    auto out = run(e, 64, 32, {nvmm_nv12_buffer(64, 32, solid(kRed)),
                               nvmm_nv12_buffer(64, 32, solid(kBlue))});
    const Rgba blue = vic_rgba_at(solid(kBlue), 64, 32, 0, 0);
    ASSERT_EQ(dets_of(out[0])->num_objects, 0u);
    ASSERT_EQ(dets_of(out[1])->num_objects, 1u);
    ASSERT_EQ(dets_of(out[1])->objects[0].confidence, scaled(blue[2], 1.0 / 255.0));
    unref_all(out);
}

TEST(infer_interval_two_attaches_det_meta_to_every_other_frame_numbered_by_inference) {
    auto eng = detector_engine();
    GstElement *e = make("nvmminfer");
    g_object_set(e, "engine-file", eng->path.c_str(), "infer-interval", 2u, NULL);
    std::vector<GstBuffer *> in;
    for (int i = 0; i < 3; i++) in.push_back(nvmm_nv12_buffer(64, 32, solid(kRed)));
    auto out = run(e, 64, 32, in);
    ASSERT_NOT_NULL(dets_of(out[0]));
    ASSERT_TRUE(dets_of(out[1]) == nullptr);
    ASSERT_NOT_NULL(dets_of(out[2]));
    ASSERT_EQ(dets_of(out[2])->frame_number, 1u);
    unref_all(out);
}

TEST(infer_gate_frames_holds_decimation_off_until_that_many_frames_detected) {
    auto eng = detector_engine();
    GstElement *e = make("nvmminfer");
    g_object_set(e, "engine-file", eng->path.c_str(), "infer-interval", 2u,
                 "infer-gate-frames", 2u, NULL);
    std::vector<GstBuffer *> in;
    for (int i = 0; i < 4; i++) in.push_back(nvmm_nv12_buffer(64, 32, solid(kRed)));
    auto out = run(e, 64, 32, in);
    ASSERT_NOT_NULL(dets_of(out[0]));
    ASSERT_NOT_NULL(dets_of(out[1]));
    ASSERT_NOT_NULL(dets_of(out[2]));
    ASSERT_TRUE(dets_of(out[3]) == nullptr);
    unref_all(out);
}

std::string infer_start_error(const char *factory, const char *engine_path)
{
    GstElement *e = make(factory);
    if (engine_path) g_object_set(e, "engine-file", engine_path, NULL);
    const std::string err = start_failure(e);
    gst_object_unref(e);
    return err;
}

TEST(nvmminfer_without_engine_file_refuses_to_start) {
    ASSERT_EQ(infer_start_error("nvmminfer", nullptr),
              std::string("the \"engine-file\" property is required"));
}

TEST(nvmminfer_rejects_a_file_that_is_not_a_serialized_engine) {
    gchar *path = nullptr;
    const gint fd = g_file_open_tmp("nvmm-test-XXXXXX.engine", &path, nullptr);
    ASSERT_TRUE(fd >= 0);
    ASSERT_EQ(write(fd, "not an engine", 13), 13);
    close(fd);
    EngineFile garbage(path);
    g_free(path);
    ASSERT_EQ(infer_start_error("nvmminfer", garbage.path.c_str()),
              "failed to load TensorRT engine \"" + garbage.path + "\"");
}

TEST(nvmminfer_rejects_an_engine_with_two_outputs) {
    auto eng = engine_with(Dims4{1, 3, 32, 32}, {Dims3{1, 5, 8}, Dims3{1, 5, 8}});
    ASSERT_EQ(infer_start_error("nvmminfer", eng->path.c_str()),
              std::string("engine has 1 input(s)/2 output(s), expected exactly 1/1"));
}

TEST(nvmminfer_rejects_an_input_that_is_not_three_channel_nchw) {
    auto eng = engine_with(Dims4{1, 1, 32, 32}, {Dims3{1, 5, 8}});
    ASSERT_EQ(infer_start_error("nvmminfer", eng->path.c_str()),
              std::string("input \"images\" is 1x1x32x32, expected 1x3xHxW (NCHW)"));
}

TEST(nvmminfer_rejects_a_transposed_head_with_more_channels_than_proposals) {
    auto eng = engine_with(Dims4{1, 3, 32, 32}, {Dims3{1, 8, 5}});
    ASSERT_EQ(infer_start_error("nvmminfer", eng->path.c_str()),
              std::string("output \"output0\" is 1x8x5, expected channels-first "
                          "[1, 4+classes, proposals]"));
}

TEST(nvmminfer_rejects_a_head_with_box_channels_but_no_class_channel) {
    auto eng = engine_with(Dims4{1, 3, 32, 32}, {Dims3{1, 4, 8}});
    ASSERT_EQ(infer_start_error("nvmminfer", eng->path.c_str()),
              std::string("output \"output0\" is 1x4x8, expected channels-first "
                          "[1, 4+classes, proposals]"));
}

TEST(nvmminfer_rejects_a_square_head_with_as_many_channels_as_proposals) {
    auto eng = engine_with(Dims4{1, 3, 32, 32}, {Dims3{1, 8, 8}});
    ASSERT_EQ(infer_start_error("nvmminfer", eng->path.c_str()),
              std::string("output \"output0\" is 1x8x8, expected channels-first "
                          "[1, 4+classes, proposals]"));
}

Nv12Painter red_left_blue_right()
{
    return [](int x, int) { return x < 32 ? kRed : kBlue; };
}

NvmmDetObject roi(float l, float w, uint64_t tracker)
{
    NvmmDetObject o{};
    o.left = l; o.top = 8; o.width = w; o.height = 16;
    o.tracker_id = tracker;
    return o;
}

GstBuffer *split_frame_with(std::initializer_list<NvmmDetObject> objs)
{
    GstBuffer *buf = nvmm_nv12_buffer(64, 32, red_left_blue_right());
    static NvmmFrameMeta fm;
    std::memset(&fm, 0, sizeof fm);
    fm.infer_width = 64;
    fm.infer_height = 32;
    for (const auto &o : objs) fm.objects[fm.num_objects++] = o;
    gst_buffer_add_nvmm_det_meta(buf, &fm);
    return buf;
}

struct LabelsFile {
    std::string path;
    explicit LabelsFile(const char *text)
    {
        gchar *p = nullptr;
        const gint fd = g_file_open_tmp("nvmm-test-XXXXXX.txt", &p, nullptr);
        close(fd);
        g_file_set_contents(p, text, -1, nullptr);
        path = p;
        g_free(p);
    }
    ~LabelsFile() { g_unlink(path.c_str()); }
};

GstElement *classifier(const EngineFile &eng, const LabelsFile &labels, const char *act)
{
    GstElement *e = make("nvmmsecondaryinfer");
    g_object_set(e, "engine-file", eng.path.c_str(), "labels-file", labels.path.c_str(), NULL);
    gst_util_set_object_arg(G_OBJECT(e), "output-activation", act);
    return e;
}

TEST(each_roi_is_classified_from_its_own_pixels_with_labels_from_the_file) {
    auto eng = classifier_engine();
    LabelsFile labels("red\ngreen\n\nblue\n");
    auto out = run(classifier(*eng, labels, "none"), 64, 32,
                   {split_frame_with({roi(8, 16, 0), roi(40, 16, 0)})});
    const Rgba red = vic_rgba_at(red_left_blue_right(), 64, 32, 0, 0);
    const Rgba blue = vic_rgba_at(red_left_blue_right(), 64, 32, 63, 0);
    GstNvmmClassMeta *cm = gst_buffer_get_nvmm_class_meta(out[0]);
    ASSERT_NOT_NULL(cm);
    ASSERT_EQ(cm->num_objects, 2u);
    ASSERT_EQ(cm->objects[0].class_id, 0);
    ASSERT_EQ(std::string(cm->objects[0].label), std::string("red"));
    ASSERT_EQ(cm->objects[0].confidence, scaled(red[0], 1.0 / 255.0));
    ASSERT_EQ(cm->objects[0].fresh, 1u);
    ASSERT_EQ(cm->objects[1].class_id, 2);
    ASSERT_EQ(std::string(cm->objects[1].label), std::string("blue"));
    ASSERT_EQ(cm->objects[1].confidence, scaled(blue[2], 1.0 / 255.0));
    unref_all(out);
}

TEST(softmax_activation_reports_the_top1_probability_over_all_logits) {
    auto eng = classifier_engine();
    LabelsFile labels("red\ngreen\nblue\n");
    auto out = run(classifier(*eng, labels, "softmax"), 64, 32,
                   {split_frame_with({roi(8, 16, 0)})});
    const Rgba red = vic_rgba_at(red_left_blue_right(), 64, 32, 0, 0);
    double sum = 0.0;
    for (int c = 0; c < 3; c++)
        sum += std::exp((double)scaled(red[c], 1.0 / 255.0) - (double)scaled(red[0], 1.0 / 255.0));
    GstNvmmClassMeta *cm = gst_buffer_get_nvmm_class_meta(out[0]);
    ASSERT_EQ(cm->objects[0].class_id, 0);
    ASSERT_EQ(cm->objects[0].confidence, (float)(1.0 / sum));
    unref_all(out);
}

TEST(bgr_classifier_sees_red_in_its_last_channel) {
    auto eng = classifier_engine();
    LabelsFile labels("a\nb\nc\n");
    GstElement *e = classifier(*eng, labels, "none");
    gst_util_set_object_arg(G_OBJECT(e), "color-order", "bgr");
    auto out = run(e, 64, 32, {split_frame_with({roi(8, 16, 0)})});
    ASSERT_EQ(gst_buffer_get_nvmm_class_meta(out[0])->objects[0].class_id, 2);
    unref_all(out);
}

TEST(class_id_past_the_labels_file_falls_back_to_class_n) {
    auto eng = classifier_engine();
    LabelsFile labels("red\n");
    auto out = run(classifier(*eng, labels, "none"), 64, 32,
                   {split_frame_with({roi(40, 16, 0)})});
    ASSERT_EQ(std::string(gst_buffer_get_nvmm_class_meta(out[0])->objects[0].label),
              std::string("class2"));
    unref_all(out);
}

TEST(tracked_object_is_served_from_cache_between_infer_intervals) {
    auto eng = classifier_engine();
    LabelsFile labels("red\ngreen\nblue\n");
    auto out = run(classifier(*eng, labels, "none"), 64, 32,
                   {split_frame_with({roi(8, 16, 7)}), split_frame_with({roi(8, 16, 7)})});
    const NvmmClassEntry &second = gst_buffer_get_nvmm_class_meta(out[1])->objects[0];
    ASSERT_EQ(gst_buffer_get_nvmm_class_meta(out[0])->objects[0].fresh, 1u);
    ASSERT_EQ(second.fresh, 0u);
    ASSERT_EQ(second.class_id, 0);
    ASSERT_EQ(std::string(second.label), std::string("red"));
    unref_all(out);
}

TEST(roi_narrower_than_min_roi_size_is_left_unclassified) {
    auto eng = classifier_engine();
    LabelsFile labels("red\n");
    auto out = run(classifier(*eng, labels, "none"), 64, 32,
                   {split_frame_with({roi(8, 14, 0)})});
    const NvmmClassEntry &e = gst_buffer_get_nvmm_class_meta(out[0])->objects[0];
    ASSERT_EQ(e.class_id, -1);
    ASSERT_EQ(e.label[0], '\0');
    unref_all(out);
}

TEST(nvmmsecondaryinfer_rejects_a_detector_head) {
    auto eng = detector_engine();
    ASSERT_EQ(infer_start_error("nvmmsecondaryinfer", eng->path.c_str()),
              std::string("output \"output0\" is 1x5x8, expected a per-class score vector "
                          "([1,C], [1,C,1,1] or [C])"));
}

TEST(nvmmsecondaryinfer_rejects_offsets_that_are_not_three_numbers) {
    auto eng = classifier_engine();
    GstElement *e = make("nvmmsecondaryinfer");
    g_object_set(e, "engine-file", eng->path.c_str(), "offsets", "1,2", NULL);
    ASSERT_EQ(start_failure(e), std::string("offsets \"1,2\" is not 3 comma-separated numbers"));
    gst_object_unref(e);
}

TEST(nvmmsecondaryinfer_rejects_a_missing_labels_file) {
    auto eng = classifier_engine();
    GstElement *e = make("nvmmsecondaryinfer");
    g_object_set(e, "engine-file", eng->path.c_str(), "labels-file", "/nonexistent/labels", NULL);
    ASSERT_EQ(start_failure(e), std::string("failed to read labels-file \"/nonexistent/labels\""));
    gst_object_unref(e);
}

TEST(detgate_without_engine_dir_passes_every_detection_through) {
    auto out = run(make("nvmmdetgate"), 64, 32,
                   {split_frame_with({roi(8, 16, 0), roi(40, 16, 0)})});
    ASSERT_EQ(dets_of(out[0])->num_objects, 2u);
    unref_all(out);
}

TEST(samurai_without_engine_dir_refuses_to_start) {
    GstElement *e = make("nvmmsamurai");
    ASSERT_EQ(start_failure(e), std::string("engine-dir property is required"));
    gst_object_unref(e);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
