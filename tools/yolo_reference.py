#!/usr/bin/env python3
"""YOLO11/v8 reference on the ONNX the TRT engine was built from, with its own letterbox, decode and NMS.

Prints {"image_w", "image_h", "detections": [{class_id, label, conf, x, y, w, h}]} in original-image pixels."""
import argparse
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


def letterbox(img, size):
    """Resize keeping aspect ratio, pad to a square `size` (gray 114). Returns
    the padded image plus the scale and (left,top) pad used to un-map boxes."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, px, py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read image: {a.image}", file=sys.stderr)
        return 2
    H, W = img.shape[:2]

    padded, scale, px, py = letterbox(img, a.imgsz)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    pred = out[0].T

    boxes_cxcywh = pred[:, :4]
    scores_all = pred[:, 4:]
    class_ids = scores_all.argmax(1)
    confs = scores_all.max(1)
    keep = confs >= a.conf
    boxes_cxcywh, class_ids, confs = boxes_cxcywh[keep], class_ids[keep], confs[keep]

    xywh = boxes_cxcywh.copy()
    xywh[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - px) / scale
    xywh[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - py) / scale
    xywh[:, 2] = boxes_cxcywh[:, 2] / scale
    xywh[:, 3] = boxes_cxcywh[:, 3] / scale

    idx = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), a.conf, a.iou)
    dets = []
    for i in np.array(idx).flatten():
        cid = int(class_ids[i])
        x, y, w, h = (float(v) for v in xywh[i])
        dets.append({
            "class_id": cid,
            "label": COCO_CLASS_NAMES[cid] if cid < len(COCO_CLASS_NAMES) else str(cid),
            "conf": round(float(confs[i]), 4),
            "x": round(x, 1), "y": round(y, 1),
            "w": round(w, 1), "h": round(h, 1),
        })
    dets.sort(key=lambda d: -d["conf"])
    json.dump({"image_w": W, "image_h": H, "detections": dets}, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
