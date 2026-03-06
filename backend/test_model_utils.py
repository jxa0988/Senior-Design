import numpy as np
import pytest

import core.model_utils as mu


# ------------------------------------------------------------
# Test sigmoid function
# Ensures the mathematical sigmoid behaves correctly
# ------------------------------------------------------------
def test_sigmoid_basic():
    x = np.array([0.0])
    y = mu.sigmoid(x)

    # sigmoid(0) should equal 0.5
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(0.5, rel=1e-6)


# ------------------------------------------------------------
# Test bounding box conversion
# Converts center format (cx, cy, w, h) → (xmin, ymin, xmax, ymax)
# ------------------------------------------------------------
def test_box_cxcywh_to_xyxyn():
    # Example box
    # cx=10, cy=20, w=4, h=6
    inp = np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)

    out = mu.box_cxcywh_to_xyxyn(inp)

    # Expected coordinates
    # xmin = 10 - 2
    # ymin = 20 - 3
    # xmax = 10 + 2
    # ymax = 20 + 3
    assert out.shape == (1, 4)
    assert out[0].tolist() == pytest.approx([8.0, 17.0, 12.0, 23.0])


# ------------------------------------------------------------
# Test IoU calculation between boxes
# Ensures overlap and non-overlap cases are handled correctly
# ------------------------------------------------------------
def test_iou_xyxy_simple_overlap():
    a = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

    b = np.array([
        [0.0, 0.0, 2.0, 2.0],   # identical box
        [1.0, 1.0, 3.0, 3.0],   # partial overlap
        [3.0, 3.0, 4.0, 4.0],   # no overlap
    ], dtype=np.float32)

    ious = mu._iou_xyxy(a, b)

    # identical boxes → IoU ≈ 1
    assert float(ious[0]) == pytest.approx(1.0, rel=1e-6)

    # no overlap → IoU = 0
    assert float(ious[2]) == pytest.approx(0.0, abs=1e-6)

    # partial overlap → between 0 and 1
    assert 0.0 < float(ious[1]) < 1.0


# ------------------------------------------------------------
# Test Non-Maximum Suppression (NMS)
# Ensures overlapping boxes with lower scores are removed
# ------------------------------------------------------------
def test_nms_numpy_keeps_best_when_overlap_high():

    boxes = np.array([
        [0.0, 0.0, 2.0, 2.0],   # best box
        [0.2, 0.2, 2.2, 2.2],   # overlapping box (should be removed)
        [5.0, 5.0, 6.0, 6.0],   # separate box (should remain)
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = mu.nms_numpy(boxes, scores, iou_thres=0.5)

    # Only best overlapping box + independent box remain
    assert keep.tolist() == [0, 2]


# ------------------------------------------------------------
# Edge case: NMS with no boxes
# Should return an empty array
# ------------------------------------------------------------
def test_nms_numpy_empty_returns_empty():
    keep = mu.nms_numpy(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32)
    )

    assert keep.dtype == np.int64
    assert keep.size == 0


# ------------------------------------------------------------
# Test class-wise NMS
# Ensures suppression happens independently per class
# ------------------------------------------------------------
def test_classwise_nms_runs_per_class_and_sorts():

    boxes = np.array([
        [0.0, 0.0, 2.0, 2.0],      # class 0 (best)
        [0.2, 0.2, 2.2, 2.2],      # class 0 (overlap → removed)
        [10.0, 10.0, 12.0, 12.0],  # class 1
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
    labels = np.array([0, 0, 1], dtype=np.int64)

    keep = mu.classwise_nms(boxes, scores, labels, iou_thres=0.5)

    # Highest score kept per class
    # Sorted by score descending
    assert keep.tolist() == [0, 2]


# ------------------------------------------------------------
# Test inference pipeline when model detects nothing
# Should return (None, None)
# ------------------------------------------------------------
def test_run_rfdetr_inference_returns_none_when_no_detections(monkeypatch):

    class FakeModel:
        def predict(self, image_path, confidence_threshold=0.5):
            return np.array([]), np.array([]), np.zeros((0, 4)), None

    dets, path = mu.run_rfdetr_inference(FakeModel(), "img.jpg", threshold=0.5)

    assert dets is None
    assert path is None


# ------------------------------------------------------------
# Test inference pipeline when detections exist
# Ensures detection dictionary is constructed correctly
# ------------------------------------------------------------
def test_run_rfdetr_inference_builds_detection_dict(monkeypatch, tmp_path):

    class FakeModel:

        # Simulate model output
        def predict(self, image_path, confidence_threshold=0.5):
            scores = np.array([0.9], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)
            boxes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            masks = None
            return scores, labels, boxes, masks

        # Simulate saving an output image
        def save_detections(self, image_path, boxes, labels, masks, save_image_path, class_names=None):
            save_image_path.parent.mkdir(parents=True, exist_ok=True)
            save_image_path.write_bytes(b"fake-image")

    dets, out_path = mu.run_rfdetr_inference(
        FakeModel(),
        "img.jpg",
        class_names=["wind", "hail"],
        save_dir=str(tmp_path),
        threshold=0.5,
    )

    # Ensure detection dictionary is correctly formed
    assert isinstance(dets, dict)
    assert dets["scores"] == pytest.approx([0.9], rel=1e-6)
    assert dets["labels"] == [1]
    assert dets["boxes"] == [[1.0, 2.0, 3.0, 4.0]]

    # Ensure output image path was created
    assert out_path.endswith("_pred.jpg")