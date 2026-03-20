import numpy as np
import pytest
import io
from PIL import Image

import core.model_utils as mu


# Test open_image with a local file
# Covers lines 30–31
def test_open_image_local(tmp_path):

    img_path = tmp_path / "test.jpg"

    # create a simple image file
    img = Image.new("RGB", (10, 10))
    img.save(img_path)

    loaded = mu.open_image(str(img_path))

    assert loaded.size == (10, 10)



# Test open_image with URL (mocked request)
# Covers lines 28–29
def test_open_image_url(monkeypatch):

    img = Image.new("RGB", (5, 5))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class FakeResponse:
        content = buf.read()

    monkeypatch.setattr(mu.requests, "get", lambda url: FakeResponse())

    loaded = mu.open_image("https://example.com/image.jpg")

    assert loaded.size == (5, 5)


# Test sigmoid function
# Ensures the mathematical sigmoid behaves correctly
def test_sigmoid_basic():
    x = np.array([0.0])
    y = mu.sigmoid(x)

    # sigmoid(0) should equal 0.5
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(0.5, rel=1e-6)



# Test bounding box conversion
# Converts center format (cx, cy, w, h) → (xmin, ymin, xmax, ymax)
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


# Test IoU calculation between boxes
# Ensures overlap and non-overlap cases are handled correctly
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



# Test Non-Maximum Suppression (NMS)
# Ensures overlapping boxes with lower scores are removed
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



# Edge case: NMS with no boxes
# Should return an empty array
def test_nms_numpy_empty_returns_empty():
    keep = mu.nms_numpy(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32)
    )

    assert keep.dtype == np.int64
    assert keep.size == 0



# Test class-wise NMS
# Ensures suppression happens independently per class
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



# Test inference pipeline when model detects nothing
# Should return (None, None)
def test_run_rfdetr_inference_returns_none_when_no_detections(monkeypatch):

    class FakeModel:
        def predict(self, image_path, confidence_threshold=0.5):
            return np.array([]), np.array([]), np.zeros((0, 4)), None

    dets, path = mu.run_rfdetr_inference(FakeModel(), "img.jpg", threshold=0.5)

    assert dets is None
    assert path is None


# Test inference pipeline when detections exist
# Ensures detection dictionary is constructed correctly
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

def test_open_image_raises_when_file_missing():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mu.open_image("definitely_missing_file.jpg")

def test_rfdetr_onnx_init_sets_input_shape(monkeypatch):
    class FakeInput:
        shape = [1, 3, 640, 640]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())

    model = mu.RFDETR_ONNX("fake_model.onnx")

    assert model.input_height == 640
    assert model.input_width == 640
    assert isinstance(model.ort_session, FakeSession)

def test_rfdetr_onnx_init_raises_runtime_error(monkeypatch):
    def fake_session(path):
        raise Exception("load failed")

    monkeypatch.setattr(mu.ort, "InferenceSession", fake_session)

    with pytest.raises(RuntimeError, match="Failed to load ONNX model"):
        mu.RFDETR_ONNX("bad_model.onnx")

def test_preprocess_outputs_correct_shape_and_type(monkeypatch):
    class FakeInput:
        shape = [1, 3, 64, 64]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    # Mock ONNX session so model initializes
    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())

    model = mu.RFDETR_ONNX("fake.onnx")

    # Create a simple image
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))

    output = model._preprocess(img)

    # Expected shape: (1, 3, H, W)
    assert output.shape == (1, 3, 64, 64)

    # Ensure correct dtype
    assert output.dtype == np.float32

def test_post_process_without_masks(monkeypatch):
    class FakeInput:
        shape = [1, 3, 64, 64]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())
    model = mu.RFDETR_ONNX("fake.onnx")

    # 2 boxes in cx, cy, w, h normalized format
    boxes = np.array([[
        [0.5, 0.5, 0.4, 0.4],
        [0.2, 0.2, 0.2, 0.2],
    ]], dtype=np.float32)

    # logits so sigmoid gives strong scores
    logits = np.array([[
        [5.0, 1.0],
        [4.0, 0.5],
    ]], dtype=np.float32)

    outputs = [boxes, logits]

    scores, labels, out_boxes, masks = model._post_process(
        outputs,
        origin_height=100,
        origin_width=200,
        confidence_threshold=0.5,
        max_number_boxes=300,
    )

    assert masks is None
    assert len(scores) == 2
    assert len(labels) == 2
    assert out_boxes.shape == (2, 4)

def test_post_process_with_masks(monkeypatch):
    class FakeInput:
        shape = [1, 3, 64, 64]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())
    model = mu.RFDETR_ONNX("fake.onnx")

    boxes = np.array([[
        [0.5, 0.5, 0.4, 0.4],
        [0.2, 0.2, 0.2, 0.2],
    ]], dtype=np.float32)

    logits = np.array([[
        [5.0, 1.0],
        [4.0, 0.5],
    ]], dtype=np.float32)

    # two small masks
    masks = np.array([[
        [[0, 1], [1, 0]],
        [[1, 1], [0, 0]],
    ]], dtype=np.uint8)

    outputs = [boxes, logits, masks]

    scores, labels, out_boxes, out_masks = model._post_process(
        outputs,
        origin_height=50,
        origin_width=60,
        confidence_threshold=0.5,
        max_number_boxes=300,
    )

    assert len(scores) == 2
    assert len(labels) == 2
    assert out_boxes.shape == (2, 4)
    assert out_masks is not None
    assert out_masks.shape[0] == 2
    assert out_masks.dtype == np.uint8

def test_predict_calls_pipeline_correctly(monkeypatch):
    class FakeInput:
        name = "input"
        shape = [1, 3, 64, 64]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _, __):
            return [np.zeros((1, 2, 4), dtype=np.float32)]

    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())

    model = mu.RFDETR_ONNX("fake.onnx")

    fake_img = Image.new("RGB", (100, 200))
    monkeypatch.setattr(mu, "open_image", lambda path: fake_img)
    monkeypatch.setattr(model, "_preprocess", lambda img: "processed")
    monkeypatch.setattr(
        model,
        "_post_process",
        lambda outputs, h, w, t, m: ("scores", "labels", "boxes", "masks"),
    )

    result = model.predict("img.jpg", confidence_threshold=0.5)

    assert result == ("scores", "labels", "boxes", "masks")

def test_save_detections_without_masks_uses_default_font(monkeypatch, tmp_path):
    class FakeInput:
        shape = [1, 3, 64, 64]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())
    model = mu.RFDETR_ONNX("fake.onnx")

    base_img = Image.new("RGB", (50, 50), color=(0, 0, 0))
    monkeypatch.setattr(mu, "open_image", lambda path: base_img)

    real_default_font = mu.ImageFont.load_default()

    def fake_truetype(*args, **kwargs):
        raise Exception("font missing")

    monkeypatch.setattr(mu.ImageFont, "truetype", fake_truetype)
    monkeypatch.setattr(mu.ImageFont, "load_default", lambda: real_default_font)

    boxes = np.array([[5, 5, 20, 20]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    out_path = tmp_path / "out_no_masks.jpg"
    model.save_detections("img.jpg", boxes, labels, None, out_path, class_names=None)

    assert out_path.exists()

def test_save_detections_with_masks_and_class_names(monkeypatch, tmp_path):
    class FakeInput:
        shape = [1, 3, 64, 64]

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

    monkeypatch.setattr(mu.ort, "InferenceSession", lambda path: FakeSession())
    model = mu.RFDETR_ONNX("fake.onnx")

    base_img = Image.new("RGB", (60, 60), color=(10, 10, 10))
    monkeypatch.setattr(mu, "open_image", lambda path: base_img)

    boxes = np.array([[10, 10, 30, 30]], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)

    masks = np.zeros((1, 60, 60), dtype=np.uint8)
    masks[0, 10:30, 10:30] = 255

    out_path = tmp_path / "out_with_masks.jpg"
    model.save_detections(
        "img.jpg",
        boxes,
        labels,
        masks,
        out_path,
        class_names=["wind", "hail"],
    )

    assert out_path.exists()


def test_tiled_inference_raises_when_overlap_too_large(monkeypatch):
    fake_img = Image.new("RGB", (100, 100))
    monkeypatch.setattr(mu, "open_image", lambda path: fake_img)

    class FakeModel:
        ort_session = None

    with pytest.raises(ValueError, match="overlap too large"):
        mu.run_rfdetr_inference_tiled(
            FakeModel(),
            "img.jpg",
            tile_size=100,
            overlap=1.0,  # makes step = 0
        )


def test_tiled_inference_returns_none_when_no_detections(monkeypatch):
    fake_img = Image.new("RGB", (100, 100))
    monkeypatch.setattr(mu, "open_image", lambda path: fake_img)

    class FakeInput:
        name = "input"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _, __):
            return ["fake"]

    class FakeModel:
        def __init__(self):
            self.ort_session = FakeSession()

        def _preprocess(self, tile):
            return "processed"

        def _post_process(self, *args, **kwargs):
            return None, None, None, None  # triggers continue

    dets, path = mu.run_rfdetr_inference_tiled(
        FakeModel(),
        "img.jpg",
    )

    assert dets is None
    assert path is None


def test_tiled_inference_success(monkeypatch, tmp_path):
    fake_img = Image.new("RGB", (100, 100))
    monkeypatch.setattr(mu, "open_image", lambda path: fake_img)

    class FakeInput:
        name = "input"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _, __):
            return ["fake"]

    class FakeModel:
        def __init__(self):
            self.ort_session = FakeSession()

        def _preprocess(self, tile):
            return "processed"

        def _post_process(self, *args, **kwargs):
            scores = np.array([0.9], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)
            boxes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            return scores, labels, boxes, None

        def save_detections(self, *args, **kwargs):
            save_path = args[4]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(b"fake")

    dets, path = mu.run_rfdetr_inference_tiled(
        FakeModel(),
        "img.jpg",
        save_dir=str(tmp_path),
    )

    assert isinstance(dets, dict)
    assert dets["scores"] == pytest.approx([0.9])
    assert dets["labels"] == [1]
    assert isinstance(path, str)