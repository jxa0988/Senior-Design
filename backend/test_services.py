import pytest
from core.services import RFDETRService


def test_predict_normal(monkeypatch):
    monkeypatch.setattr(
        RFDETRService,
        "_load_model",
        classmethod(lambda cls: ("MODEL", ["wind"], "ONNX")),
    )

    def fake_run(model, image_path, class_names, threshold):
        return ["det"], "output.jpg"

    monkeypatch.setattr(RFDETRService, "_run_normal", fake_run, raising=False)

    detections, path = RFDETRService.predict("image.jpg", mode="normal", threshold=0.55)

    assert detections == ["det"]
    assert path == "output.jpg"


def test_predict_tiled(monkeypatch):
    monkeypatch.setattr(
        RFDETRService,
        "_load_model",
        classmethod(lambda cls: ("MODEL", ["wind"], "ONNX")),
    )

    monkeypatch.setattr(RFDETRService, "_download_image", lambda url: "downloaded.jpg")

    def fake_run_tiled(**kwargs):
        return ["det"], "tiled.jpg"

    monkeypatch.setattr(RFDETRService, "_run_tiled", fake_run_tiled, raising=False)

    detections, path = RFDETRService.predict(
        "http://example.com/img.jpg",
        mode="tiled",
        threshold=0.2,
        tile_size=600,
    )

    assert detections == ["det"]
    assert path == "tiled.jpg"