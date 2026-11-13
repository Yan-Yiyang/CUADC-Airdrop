"""检测预处理与坐标解算的相机契约，使用合成图案而非模型权重。"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from airdrop import Config, Detection, PixelBox, TargetTracker, TelemetrySnapshot
from airdrop.georef import CameraModel
from airdrop.perception.detector import Detector, DetectorConfig, FrameSizeMismatchError


class _Array:
    """提供检测器读取张量所需的最小接口，不导入 GPU 运行库。"""

    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _Boxes(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.xyxy.values)


class _MarkerModel:
    """从实际输入画面找亮点；只替换识别器，保留真实 remap 与几何解算。"""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray, **kwargs):
        self.calls += 1
        weights = image[:, :, 0].astype(float)
        yy, xx = np.indices(weights.shape)
        u = float(np.sum(weights * xx) / weights.sum())
        v = float(np.sum(weights * yy) / weights.sum())
        boxes = _Boxes(
            xyxy=_Array([[u - 2, v - 2, u + 2, v + 2]]),
            conf=_Array([1.0]),
            cls=_Array([0]),
        )
        return [SimpleNamespace(boxes=boxes)]


@pytest.mark.parametrize("ray", [(0.45, 0.28), (-0.45, -0.28), (0.0, 0.0)])
def test_remapped_and_raw_pixels_resolve_the_same_known_ray(ray: tuple[float, float]) -> None:
    """正向径向/切向畸变构造输入，期望地面点由针孔比例独立给出。"""
    camera = CameraModel(
        camera_matrix=np.array([[450.0, 0, 301.0], [0, 470.0, 227.0], [0, 0, 1.0]]),
        dist_coeffs=np.array([-0.3, 0.04, 0.002, -0.003, 0.0]),
        image_size=(640, 480),
        r_bc=np.eye(3),
    )
    x, y = ray
    r2 = x * x + y * y
    radial = 1 - 0.3 * r2 + 0.04 * r2 * r2
    xd = x * radial + 2 * 0.002 * x * y - 0.003 * (r2 + 2 * x * x)
    yd = y * radial + 0.002 * (r2 + 2 * y * y) - 2 * 0.003 * x * y
    raw_pixel = (450.0 * xd + 301.0, 470.0 * yd + 227.0)
    yy, xx = np.indices((480, 640))
    spot = np.rint(255 * np.exp(-((xx - raw_pixel[0]) ** 2 + (yy - raw_pixel[1]) ** 2) / 8))
    image = np.repeat(spot.astype(np.uint8)[:, :, None], 3, axis=2)
    snapshot = TelemetrySnapshot(
        north_m=10.0, east_m=20.0, down_m=-50.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0
    )
    detector = Detector(
        DetectorConfig(
            camera_matrix=camera.camera_matrix,
            dist_coeffs=camera.dist_coeffs,
            calib_size=camera.image_size,
        )
    )
    detector._model = _MarkerModel()
    batch = detector.detect(image, telemetry=snapshot, mode="cls12")
    assert batch.undistorted
    assert len(batch.detections) == 1
    raw = Detection(
        frame_index=0,
        capture_timestamp=0.0,
        pixel=raw_pixel,
        box=PixelBox(0, 0, 1, 1),
        confidence=1.0,
        telemetry=snapshot,
        extra={"undistorted": False, "image_size": (640, 480)},
    )
    tracker = TargetTracker(Config(), camera=camera)
    raw_point = tracker.add(raw)
    rectified_point = tracker.add(batch.detections[0])
    assert raw_point is not None and rectified_point is not None
    expected = (10.0 + 50.0 * x, 20.0 + 50.0 * y)
    # 逐点反畸变使用有限次迭代；此强畸变样本的地面误差应小于 0.5mm。
    assert (raw_point.north_m, raw_point.east_m) == pytest.approx(expected, abs=5e-4)
    # 真实栅格图经线性插值后，亮点质心允许亚像素离散误差（50m / 450px）。
    assert (rectified_point.north_m, rectified_point.east_m) == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize(
    ("enabled", "override"),
    [
        pytest.param(True, None, id="first-frame"),
        pytest.param(False, None, id="disabled-in-config"),
        pytest.param(True, False, id="disabled-for-frame"),
    ],
)
def test_detector_rejects_wrong_size_before_inference(
    enabled: bool, override: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首帧与标定尺寸不符时，任何去畸变设置都应在加载模型前报错。"""
    detector = Detector(
        DetectorConfig(
            camera_matrix=np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1.0]]),
            calib_size=(640, 480),
            undistort=enabled,
        )
    )
    monkeypatch.setattr(detector, "load", lambda: pytest.fail("尺寸不符不应加载或运行模型"))
    with pytest.raises(FrameSizeMismatchError, match="320x240.*640, 480.*停止运行"):
        detector.detect(np.zeros((240, 320, 3), np.uint8), undistort=override)


def test_detector_rejects_size_change_when_only_the_cached_size_is_known() -> None:
    detector = Detector(
        DetectorConfig(camera_matrix=np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1.0]]))
    )
    model = _MarkerModel()
    detector._model = model
    first = np.zeros((480, 640, 3), np.uint8)
    first[240, 320] = 255
    detector.detect(first)
    assert model.calls == 1
    image = np.zeros((240, 320, 3), np.uint8)
    image[120, 160] = 255
    with pytest.raises(FrameSizeMismatchError):
        detector.detect(image, undistort=False)
    assert model.calls == 1


def test_detector_records_matching_size() -> None:
    detector = Detector(DetectorConfig(calib_size=(64, 48)))
    model = _MarkerModel()
    detector._model = model
    good = np.zeros((48, 64, 3), np.uint8)
    good[24, 32] = 255
    detection = detector.detect(good).detections[0]
    assert detection.extra["image_size"] == (64, 48)
    assert model.calls == 1


@pytest.mark.parametrize(
    ("camera_size", "extra"),
    [
        pytest.param((640, 480), {"image_size": [320, 240]}, id="different-size"),
        pytest.param((640, 480), {}, id="missing-field"),
        pytest.param((640, 480), {"image_size": [640.0, 480]}, id="float-equals-int"),
        pytest.param(None, {"image_size": None}, id="null-without-reference"),
        pytest.param(None, {"image_size": "640x480"}, id="not-a-sequence-of-dimensions"),
        pytest.param(None, {"image_size": [640]}, id="missing-height"),
        pytest.param(None, {"image_size": [True, 480]}, id="boolean"),
        pytest.param(None, {"image_size": [640, 0]}, id="nonpositive"),
    ],
)
def test_tracker_rejects_invalid_or_incompatible_image_size(camera_size, extra) -> None:
    """外部记录缺尺寸、尺寸畸形或与标定不符时不产生定位点。"""
    camera = CameraModel(
        camera_matrix=np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1.0]]),
        image_size=camera_size,
    )
    events: list[str] = []
    tracker = TargetTracker(
        Config(), camera=camera, undistort=False, on_event=lambda kind, _: events.append(kind)
    )
    detection = Detection(
        frame_index=1,
        capture_timestamp=1.0,
        pixel=(320.0, 240.0),
        box=PixelBox(310, 230, 330, 250),
        confidence=1.0,
        telemetry=TelemetrySnapshot(
            north_m=10.0, east_m=20.0, down_m=-50.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0
        ),
        extra=extra,
    )
    assert tracker.add(detection) is None
    assert not tracker.points
    assert tracker.stats["image_size_mismatch"] == 1
    assert events == ["image_size_mismatch"]
    tracker.reset()
    assert tracker.stats["image_size_mismatch"] == 0


def test_tracker_accepts_matching_sizes_and_json_records() -> None:
    camera = CameraModel(
        camera_matrix=np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1.0]]),
        image_size=(640, 480),
    )
    detector = Detector(DetectorConfig())
    detector._model = _MarkerModel()
    image = np.zeros((480, 640, 3), np.uint8)
    image[240, 320] = 255
    snapshot = TelemetrySnapshot(
        north_m=10.0, east_m=20.0, down_m=-50.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0
    )
    detection = detector.detect(image, telemetry=snapshot).detections[0]
    # 实际写入 JSON 后尺寸变成列表，也必须与元组形式兼容。
    extra = json.loads(json.dumps(detection.as_dict()))["extra"]
    assert extra["image_size"] == [640, 480]
    from_json = replace(detection, extra=extra)
    for model, record in (
        (camera, detection),
        (camera, from_json),
        (replace(camera, image_size=None), from_json),
    ):
        tracker = TargetTracker(Config(), camera=model)
        point = tracker.add(record)
        assert point is not None
        assert (point.north_m, point.east_m) == pytest.approx((10.0, 20.0))
        assert tracker.stats["image_size_mismatch"] == 0


def test_scaled_camera_must_be_shared_by_detector_and_tracker() -> None:
    camera = CameraModel(
        camera_matrix=np.array([[450.0, 0, 320], [0, 450.0, 240], [0, 0, 1.0]]),
        image_size=(640, 480),
    )
    scaled = camera.scaled(0.5)
    config = Config()
    pipeline = config.perception.to_pipeline_config(scaled)
    detector = Detector(
        DetectorConfig(
            camera_matrix=pipeline.camera_matrix,
            dist_coeffs=pipeline.dist_coeffs,
            calib_size=pipeline.calib_size,
        )
    )
    detector._model = _MarkerModel()
    image = np.zeros((240, 320, 3), np.uint8)
    image[120, 160] = 255
    snapshot = TelemetrySnapshot(
        north_m=10.0, east_m=20.0, down_m=-50.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0
    )
    detection = detector.detect(image, telemetry=snapshot).detections[0]
    assert TargetTracker(config, camera=camera).add(detection) is None
    point = TargetTracker(config, camera=scaled).add(detection)
    assert point is not None
    assert (point.north_m, point.east_m) == pytest.approx((10.0, 20.0))
