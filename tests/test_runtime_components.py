"""在线与回放共用运行时组件工厂的回归。"""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import cv2
import pytest

from airdrop.runtime.components import (
    PerceptionComponents,
    build_alignment_buffer,
    prepare_perception_components,
)

if TYPE_CHECKING:
    from airdrop.georef.camera import CameraModel
    from airdrop.perception.pipeline import PerceptionWorker


def test_alignment_factory_uses_shared_raw_capacity_and_supports_disabled_capture(
    tmp_path: Path,
) -> None:
    assert build_alignment_buffer(capture=False) is None
    buffer = build_alignment_buffer(capture=True, ts_directory=tmp_path)
    assert buffer is not None
    try:
        stats = buffer.stats
        assert stats.storage == "raw"
        assert stats.capacity == 30 * 180
        assert stats.max_bytes is None
    finally:
        buffer.close_cold_storage()


def test_shared_perception_preparation_warms_detector_at_output_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class Detector:
        def load(self) -> None:
            calls.append(("load",))

        def warmup(self, width: int, height: int) -> bool:
            calls.append(("warmup", width, height))
            return True

    monkeypatch.setattr(cv2, "setNumThreads", lambda count: calls.append(("threads", count)))
    components = PerceptionComponents(
        camera=cast("CameraModel", SimpleNamespace(image_size=(1920, 1080))),
        worker=cast("PerceptionWorker", SimpleNamespace(detector=Detector())),
    )

    prepare_perception_components(components, output_size=(1920, 1080), opencv_threads=3)

    assert calls == [("load",), ("threads", 3), ("warmup", 1920, 1080)]


def test_shared_perception_preparation_rejects_calibration_size_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda _count: calls.append("threads"))
    components = PerceptionComponents(
        camera=cast("CameraModel", SimpleNamespace(image_size=(1280, 720))),
        worker=cast(
            "PerceptionWorker",
            SimpleNamespace(
                detector=SimpleNamespace(
                    load=lambda: calls.append("load"),
                    warmup=lambda *_size: calls.append("warmup") or True,
                )
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="标定尺寸"):
        prepare_perception_components(components, output_size=(1920, 1080), opencv_threads=3)

    assert calls == []
