"""任务运行模式与阶段开关。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RunOptions:
    """一次真实或 SITL 任务的阶段选择。"""

    mode: Literal["real", "sitl"]
    run_mission: bool = True
    capture: bool = True
    recognize: bool = True
    record: bool = True
    drop_stage: bool = True
    real_release: bool = False
    auto_recon: bool | None = None
    auto_land: bool = True
    max_seconds: float | None = None
    qgc_video: bool | None = None

    def validated(self) -> RunOptions:
        """检查互斥与模式约束，确保失败发生在任何资源创建之前。"""
        if self.mode not in {"real", "sitl"}:
            raise ValueError("运行模式只能是 real 或 sitl")
        if self.recognize and not self.capture:
            raise ValueError("开启识别必须同时开启视频采集")
        if self.qgc_video and not self.capture:
            raise ValueError("开启 QGC 视频发布必须同时开启视频采集")
        if self.real_release and (
            self.mode != "real" or not self.run_mission or not self.drop_stage
        ):
            raise ValueError("真实投弹只允许 real 模式的任务投放阶段")
        if self.mode == "sitl" and self.real_release:
            raise ValueError("SITL 禁止真实投弹")
        if self.max_seconds is not None and (
            not math.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds 必须是正的有限秒数")
        if self.run_mission and self.max_seconds is not None:
            raise ValueError("max_seconds 只用于被动采集模式")
        return self

    @property
    def recon_upload(self) -> str:
        """返回侦查航线上传方。"""
        enabled = self.auto_recon if self.auto_recon is not None else self.mode == "sitl"
        return "auto" if enabled else "operator"
