"""集中式运行模块：``python -m airdrop.run <子命令> [选项]``。

为什么要有它
------------
示例与工具都是纯库模块（顶部常量 = 默认值，暴露 ``build_config(**覆盖)`` 与
``main(**kwargs)``）；命令行解析只在这里做一份，于是：

* 每个入口不必各自 ``import argparse``、各自写一遍参数；
* 参数到配置走关键字覆盖（各模块内部用 ``dataclasses.replace`` 派生），
  默认值直接取入口模块的常量，这里不重复写第二遍；
* 重依赖（torch / ultralytics / cv2 / mavsdk / rapidocr / onnxruntime）只在
  handler 真正跑的时候才由各模块的函数体导入——``--help`` 与 ``check-docs``
  都不会加载它们（``tests/test_cli.py` 用子进程盯着这条）。

用法
----
::

    ./.venv/Scripts/python.exe -m airdrop.run --help
    ./.venv/Scripts/python.exe -m airdrop.run full-mission --help
    ./.venv/Scripts/python.exe -m airdrop.run full-mission --rtsp-url rtsp://... --no-preflight
    ./.venv/Scripts/python.exe -m airdrop.run replay --flight flights/20260913-185512 --speed 2
    ./.venv/Scripts/python.exe -m airdrop.run calibrate --flight flights/20260913-185512 --strict
    ./.venv/Scripts/python.exe -m airdrop.run check-docs

退出码
------
``0`` 正常；``2`` 参数/子命令不合法（argparse 的约定，也是各 handler 自己的"未成功"
口径）；其余由 handler 决定（例如 ``fit-ballistics`` 的 2 = 参数不可信、3 = 缺测量）。

加一个子命令
------------
在 :data:`SUBCOMMANDS` 里加一条即可：``module`` 指向入口模块，``handler`` 是它的函数名，
``options`` 用 :class:`Option` 描述（``source`` 写入口模块里的常量名，默认值取自它）。
``tools/check_docs.py`` 会用这份注册表核验手册是否覆盖了全部子命令。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SUBCOMMANDS", "Option", "Subcommand", "build_parser", "main", "resolve_kwargs"]


# ----------------------------------------------------------------------
# 选项与子命令的描述（纯数据：不 import 任何入口模块）
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Option:
    """一个命令行选项 → 传给入口模块 ``main(**kwargs)`` 的关键字。

    ``source`` 是入口模块里的常量名：``--help`` 里显示的默认值就是它的当前值，
    所以默认值只在模块里写一遍（这里不重复）。
    """

    flags: tuple[str, ...]
    dest: str
    help: str
    #: str / int / float / path / choice / flag / flag-off / bool / pair / vec2 / vec3 / const
    kind: str = "str"
    choices: tuple[str, ...] = ()
    #: 默认值取自入口模块的这个常量（``const`` 用它作为"开关打开时要传的值"）
    source: str = ""


@dataclass(frozen=True, slots=True)
class Subcommand:
    """一个子命令：入口模块 + 选项表。"""

    name: str
    module: str
    help: str
    options: tuple[Option, ...] = ()
    #: 入口模块里干活的函数（签名形如 ``main(**kwargs) -> int``）
    handler: str = "main"


#: 子命令注册表（唯一的命令行出处；顺序即 ``--help`` 里的显示顺序）
SUBCOMMANDS: dict[str, Subcommand] = {
    "full-mission": Subcommand(
        name="full-mission",
        module="examples.full_mission",
        help="完整任务：自检 → 等起飞 → 侦察 → 飞掠投放 → 降落（正式任务档）",
        options=(
            Option(
                ("--system-address",),
                "system_address",
                "MAVSDK 系统地址（飞控连接）",
                source="SYSTEM_ADDRESS",
            ),
            Option(("--rtsp-url",), "rtsp_url", "RTSP 视频地址", source="RTSP_URL"),
            Option(
                ("--telemetry-lag",),
                "telemetry_lag_s",
                "画面-遥测链路延时（秒）",
                kind="float",
                source="TELEMETRY_LAG_S",
            ),
            Option(
                ("--land-plan",),
                "land_plan",
                "降落段用的 QGC .plan 路径（与内置降落航点二选一）",
                source="LAND_PLAN",
            ),
            Option(
                ("--recon-upload",),
                "recon_upload",
                "侦察航线谁上传：operator=操作手在 QGC 启动（正式任务）/ auto=本包上传（自动测试）",
                kind="choice",
                choices=("operator", "auto"),
                source="RECON_UPLOAD",
            ),
            Option(
                ("--no-video",),
                "use_video",
                "本架次不接视频（跳过视频与感知，起飞前自检的四项检查也相应关闭）",
                kind="flag-off",
                source="USE_VIDEO",
            ),
            Option(
                ("--no-preflight",),
                "preflight",
                "关闭 config.preflight 的全部检查（关闭不等于通过：每项仍记 ok=null 事件）",
                kind="const",
                source="PREFLIGHT_OFF",
            ),
            Option(
                ("--require-airborne",),
                "require_airborne",
                "是否等到飞机确实在空中才进侦察（地面演练用 --no-require-airborne）",
                kind="bool",
                source="REQUIRE_AIRBORNE",
            ),
            Option(
                ("--dry-run",),
                "dry_run",
                "投放只记日志、不下发指令（不装弹演练）",
                kind="flag",
                source="DRY_RUN",
            ),
            Option(
                ("--no-record",),
                "record",
                "不写飞行目录（默认记录五个文件）",
                kind="flag-off",
                source="RECORD",
            ),
        ),
    ),
    "sitl": Subcommand(
        name="sitl",
        module="examples.sitl_mission",
        help="PX4 SITL 演练：合成目标 + DryRunController（不装弹、不用视频）",
        options=(
            Option(
                ("--system-address",),
                "system_address",
                "MAVSDK 系统地址（SITL 默认 udpin://0.0.0.0:14540）",
                source="SYSTEM_ADDRESS",
            ),
            Option(
                ("--land-plan",),
                "land_plan",
                "降落段用的 QGC .plan 路径（为空则用内置降落航点）",
                source="LAND_PLAN",
            ),
            Option(
                ("--recon-upload",),
                "recon_upload",
                "侦察航线谁上传：auto=本包上传（演练档）/ operator=等 QGC 启动",
                kind="choice",
                choices=("operator", "auto"),
                source="RECON_UPLOAD",
            ),
            Option(
                ("--require-airborne",),
                "require_airborne",
                "是否等到飞机确实在空中（演练默认不等待，见 REQUIRE_AIRBORNE）",
                kind="bool",
                source="REQUIRE_AIRBORNE",
            ),
            Option(
                ("--target-offset",),
                "target_offset_ned",
                "合成目标相对盘旋点的 NED 偏移，格式 N,E,D（米）",
                kind="vec3",
                source="TARGET_OFFSET_NED",
            ),
            Option(
                ("--no-record",),
                "record",
                "不写飞行目录（默认记录五个文件）",
                kind="flag-off",
                source="RECORD",
            ),
        ),
    ),
    "replay": Subcommand(
        name="replay",
        module="examples.replay_flight",
        help="回放全链路：识别 → 坐标 → 统计 → 航线（离线，不上传）",
        options=(
            Option(
                ("--flight",),
                "flight_dir",
                "飞行目录（或 flights/ 取最新）",
                kind="path",
                source="FLIGHT_DIR",
            ),
            Option(
                ("--speed",),
                "speed",
                "回放速度：0=全速 / 1.0=原速 / 2.0=两倍速",
                kind="float",
                source="SPEED",
            ),
            Option(
                ("--strict",),
                "strict",
                "回放异常（缺帧/sink 抛异常）就让本次以 error 结束",
                kind="bool",
                source="STRICT",
            ),
            Option(
                ("--buffer-seconds",),
                "buffer_seconds",
                "环形缓冲保留时长（秒）",
                kind="float",
                source="BUFFER_SECONDS",
            ),
            Option(
                ("--perception-mode",),
                "perception_mode",
                "感知模式：ocr（YOLO + 读数）/ cls12（12 类直出）",
                kind="choice",
                choices=("ocr", "cls12"),
                source="PERCEPTION_MODE",
            ),
            Option(
                ("--selection-rule",),
                "selection_rule",
                "跨候选类选唯一结果的规则：median / max",
                kind="choice",
                choices=("median", "max"),
                source="SELECTION_RULE",
            ),
        ),
    ),
    "basic": Subcommand(
        name="basic",
        module="examples.basic_usage",
        help="最小示例：遥测订阅 + 按时间戳查询（可选 arm/takeoff/land 演示）",
        options=(
            Option(("--system-address",), "system_address", "MAVSDK 系统地址", source="ADDRESS"),
            Option(
                ("--demo-commands",),
                "demo_commands",
                "演示 arm → takeoff → land（仅在拆除桨叶或 SITL 环境中使用）",
                kind="flag",
                source="DEMO_COMMANDS",
            ),
        ),
    ),
    "hm30-video": Subcommand(
        name="hm30-video",
        module="examples.hm30_video",
        help="RTSP 视频拉流：统计链路质量（可选预览/存盘）",
        options=(
            Option(("--rtsp-url",), "url", "视频 RTSP 地址", source="URL"),
            Option(
                ("--preview",), "preview", "开窗口预览（按 q 退出）", kind="flag", source="PREVIEW"
            ),
            Option(
                ("--save-path",), "save_path", "存成 mp4 的路径（默认不存）", source="SAVE_PATH"
            ),
            Option(
                ("--duration",), "duration", "运行秒数（0=不限）", kind="float", source="DURATION"
            ),
        ),
    ),
    "video-sync": Subcommand(
        name="video-sync",
        module="examples.video_telemetry_sync",
        help="帧-遥测对齐与逐帧留存演示（每一帧都进环形缓冲）",
        options=(
            Option(("--system-address",), "system_address", "MAVSDK 系统地址", source="ADDRESS"),
            Option(("--rtsp-url",), "video_url", "视频 RTSP 地址", source="VIDEO_URL"),
            Option(
                ("--telemetry-lag",), "lag", "画面-遥测链路延时（秒）", kind="float", source="LAG"
            ),
            Option(
                ("--run-seconds",),
                "run_seconds",
                "运行时长（0=一直跑）",
                kind="float",
                source="RUN_SECONDS",
            ),
        ),
    ),
    "calibration-capture": Subcommand(
        name="calibration-capture",
        module="examples.calibration_capture",
        help="标定素材采集：录一个标准飞行目录给 calibrate 用",
        options=(
            Option(
                ("--system-address",),
                "system_address",
                "MAVSDK 系统地址（None = 用默认）",
                source="SYSTEM_ADDRESS",
            ),
            Option(
                ("--rtsp-url",),
                "rtsp_url",
                "RTSP 视频地址（None = 用默认地址）",
                source="RTSP_URL",
            ),
            Option(
                ("--telemetry-lag",),
                "telemetry_lag_s",
                "写入录像配置的链路延时（秒）——本次采集不要用旧标定值",
                kind="float",
                source="TELEMETRY_LAG_S",
            ),
            Option(
                ("--buffer-seconds",),
                "buffer_seconds",
                "环形缓冲保留时长（秒）",
                kind="float",
                source="BUFFER_SECONDS",
            ),
            Option(
                ("--max-seconds",),
                "max_seconds",
                "采集上限（秒），到点自动停",
                kind="float",
                source="MAX_SECONDS",
            ),
        ),
    ),
    "calibrate": Subcommand(
        name="calibrate",
        module="tools.calibrate",
        help="三步相机标定：内参 → 画面/遥测时间差 → 手眼外参",
        options=(
            Option(
                ("--flight",), "flight_dir", "标定采集的飞行目录", kind="path", source="FLIGHT_DIR"
            ),
            Option(
                ("--out",),
                "output_path",
                "标定结果 JSON 的输出路径",
                kind="path",
                source="OUTPUT_PATH",
            ),
            Option(
                ("--pattern",),
                "pattern_size",
                "棋盘格内角点数，格式 列,行（内角点数，不是方格数）",
                kind="pair",
                source="PATTERN_SIZE",
            ),
            Option(
                ("--square",), "square_size", "方格边长（米）", kind="float", source="SQUARE_SIZE_M"
            ),
            Option(
                ("--lag-search",),
                "lag_search_s",
                "时间差搜索半径（秒）",
                kind="float",
                source="LAG_SEARCH_S",
            ),
            Option(
                ("--lag-step",),
                "lag_step_s",
                "时间差搜索步长（秒）",
                kind="float",
                source="LAG_STEP_S",
            ),
            Option(
                ("--max-frames",), "max_frames", "最多读取多少帧", kind="int", source="MAX_FRAMES"
            ),
            Option(
                ("--strict",),
                "strict",
                "严格模式：外参/时间差没标出来就返回 2（默认只警告）",
                kind="flag",
                source="STRICT",
            ),
        ),
    ),
    "fit-ballistics": Subcommand(
        name="fit-ballistics",
        module="tools.fit_ballistics",
        help="投放试验反演弹道参数（最小二乘 + 可辨识性诊断）",
        options=(
            Option(
                ("--drops",),
                "drops",
                "只跑这一个架次：飞行目录或 drops.jsonl（默认取最新架次）",
                kind="path",
                source="DROPS",
            ),
            Option(
                ("--impacts",),
                "impacts",
                "实测落点文件（impacts.jsonl / impacts.csv）",
                kind="path",
                source="IMPACTS",
            ),
            Option(
                ("--out",),
                "output_path",
                "反演报告 JSON 的输出路径",
                kind="path",
                source="OUTPUT_PATH",
            ),
            Option(
                ("--mass-kg",),
                "mass_kg",
                "弹体质量（kg）——实测值，不参与反演",
                kind="float",
                source="MEASURED_MASS_KG",
            ),
        ),
    ),
    "fetch-models": Subcommand(
        name="fetch-models",
        module="tools.fetch_models",
        help="把外部模型权重复制进 models/（只复制、不联网）",
        options=(
            Option(
                ("--source-dir",),
                "source_dir",
                "YOLO 权重所在目录",
                kind="path",
                source="SOURCE_DIR",
            ),
            Option(
                ("--source-ocr-dir",),
                "source_ocr_dir",
                "PP-OCR 权重所在目录",
                kind="path",
                source="SOURCE_OCR_DIR",
            ),
            Option(
                ("--target-dir",),
                "target_dir",
                "目标目录（工程内）",
                kind="path",
                source="TARGET_DIR",
            ),
        ),
    ),
    "make-world": Subcommand(
        name="make-world",
        module="tools.make_world",
        help="生成 CUADC 赛区 Gazebo 世界（round1 图片靶标 / round2 数字靶标）",
        options=(
            Option(
                ("--out-dir",),
                "out_dir",
                "输出目录（默认 sim/worlds/cuadc）",
                kind="path",
                source="OUT_DIR",
            ),
            Option(
                ("--seed",),
                "seed",
                "随机种子（决定各天井朝向；默认 0 = 入库布局）",
                kind="int",
                source="SEED",
            ),
            Option(
                ("--rounds",),
                "rounds",
                "生成哪几轮：both / 1 / 2",
                kind="choice",
                choices=("both", "1", "2"),
                source="ROUNDS",
            ),
            Option(
                ("--wind",),
                "wind_enu_m_s",
                "世界恒定风（东,北,天 m/s），默认静风",
                kind="vec3",
                source="WIND_ENU_M_S",
            ),
            Option(
                ("--well-arrows",),
                "well_arrows",
                '在天井里画"天井箭头"（规则文本提到、规则图 2/4 里没有；默认不画）',
                kind="flag",
                source="WELL_ARROWS",
            ),
            Option(
                ("--sun",),
                "sun_azel_deg",
                "太阳位置（方位角,仰角 度；方位从北起顺时针，仰角 0~90）——换阴影环境用，"
                "例如 90,12 = 东侧低角度长阴影",
                kind="vec2",
                source="SUN_AZEL_DEG",
            ),
            Option(
                ("--sun-random",),
                "sun_random",
                "太阳位置由 seed 随机（方位全向、仰角 15~80°，换 seed 换光照）；"
                "开了它 --sun 的固定值不生效",
                kind="flag",
                source="SUN_RANDOM",
            ),
            Option(
                ("--aerial-patches",),
                "aerial_patches",
                "比赛区域外铺几块航拍干扰底图（0 = 不铺；默认 6）",
                kind="int",
                source="AERIAL_PATCHES",
            ),
        ),
    ),
    "make-backdrops": Subcommand(
        name="make-backdrops",
        module="tools.make_backdrops",
        help="生成场景底图贴图：目标区地面纹理 + 比赛区域外的随机航拍干扰底图",
        options=(
            Option(
                ("--out-dir",),
                "out_dir",
                "输出目录（默认 sim/worlds/cuadc）",
                kind="path",
                source="OUT_DIR",
            ),
            Option(
                ("--seed",),
                "seed",
                "随机种子（决定底图长什么样；默认 0）",
                kind="int",
                source="SEED",
            ),
            Option(
                ("--size",),
                "size",
                "贴图边长（像素，默认 512）",
                kind="int",
                source="SIZE_PX",
            ),
            Option(
                ("--parts",),
                "parts",
                "写哪一类：all / ground（只目标区纹理，不动 fetch-aerial 抓来的航拍图）/ aerial",
                kind="choice",
                choices=("all", "ground", "aerial"),
                source="PARTS",
            ),
        ),
    ),
    "preview-world": Subcommand(
        name="preview-world",
        module="tools.preview_world",
        help="把赛区世界渲染成俯视预览图（PNG，带中文标注，供人眼核对布局）",
        options=(
            Option(
                ("--world-dir",),
                "world_dir",
                "世界目录（默认 sim/worlds/cuadc）",
                kind="path",
                source="WORLD_DIR",
            ),
            Option(
                ("--round",),
                "round_no",
                "预览哪一轮（1 = 图片靶标 / 2 = 数字靶标）",
                kind="int",
                source="ROUND",
            ),
            Option(
                ("--out",),
                "out_path",
                "输出 PNG 路径（默认 sim/worlds/cuadc/preview.png，已进 .gitignore）",
                kind="path",
                source="OUT_PATH",
            ),
        ),
    ),
    "fetch-aerial": Subcommand(
        name="fetch-aerial",
        module="tools.fetch_aerial",
        help="抓真实航拍底图（USGS NAIP，公有领域，需联网）替换程序化底图",
        options=(
            Option(
                ("--out-dir",),
                "out_dir",
                "输出目录（默认 sim/worlds/cuadc）",
                kind="path",
                source="OUT_DIR",
            ),
            Option(
                ("--size",),
                "size",
                "贴图边长（像素，默认 512）",
                kind="int",
                source="SIZE_PX",
            ),
            Option(
                ("--tile-width-m",),
                "tile_width_m",
                "每张底图覆盖的地面宽度（米，默认 160）",
                kind="float",
                source="TILE_WIDTH_M",
            ),
        ),
    ),
    "dump-api": Subcommand(
        name="dump-api",
        module="tools.dump_api",
        help="自省导出接口清单（docs/api_reference.md）",
        options=(
            Option(
                ("--out",), "output_path", "输出 Markdown 路径", kind="path", source="OUTPUT_PATH"
            ),
        ),
    ),
    "check-docs": Subcommand(
        name="check-docs",
        module="tools.check_docs",
        help="核验 docs/handbook.md 与代码是否一致（0 = 全过 / 1 = 有缺口）",
    ),
}


# ----------------------------------------------------------------------
# 解析器
# ----------------------------------------------------------------------
def _default_of(module: Any, option: Option) -> Any:
    """默认值直接取入口模块的常量（这里不写第二遍）。"""
    if not option.source:
        return None
    if not hasattr(module, option.source):
        raise AttributeError(
            f"{module.__name__} 里没有常量 {option.source}（{option.flags[0]} 的默认值出处）"
        )
    return getattr(module, option.source)


def _add_option(parser: argparse.ArgumentParser, module: Any, option: Option) -> None:
    default = _default_of(module, option)
    if option.kind == "const":
        help_text = f"{option.help}（开关；打开时传 {module.__name__}.{option.source}）"
        parser.add_argument(
            *option.flags, dest=option.dest, action="store_true", default=False, help=help_text
        )
        return
    help_text = f"{option.help}（默认：{module.__name__}.{option.source} = {default!r}）"
    common: dict[str, Any] = {"dest": option.dest, "default": default, "help": help_text}
    if option.kind == "flag":
        parser.add_argument(*option.flags, action="store_true", **common)
    elif option.kind == "flag-off":
        parser.add_argument(*option.flags, action="store_false", **common)
    elif option.kind == "bool":
        parser.add_argument(*option.flags, action=argparse.BooleanOptionalAction, **common)
    elif option.kind == "choice":
        parser.add_argument(*option.flags, choices=option.choices, **common)
    elif option.kind in ("int", "float"):
        parser.add_argument(*option.flags, type=float if option.kind == "float" else int, **common)
    elif option.kind == "path":
        parser.add_argument(*option.flags, type=Path, **common)
    elif option.kind == "pair":
        parser.add_argument(*option.flags, type=_pair, metavar="A,B", **common)
    elif option.kind == "vec2":
        parser.add_argument(*option.flags, type=_vec2, metavar="AZ,EL", **common)
    elif option.kind == "vec3":
        parser.add_argument(*option.flags, type=_vec3, metavar="N,E,D", **common)
    elif option.kind == "str":
        parser.add_argument(*option.flags, **common)
    else:  # pragma: no cover - 注册表写错时立刻抛出异常，不静默
        raise ValueError(f"未知的选项类型：{option.kind!r}（{option.flags[0]}）")


def _pair(text: str) -> tuple[int, int]:
    """``"9,6"`` → ``(9, 6)``（列,行）。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"要两个整数（形如 9,6）：{text!r}")
    return int(parts[0]), int(parts[1])


def _vec2(text: str) -> tuple[float, float]:
    """``"218,55"`` → ``(218.0, 55.0)``（例如太阳的 方位角,仰角）。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"要两个数（形如 218,55）：{text!r}")
    return float(parts[0]), float(parts[1])


def _vec3(text: str) -> tuple[float, float, float]:
    """``"300,0,-5"`` → ``(300.0, 0.0, -5.0)``（北,东,地）。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"要三个数（形如 300,0,0）：{text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def build_parser() -> argparse.ArgumentParser:
    """按 :data:`SUBCOMMANDS` 建出完整解析器（会 import 入口模块读默认值）。

    入口模块都是"纯库模块"（重依赖在函数体内），所以建解析器这一步不加载
    cv2 / mavsdk / torch——``--help`` 因此是廉价的。
    """
    parser = argparse.ArgumentParser(
        prog="python -m airdrop.run",
        description="AirDrop 集中式入口：子命令 + 选项，参数到配置走关键字覆盖",
        epilog="子命令的 --help 给出该入口的全部选项与默认值来源。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<子命令>")
    for spec in SUBCOMMANDS.values():
        module = importlib.import_module(spec.module)
        sub = subparsers.add_parser(
            spec.name,
            help=spec.help,
            description=f"{spec.help}（入口模块：{spec.module}）",
        )
        sub.set_defaults(command=spec.name)
        for option in spec.options:
            _add_option(sub, module, option)
    return parser


def _kwargs_of(spec: Subcommand, module: Any, args: argparse.Namespace) -> dict[str, Any]:
    """把解析结果翻成入口模块 ``main(**kwargs)`` 的关键字。"""
    kwargs: dict[str, Any] = {}
    for option in spec.options:
        value = getattr(args, option.dest)
        if option.kind == "const":
            if value:  # 开关打开 → 传入口模块里那个常量（例如 PREFLIGHT_OFF）
                kwargs[option.dest] = _default_of(module, option)
            continue
        kwargs[option.dest] = value
    return kwargs


def resolve_kwargs(command: str, argv: Sequence[str]) -> dict[str, Any]:
    """解析 ``argv`` 并返回"会传给入口模块 ``main(**kwargs)``"的关键字。

    给测试与排障用（``tests/test_cli.py`` 就是靠它验证"选项真的变成了配置覆盖"）：
    它不调用 handler，因此不会碰硬件，也不会跑任务。``argv`` 只写该子命令自己的选项
    （子命令名由 ``command`` 给出）。
    """
    spec = SUBCOMMANDS.get(command)
    if spec is None:
        raise KeyError(f"未知子命令：{command!r}（可选 {sorted(SUBCOMMANDS)}）")
    parser = build_parser()
    # 借用完整解析器：只传入该子命令的选项，避免各子命令的解析器被复制一份
    args = parser.parse_args([command, *argv])
    module = importlib.import_module(spec.module)
    return _kwargs_of(spec, module, args)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；返回进程退出码（参数不合法时由 argparse 以 2 退出）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = SUBCOMMANDS[args.command]
    module = importlib.import_module(spec.module)
    handler = getattr(module, spec.handler)
    result = handler(**_kwargs_of(spec, module, args))
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
