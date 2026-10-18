"""主仓命令行入口：按需加载所选命令，默认值读取相应库模块常量。

总帮助不导入入口模块。可选工具旧命令由独立工具 CLI 解析；缺失子模块
会返回初始化提示，不自动安装。主仓新命令在 SUBCOMMANDS 注册。
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
    metavar: str | None = None
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


#: 旧工具命令仅在调用时转发，参数定义由独立 CLI 维护。
TOOL_COMMANDS = {
    "calibrate": "airdrop_calibration",
    "make-world": "airdrop_sim",
    "make-vehicle": "airdrop_sim",
    "make-backdrops": "airdrop_sim",
    "fetch-aerial": "airdrop_sim",
    "preview-world": "airdrop_sim",
}


#: 子命令注册表（唯一的命令行出处；顺序即 ``--help`` 里的显示顺序）
SUBCOMMANDS: dict[str, Subcommand] = {
    "mission": Subcommand(
        name="mission",
        module="airdrop.runtime.mission",
        help="统一任务入口：真实飞行或 PX4 SITL，按需启用采集、识别和任务阶段",
        options=(
            Option(
                ("--mode",),
                "mode",
                "运行模式",
                kind="choice",
                choices=("real", "sitl"),
                source="MODE",
            ),
            Option(
                ("--mission-control",),
                "run_mission",
                "运行任务状态机；关闭后仅被动采集",
                kind="bool",
                source="RUN_MISSION",
            ),
            Option(("--capture",), "capture", "接入视频采集", kind="bool", source="CAPTURE"),
            Option(
                ("--recognize",),
                "recognize",
                "启用检测、识别与坐标解算",
                kind="bool",
                source="RECOGNIZE",
            ),
            Option(("--record",), "record", "写入标准飞行记录目录", kind="bool", source="RECORD"),
            Option(
                ("--drop-stage",),
                "drop_stage",
                "运行投放判据与飞掠阶段",
                kind="bool",
                source="DROP_STAGE",
            ),
            Option(
                ("--real-release",),
                "real_release",
                "允许真实挂架投放（仅 real 模式）",
                kind="bool",
                source="REAL_RELEASE",
            ),
            Option(("--auto-recon",), "auto_recon", "自动上传并启动侦查航线", kind="bool"),
            Option(
                ("--auto-land",),
                "auto_land",
                "投放后自动执行降落段",
                kind="bool",
                source="AUTO_LAND",
            ),
            Option(
                ("--max-seconds",),
                "max_seconds",
                "被动采集运行上限；省略时按 Ctrl+C 停止",
                kind="float",
            ),
            Option(("--system-address",), "system_address", "MAVSDK 系统地址"),
            Option(("--rtsp-url",), "rtsp_url", "视频源地址"),
            Option(("--qgc-video",), "qgc_video", "向 QGroundControl 发布实时视频", kind="bool"),
            Option(("--qgc-host",), "qgc_host", "QGC 视频接收主机"),
            Option(("--qgc-port",), "qgc_port", "QGC 视频接收端口", kind="int"),
            Option(
                ("--qgc-transport",),
                "qgc_transport",
                "QGC 视频传输格式",
                kind="choice",
                choices=("rtp", "mpegts"),
            ),
            Option(("--qgc-fps",), "qgc_fps", "QGC 视频帧率", kind="float"),
            Option(("--qgc-bitrate-kbps",), "qgc_bitrate_kbps", "QGC 视频码率", kind="int"),
            Option(("--calib",), "calib_file", "相机标定 JSON 文件"),
            Option(
                ("--recon-plan",),
                "recon_plan",
                "QGC 侦查任务计划；自动侦查上传时必需",
            ),
            Option(("--land-plan",), "land_plan", "QGC 降落计划；自动降落时必需"),
            Option(
                ("--backup-point",),
                "backup_point",
                "备用点 WGS84 纬度,经度,相对高度（米）",
                kind="vec3",
                metavar="LAT,LON,ALT",
            ),
            Option(("--manifest",), "manifest_path", "SITL 场景 manifest", kind="path"),
            Option(("--work-dir",), "work_dir", "SITL 标定、SDP 与报告目录"),
            Option(
                ("--lag",),
                "lag_s",
                "SITL 帧遥测延时覆盖（秒）",
                kind="float",
                source="LAG_OVERRIDE_S",
            ),
        ),
    ),
    "replay": Subcommand(
        name="replay",
        module="airdrop.runtime.replay",
        help="离线回放飞行目录并复用识别与坐标处理链路",
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
            Option(("--strict",), "strict", "回放异常时以失败结束", kind="bool", source="STRICT"),
            Option(
                ("--drain-timeout",),
                "drain_timeout_s",
                "尾帧与 OCR 排空最长等待秒数",
                kind="float",
                source="DRAIN_TIMEOUT_S",
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
                "感知模式",
                kind="choice",
                choices=("ocr", "cls12"),
                source="PERCEPTION_MODE",
            ),
            Option(
                ("--selection-rule",),
                "selection_rule",
                "跨候选类选唯一结果的规则",
                kind="choice",
                choices=("median", "max"),
                source="SELECTION_RULE",
            ),
            Option(("--calib",), "calib_file", "相机标定文件", source="CALIB_FILE"),
            Option(("--qgc-video",), "qgc_video", "向 QGroundControl 发布回放视频", kind="bool"),
            Option(("--qgc-host",), "qgc_host", "QGC 视频接收主机"),
            Option(("--qgc-port",), "qgc_port", "QGC 视频接收端口", kind="int"),
            Option(
                ("--qgc-transport",),
                "qgc_transport",
                "QGC 视频传输格式",
                kind="choice",
                choices=("rtp", "mpegts"),
            ),
            Option(("--qgc-fps",), "qgc_fps", "QGC 视频帧率", kind="float"),
            Option(("--qgc-bitrate-kbps",), "qgc_bitrate_kbps", "QGC 视频码率", kind="int"),
            Option(("--land-plan",), "land_plan", "离线规划使用的降落航线", source="LAND_PLAN"),
            Option(
                ("--side-check",),
                "side_check_tolerance",
                "边长互校门限",
                kind="float",
                source="SIDE_CHECK_TOLERANCE",
            ),
        ),
    ),
    "calibrate": Subcommand(
        name="calibrate",
        module="airdrop_calibration.cli",
        help="三步相机标定：内参、时间差与手眼外参",
    ),
    "fit-ballistics": Subcommand(
        name="fit-ballistics",
        module="tools.fit_ballistics",
        help="投放试验弹道参数拟合",
        options=(
            Option(("--drops",), "drops", "飞行目录或 drops.jsonl", kind="path", source="DROPS"),
            Option(("--impacts",), "impacts", "实测落点文件", kind="path", source="IMPACTS"),
            Option(("--out",), "output_path", "拟合报告路径", kind="path", source="OUTPUT_PATH"),
            Option(
                ("--mass-kg",),
                "mass_kg",
                "实测弹体质量（kg）",
                kind="float",
                source="MEASURED_MASS_KG",
            ),
        ),
    ),
    "fetch-models": Subcommand(
        name="fetch-models",
        module="tools.fetch_models",
        help="把外部模型权重复制进 models/（只复制、不联网）",
    ),
    "make-world": Subcommand(name="make-world", module="airdrop_sim.cli", help="生成仿真世界"),
    "make-vehicle": Subcommand(
        name="make-vehicle", module="airdrop_sim.cli", help="生成仿真机型与 manifest"
    ),
    "make-backdrops": Subcommand(
        name="make-backdrops", module="airdrop_sim.cli", help="生成仿真场景底图"
    ),
    "preview-world": Subcommand(
        name="preview-world", module="airdrop_sim.cli", help="预览仿真场景"
    ),
    "fetch-aerial": Subcommand(
        name="fetch-aerial", module="airdrop_sim.cli", help="抓取真实航拍底图"
    ),
    "dump-api": Subcommand(name="dump-api", module="tools.dump_api", help="导出 API 接口清单"),
    "check-docs": Subcommand(
        name="check-docs", module="tools.check_docs", help="核验手册与代码一致性"
    ),
    "qgc-video-test": Subcommand(
        name="qgc-video-test",
        module="tools.qgc_video_test",
        help="合成 QGC 视频测试画面并发布，不连接飞控",
        options=(
            Option(("--host",), "host", "视频接收主机", source="HOST"),
            Option(("--port",), "port", "视频接收端口", kind="int", source="PORT"),
            Option(
                ("--transport",),
                "transport",
                "视频传输格式",
                kind="choice",
                choices=("rtp", "mpegts"),
                source="TRANSPORT",
            ),
            Option(("--fps",), "fps", "合成帧率", kind="float", source="FPS"),
            Option(("--seconds",), "seconds", "测试时长", kind="float", source="SECONDS"),
            Option(("--preview",), "preview_path", "保存合成画面预览", kind="path"),
            Option(("--ffmpeg",), "ffmpeg_executable", "FFmpeg 可执行文件"),
        ),
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
    help_text = option.help if default is None else f"{option.help}（默认：{default!r}）"
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
        parser.add_argument(*option.flags, type=_vec3, metavar=option.metavar or "A,B,C", **common)
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


class _CommandParser(argparse.ArgumentParser):
    """保留 build_parser().parse_args() 用法，解析时才读取所选命令。"""

    prepared_command: str | None = None

    def parse_args(self, args=None, namespace=None):
        arguments = list(sys.argv[1:] if args is None else args)
        if self.prepared_command is None and arguments and arguments[0] in SUBCOMMANDS:
            return build_parser(arguments[0]).parse_args(arguments, namespace)
        return super().parse_args(arguments, namespace)


def build_parser(command: str | None = None) -> argparse.ArgumentParser:
    """构建命令列表，仅为所选主仓命令读取模块默认值。

    入口模块都是"纯库模块"（重依赖在函数体内），所以建解析器这一步不加载
    cv2 / mavsdk / torch——``--help`` 因此是廉价的。
    """
    if command in TOOL_COMMANDS:
        from tools.optional_tools import load_tool_module

        return load_tool_module(SUBCOMMANDS[command].module).build_parser()
    parser = _CommandParser(
        prog="python -m airdrop.run",
        description="CUADC固定翼无人机侦查与打击控制项目：集中式入口（子命令 + 选项，参数到配置走关键字覆盖）",
        epilog="子命令的 --help 给出该入口的全部选项与默认值来源。",
    )
    parser.prepared_command = command
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<子命令>")
    for spec in SUBCOMMANDS.values():
        sub = subparsers.add_parser(
            spec.name,
            help=spec.help,
            description=spec.help,
        )
        sub.set_defaults(command=spec.name)
        if command == spec.name and spec.name not in TOOL_COMMANDS:
            module = importlib.import_module(spec.module)
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
    if command in TOOL_COMMANDS:
        from tools.optional_tools import load_tool_module

        cli = load_tool_module(spec.module)
        return cli.resolve_kwargs(command, argv)
    parser = build_parser(command)
    # 借用完整解析器：只传入该子命令的选项，避免各子命令的解析器被复制一份
    args = parser.parse_args([command, *argv])
    module = importlib.import_module(spec.module)
    return _kwargs_of(spec, module, args)


def main(argv: Sequence[str] | None = None) -> int:
    """仅加载被选择的入口；工具旧命令显式转发至独立 CLI。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else None
    if command in TOOL_COMMANDS:
        from tools.optional_tools import load_tool_module

        try:
            cli = load_tool_module(SUBCOMMANDS[command].module)
            return cli.main(arguments)
        except ModuleNotFoundError as exc:
            print(f"工具无法加载：{exc}", file=sys.stderr)
            return 2
    try:
        parser = build_parser(command)
        args = parser.parse_args(arguments)
        spec = SUBCOMMANDS[args.command]
        module = importlib.import_module(spec.module)
        handler = getattr(module, spec.handler)
        result = handler(**_kwargs_of(spec, module, args))
    except ModuleNotFoundError as exc:
        print(f"入口无法加载：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        parser.error(str(exc))
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
