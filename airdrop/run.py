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
    "full-mission": Subcommand(
        name="full-mission",
        module="examples.full_mission",
        help="完整任务：自检 → 等起飞 → 侦查 → 飞掠投放 → 降落（正式任务档）",
        options=(
            Option(
                ("--system-address",),
                "system_address",
                "MAVSDK 系统地址（飞控连接）",
                source="SYSTEM_ADDRESS",
            ),
            Option(("--rtsp-url",), "rtsp_url", "HM30 图传 RTSP 地址", source="RTSP_URL"),
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
                "侦查航线谁上传：operator=操作手在 QGC 启动（正式任务）/ auto=本包上传（自动测试）",
                kind="choice",
                choices=("operator", "auto"),
                source="RECON_UPLOAD",
            ),
            Option(
                ("--no-video",),
                "use_video",
                "本架次不接图传（跳过图传与感知，起飞前自检的四项检查也相应关闭）",
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
                "是否等到飞机确实在空中才进侦查（地面演练用 --no-require-airborne）",
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
        help="PX4 SITL 演练：合成目标 + DryRunController（不装弹、不用图传）",
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
                "侦查航线谁上传：auto=本包上传（演练档）/ operator=等 QGC 启动",
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
    "sitl-recon": Subcommand(
        name="sitl-recon",
        module="tools.sitl_recon",
        help="SITL 侦查精度自动测试：真飞机 + 真感知，量「解算坐标 vs 天井中心」误差",
        options=(
            Option(
                ("--manifest",),
                "manifest_path",
                "仿真 manifest：读取本次世界真值与相机参数；省略时使用默认 r2/seed 0",
                kind="path",
                source="MANIFEST_PATH",
            ),
            Option(
                ("--system-address",),
                "system_address",
                "MAVSDK 系统地址（SITL 默认 udpin://0.0.0.0:14540）",
                source="SYSTEM_ADDRESS",
            ),
            Option(
                ("--lag",),
                "lag_s",
                "图传链路延时估计（秒；报告里还会给 lag 扫描）",
                kind="float",
                source="LAG_OVERRIDE_S",
            ),
            Option(
                ("--recon-timeout",),
                "recon_timeout_s",
                "RECON 段的上限（秒，状态机在这一段等侦查航线飞完）",
                kind="float",
                source="RECON_TIMEOUT_S",
            ),
            Option(
                ("--video-wait",),
                "video_wait_s",
                "等第一帧画面的上限（秒）",
                kind="float",
                source="VIDEO_WAIT_S",
            ),
            Option(
                ("--work-dir",),
                "work_dir",
                "产物目录（标定/SDP/检测明细/报告）",
                source="WORK_DIR",
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
                ("--drain-timeout",),
                "drain_timeout_s",
                "停止生产后等待尾帧与 OCR 完成的最长秒数",
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
            Option(
                ("--calib",),
                "calib_file",
                "相机标定文件（要用录这段素材时同一份；SITL 架次是 .sitl-recon-tmp/camera_calib_sim.json）",
                kind="str",
                source="CALIB_FILE",
            ),
            Option(
                ("--land-plan",),
                "land_plan",
                "降落航线 .plan（不给就只打印飞掠段，不规划降落）",
                kind="str",
                source="LAND_PLAN",
            ),
            Option(
                ("--side-check",),
                "side_check_tolerance",
                "边长互校门限（默认关；回放优化时才给，如 0.25）",
                kind="float",
                source="SIDE_CHECK_TOLERANCE",
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
        help="HM30 图传拉流：统计链路质量（可选预览/存盘）",
        options=(
            Option(("--rtsp-url",), "url", "图传 RTSP 地址", source="URL"),
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
            Option(("--rtsp-url",), "video_url", "图传 RTSP 地址", source="VIDEO_URL"),
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
                "图传 RTSP 地址（None = 用默认 HM30 地址）",
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
        module="airdrop_calibration.cli",
        help="三步相机标定：内参 → 画面/遥测时间差 → 手眼外参",
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
        module="airdrop_sim.cli",
        help="生成 CUADC 赛区 Gazebo 世界（round1 图片靶标 / round2 数字靶标）",
    ),
    "make-vehicle": Subcommand(
        name="make-vehicle",
        module="airdrop_sim.cli",
        help="生成 rc_cessna 相机机型、模拟相机文件与仿真 manifest",
    ),
    "make-backdrops": Subcommand(
        name="make-backdrops",
        module="airdrop_sim.cli",
        help="生成场景底图贴图：目标区地面纹理 + 比赛区域外的随机航拍干扰底图",
    ),
    "preview-world": Subcommand(
        name="preview-world",
        module="airdrop_sim.cli",
        help="把赛区世界渲染成俯视预览图（PNG，带中文标注，供人眼核对布局）",
    ),
    "fetch-aerial": Subcommand(
        name="fetch-aerial",
        module="airdrop_sim.cli",
        help="抓真实航拍底图（USGS NAIP，公有领域，需联网）替换程序化底图",
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
            description=f"{spec.help}（入口模块：{spec.module}）",
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
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
