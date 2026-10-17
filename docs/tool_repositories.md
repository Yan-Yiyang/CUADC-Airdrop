# 独立工具仓库

主仓只保留任务、记录采集及跨仓演练。`vendor/airdrop-sim` 包含世界/机型/相机生成和资产，`vendor/airdrop-calibration` 包含离线内参/延时/手眼标定。两者不导入 `airdrop`，主仓生产包也不导入工具内部实现。

## 初始化与本地镜像

父仓以 gitlink 固定提交，`.gitmodules` 的相对 URL 约定两个工具仓与父仓为同级仓库。尚未配置远端；可将本地 URL 指向工具的独立克隆或本地镜像。分发到其它机器前需一并提供两个工具仓（或 Git bundle），或由维护者设置可访问的远端 URL。单独复制父仓不能携带子模块对象。

常规同级远端可直接：

```bash
git clone --recurse-submodules <主仓地址>
# 或在已有父仓内
git submodule update --init --recursive
```

使用本地镜像时，显式配置本地 URL，再初始化；本地文件传输许可只限该次调用：

```bash
git config submodule.vendor/airdrop-sim.url <仿真仓本地路径>
git config submodule.vendor/airdrop-calibration.url <标定仓本地路径>
git -c protocol.file.allow=always submodule update --init --recursive
```

工具未初始化时，`import airdrop`、主 CLI 总帮助与 `mission --help` 可正常使用；工具命令报缺少包和初始化方法，不自动安装。`mission --mode sitl` 为跨仓演练，需要仿真工具。

## 独立运行与依赖

两个工具均要求 Python ≥3.14。独立克隆后按各自 README 使用 uv；仿真核心仅标准库，预览/底图需要 Pillow；标定仅 NumPy/OpenCV，SciPy 只供合成测试。未安装工具时，主仓兼容入口会按需加载固定子模块，无需 editable 安装。

```bash
# 在对应工具仓根目录执行
python -m airdrop_sim --help
python -m airdrop_sim make-world --out-dir output/worlds
python -m airdrop_calibration calibrate --flight <飞行目录> --out <相机文件>
```

主仓原 `make-world`、`make-vehicle`、`make-backdrops`、`preview-world`、`fetch-aerial`、`calibrate` 命令短期保留，仅调用时转发到独立 CLI；Python 调用改用 `airdrop_sim` / `airdrop_calibration`，不再提供 `tools.calibrate` 等旧导入。世界生成默认写调用方的 `simulation-output/`；只读贴图与预览默认世界来自包内资产。安装包包含运行所需资产。

## 验证与升级

三仓分别运行 pytest、ruff、格式与 pyright；主仓另跑 `check-docs`。工具测试不会自动并入主仓；跨仓读写契约由主仓测试验证。WSL/ext4 用例操作临时假 PX4，验证链接和源码不变，不代表真实 SITL 飞行。

升级时先在工具仓验证并提交，再在父仓暂存对应 gitlink 和提交；运行两仓专项及主仓跨仓契约。使用 `git submodule update --init` 复现父仓固定版本，避免 `--remote` 无意跟随最新分支。主仓还以 `tests/test_tool_repositories.py` 验证无子模块启动、按需加载与兼容转发；独立工具自行验证 API 导出、参数与默认值。
