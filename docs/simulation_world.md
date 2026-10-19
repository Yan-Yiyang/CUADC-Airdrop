# 仿真工具与任务演练

世界、相机、机型生成及资产已迁入 [airdrop-sim](../vendor/airdrop-sim/README.md)；场地几何、真值与生成参数见工具仓 [世界专题](../vendor/airdrop-sim/docs/simulation_world.md)。

主仓保留 `python -m airdrop.run mission --mode sitl --manifest <路径>`，驱动实际任务与感知链路。初始化流程见 [工具仓库](tool_repositories.md)，manifest 格式见 [数据契约](tool_contracts.md)。

WSL 中的启动脚本位于 `vendor/airdrop-sim/airdrop_sim/assets/run_sitl.sh`，要求已安装的 Gazebo 和已有 PX4 SITL 构建；脚本只准备独立运行目录，不修改 PX4 源码、不安装依赖或自动构建。先按工具仓 README 配置并启动仿真，再运行主仓演练。离线测试不代表 SITL 或真实设备验收。

## 任务同步与验收边界

SIH/QGC 已验证只读任务同步：任务项数从 0 到 3、再到 4；同为 4 项而坐标改变时任务 ID 更新、旧进度清空，随后回读 4 项。专用 MAVLink 断连时 `connected=False` 且 broker 数据清空，恢复后新会话再次回读 4 项。操作手在 QGC 中执行任务上传、替换与启动；只读观察器未上传、启动或清空飞控任务。任务进度已从 1/24 到 24/24，并完成仿真落地。

正式 Gazebo 场景另已验证相机→主程序被动采集/录制→合成视频实际解码，包含 PX4 遥测，飞控保持未解锁。该次使用隔离服务器配置的软件 H.264 编码与 EGL 无窗口渲染，保留原相机尺寸和场景资产；默认 CUDA 编码路径的资源问题仍待解决。EGL 不保证硬件渲染，显卡可见也不等于 Ogre 使用了显卡。此项关闭识别和任务控制，完整感知/飞行/投放闭环及真实设备仍未验收。资源监测应覆盖 WSL 所在硬盘的持续占用、主机可用内存与仿真进程组，区分启动峰值和持续繁忙；只停止本次仿真实例。

更换机型前后应回读 `MIS_TKO_LAND_REQ`、`FW_LND_ANG`、`FW_LND_USETER`。降落预检 `check_fixed_wing_landing` 必须使用实际机型的进场角；离线默认角为 8°，已核对的 SIH 配置为 5°。只有隔离 SITL 配置且任务没有降落航线时，才将该配置的 `MIS_TKO_LAND_REQ` 设为 `0`；不要把此设置用于真机，也不要因此关闭其他 failsafe。
