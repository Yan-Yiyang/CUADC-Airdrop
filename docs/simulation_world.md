# 仿真工具与任务演练

世界、相机、机型生成及资产已迁入 [airdrop-sim](../vendor/airdrop-sim/README.md)；场地几何、真值与生成参数见工具仓 [世界专题](../vendor/airdrop-sim/docs/simulation_world.md)。

主仓保留 `python -m airdrop.run sitl`、`python -m airdrop.run sitl-recon --manifest <路径>`，驱动实际任务与感知链路。初始化流程见 [工具仓库](tool_repositories.md)，manifest 格式见 [数据契约](tool_contracts.md)。

WSL 中的启动脚本位于 `vendor/airdrop-sim/airdrop_sim/assets/run_sitl.sh`，要求已安装的 Gazebo 和已有 PX4 SITL 构建；脚本只准备独立运行目录，不修改 PX4 源码、不安装依赖或自动构建。先按工具仓 README 配置并启动仿真，再运行主仓演练。离线测试不代表 SITL 或真实设备验收。
