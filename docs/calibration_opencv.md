# 相机标定

离线三步标定位于 [airdrop-calibration](../vendor/airdrop-calibration/README.md)，算法与 OpenCV 说明见工具仓的 [专题](../vendor/airdrop-calibration/docs/calibration_opencv.md)。

主仓保留 `python -m airdrop.run mission --mode real --no-mission-control --no-recognize --record`，录制标准飞行目录。独立工具读取目录并输出 `camera_calib.json`，生产相机读取器消费该文件。字段、时间轴和坐标约定见 [数据契约](tool_contracts.md)，初始化与独立运行见 [工具仓库](tool_repositories.md)。
