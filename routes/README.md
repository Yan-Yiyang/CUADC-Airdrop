# 操作手航线

本目录用于保存操作手在 QGroundControl 中创建并导出的 `.plan` 文件。航线属于本地任务输入，Git 默认忽略，不随仓库分发；克隆后需要自行准备。

| 本地文件约定 | 配置项 | 用途 |
| --- | --- | --- |
| `recon.plan` | `RoutesConfig.recon_plan` | 侦查任务；缺少起飞项时只告警，不自动添加 |
| `land.plan` | `RoutesConfig.land_plan` | 飞掠后的返航和降落段 |

在 QGC 中另存到仓库的 `routes/` 目录，或将配置指向其它本地文件。任务任一阶段只选择 `.plan` 或配置航点中的一种来源，不要同时设置。

## 降落约束

固定翼任务会被 PX4 做可行性检查。程序规划时也执行 `check_fixed_wing_landing`，预检配置必须与实际机型参数保持一致。

- QGC 任务的 `vehicleType=1`、`firmwareType=12`。
- 降落段以 `NAV_LAND` 结束，紧前进场点必须高于落点。
- 下滑角不能超过所配置的 `FW_LND_ANG` 容限；高度差除以水平距离是斜率，不是角度。
- 支持普通任务项和 QGC `fwLandingPattern`；VTOL 降落、测绘等其它复杂项会明确拒绝。

换场地时同步核对仿真 manifest 或实际场地原点、QGC 航点经纬度和高程参考。解析回归使用测试内生成的合成任务，不依赖本目录的私人航线。
