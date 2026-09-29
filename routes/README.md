# routes/ — 操作手的 QGroundControl 航线（`.plan`）

本目录存放**在 QGC 里画好、另存为 `.plan`** 的固定翼航线。项目按 `RoutesConfig` 的路径读取它们：

| 文件 | 配置项 | 用途 |
| --- | --- | --- |
| `recon.plan` | `RoutesConfig.recon_plan` | 侦查段：整条航线**原样使用**（缺起飞项时只告警，不自动补） |
| `land.plan` | `RoutesConfig.land_plan` | 飞掠段**之后**的部分（返航 + 降落）：项目把飞掠段插在它**前面**，合成一条任务上传 |

⚠ **一条腿只能二选一**：给了 `.plan` 就不能再配 `recon_route` / `landing_route`（`Config.validated()` 会报错）——让两个来源静默竞争比报错危险得多。

## 保存方式

QGC 里画好航线 → **Save / 另存为** → 选择本目录：

* Windows 版 QGC：`C:\YYY\Python\2026test\routes\`
* WSL 里的 QGC（本地就是这样跑的）：同一目录是 `/mnt/c/YYY/Python/2026test/routes/`

## 对文件内容的要求

飞控（PX4 固定翼）会对上传的任务做**可行性检查**，不合规就**整条任务被拒**，而且飞控只会"没有可用任务，盘旋"。本包在规划阶段会照同一套判据**先预检一遍**（`check_fixed_wing_landing`），不合格直接带着原因失败。

1. `vehicleType=1`（固定翼）、`firmwareType=12`（PX4）；
2. 降落段最后一项是 `NAV_LAND`（`command=21`）；
3. **`NAV_LAND` 的紧前一项必须严格高于落点**，且
   `(前项高 − 落点高) / 水平距离 ≤ tan(FW_LND_ANG + 0.1°)`
   （`FW_LND_ANG` 默认 8° ⇒ 上限约 0.142；改过飞控参数就把 `RoutesConfig.fw_land_angle_deg` 同步）。
   ⚠ 下面例子里的数字是**斜率 tanθ，不是角度**：**20 m 高 / 距落点 300 m** ⇒ tan≈0.067（约 3.8°）✓；
   **40 m 高 / 距落点 183 m** ⇒ tan≈0.219（约 12.4°）✗ 被判"下滑角过陡"；
4. 进场点只能是普通航点或 QGC"固定翼降落航线"复杂项里的绕圈下降到高度
   （`fwLandingPattern`：`DO_LAND_START + NAV_LOITER_TO_ALT + NAV_LAND`，本包会照 QGC 的逻辑展开）；
5. 其它 QGC 复杂项（VTOL 降落、测绘/结构航线）**不支持**，请改存成普通航点。

## 已验证

2026-09 SITL（PX4 1.17 + `gz_rc_cessna`，`routes/land.plan` 即本目录中的那份）：

```
状态轨迹: RECON → HOLD_PROCESS → OVERFLY → LAND → DONE
上传/投放: 2 / 1   判据评估 408 次    ABORT/错误: 0 / 0
投放触发时预测落点误差 1.35 m（高度 20 m、飞行时间 2.15 s）
```

## 对准 CUADC 赛区

`land.plan` 的经纬度已对准 `sim/worlds/cuadc/cuadc_recon_strike_r*.sdf` 的起降区
（ENU 原点 = 跑道中点；世界原点与 PX4 默认世界一致，见 `docs/simulation_world.md`）：

* 前两个航点：跑道中心线以西 650m / 500m、50m 高（返航进入段）；
* 降落复杂项：进场点在西侧 340m、30m 高，落点在跑道西段（x = -40m），下滑斜率
  tan≈0.103 ✓（PX4 上限 ≈0.142）；
* 换场地时这两处要一起改：世界的 `spherical_coordinates` + 本文件的经纬度 +
  `examples/sitl_mission.py` 的航点常量。
