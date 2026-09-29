#!/usr/bin/env bash
# 一键起 PX4 SITL + 本项目的 CUADC 赛区世界。
#
#   bash sim/run_sitl.sh [r1|r2] [机型]
#   HEADLESS=1 bash sim/run_sitl.sh r2          # 只起 server（无 GUI）
#   bash sim/run_sitl.sh r2 gz_rc_cessna        # 换无相机机身
#
# 机型默认 gz_rc_cessna_down_cam（带下视相机 1280x720@30Hz，感知取图的机型；
# 机型文件 = sim/vehicles/rc_cessna_down_cam + airframe 4007_gz_rc_cessna_down_cam，
# 都在本仓库里，由 sim/install_px4.sh 软链进 PX4）。
# 脚本先跑 install_px4.sh（幂等：机型/世界/airframe 挂进 PX4 树），再导出
# GZ_SIM_RESOURCE_PATH（世界里的网格/贴图靠它解析），最后 PX4_GZ_WORLD=... make px4_sitl。
# 世界的布局与规则出处见 docs/simulation_world.md，模块说明见 sim/README.md。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUND="${1:-r2}"
MODEL="${2:-gz_rc_cessna_down_cam}"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WORLD="cuadc_recon_strike_${ROUND}"

if [ ! -f "$HERE/worlds/cuadc/$WORLD.sdf" ]; then
  echo "找不到 $HERE/worlds/cuadc/$WORLD.sdf——先生成：python -m airdrop.run make-world" >&2
  exit 1
fi

bash "$HERE/install_px4.sh"
export GZ_SIM_RESOURCE_PATH="$HERE/worlds/cuadc${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

echo ">>> 世界: $WORLD"
echo ">>> 资源路径: GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
echo ">>> 启动: PX4_GZ_WORLD=$WORLD make px4_sitl $MODEL"
cd "$PX4_DIR"
PX4_GZ_WORLD="$WORLD" make px4_sitl "$MODEL"
