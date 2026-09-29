#!/usr/bin/env bash
# 把本仓库 sim/ 里的仿真资产挂进 PX4 源码树（幂等）。
#
#   bash sim/install_px4.sh                # 挂进去（默认 PX4_DIR=~/PX4-Autopilot）
#   PX4_DIR=/path/to/PX4-Autopilot bash sim/install_px4.sh
#   bash sim/install_px4.sh --uninstall    # 摘掉（含还原 CMakeLists 的登记）
#
# 挂三样（都用软链，仓库是唯一出处，PX4 树里不再存副本）：
#   1) 机型   sim/vehicles/rc_cessna_down_cam            → $PX4_DIR/Tools/simulation/gz/models/
#   2) 世界   sim/worlds/cuadc/cuadc_recon_strike_r*.sdf → $PX4_DIR/Tools/simulation/gz/worlds/
#   3) airframe sim/airframes/4007_gz_rc_cessna_down_cam → $PX4_DIR/ROMFS/px4fmu_common/
#                                                            init.d-posix/airframes/
#
# ⚠ 唯一会改 PX4 **已跟踪文件**的地方：airframes/CMakeLists.txt 的 px4_add_romfs_files
#   列表必须登记 `4007_gz_rc_cessna_down_cam`（PX4 不做通配）；脚本会幂等插入并先备份
#   成 CMakeLists.txt.bak（--uninstall 用它还原）。PX4 的 `git status` 里因此会多出
#   一个 modified + 几个 untracked 软链，属预期，不要往 PX4 仓库提交。
#
# 装完用 `bash sim/run_sitl.sh r2` 起 SITL；也可以按 sim/README.md 手动起。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
MODELS_DIR="$PX4_DIR/Tools/simulation/gz/models"
WORLDS_DIR="$PX4_DIR/Tools/simulation/gz/worlds"
AIRFRAMES_DIR="$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes"
AIRFRAME_NAME="4007_gz_rc_cessna_down_cam"
AIRFRAME_SRC="$HERE/airframes/$AIRFRAME_NAME"
MODEL_SRC="$HERE/vehicles/rc_cessna_down_cam"
CMAKELISTS="$AIRFRAMES_DIR/CMakeLists.txt"
BACKUP="$CMAKELISTS.bak"
UNINSTALL=0
if [ "${1:-}" = "--uninstall" ]; then
  UNINSTALL=1
fi

for directory in "$MODELS_DIR" "$WORLDS_DIR" "$AIRFRAMES_DIR"; do
  if [ ! -d "$directory" ]; then
    echo "FAIL: 找不到 $directory —— PX4_DIR 指对了吗？（当前 PX4_DIR=$PX4_DIR）" >&2
    exit 1
  fi
done
if [ ! -f "$AIRFRAME_SRC" ] || [ ! -d "$MODEL_SRC" ]; then
  echo "FAIL: sim/airframes 或 sim/vehicles 不完整（仓库被裁剪过？）" >&2
  exit 1
fi

if [ "$UNINSTALL" = 1 ]; then
  rm -f "$MODELS_DIR/rc_cessna_down_cam" "$AIRFRAMES_DIR/$AIRFRAME_NAME"
  for world in "$HERE"/worlds/cuadc/cuadc_recon_strike_*.sdf; do
    rm -f "$WORLDS_DIR/$(basename "$world")"
  done
  if [ -f "$BACKUP" ]; then
    mv -f "$BACKUP" "$CMAKELISTS"
    echo ">>> 已从备份还原 $CMAKELISTS"
  else
    echo ">>> 没有备份可还原；若 CMakeLists 里还有 $AIRFRAME_NAME，请手动删掉那一行"
  fi
  echo ">>> 已摘掉机型 / 世界 / airframe 软链"
  exit 0
fi

#: 软链前先把"实体副本"清掉（早些时候手工拷进 PX4 的那一份要换成指向仓库的软链；
#: 目录/文件都处理，非软链才删——软链直接覆盖）
link_into() {
  if [ -e "$2" ] && [ ! -L "$2" ]; then
    echo ">>> 目标是实体副本，先移除（已移进本仓库）：$2"
    rm -rf "$2"
  fi
  ln -sfn "$1" "$2"
}

# 1) 机型 + 2) 世界：软链（世界两轮都挂，run_sitl.sh 用 PX4_GZ_WORLD 选一个）
link_into "$MODEL_SRC" "$MODELS_DIR/rc_cessna_down_cam"
world_count=0
for world in "$HERE"/worlds/cuadc/cuadc_recon_strike_*.sdf; do
  link_into "$world" "$WORLDS_DIR/$(basename "$world")"
  world_count=$((world_count + 1))
done
if [ "$world_count" = 0 ]; then
  echo "FAIL: sim/worlds/cuadc 里没有世界文件——先生成：python -m airdrop.run make-world" >&2
  exit 1
fi

# 3) airframe：软链 + 在 CMakeLists 的 romfs 列表里登记（幂等，先备份）
link_into "$AIRFRAME_SRC" "$AIRFRAMES_DIR/$AIRFRAME_NAME"
if grep -q "^\s*$AIRFRAME_NAME\s*$" "$CMAKELISTS"; then
  echo ">>> airframes/CMakeLists.txt 已登记 $AIRFRAME_NAME（跳过）"
else
  cp -n "$CMAKELISTS" "$BACKUP"
  sed -i "/4006_gz_px4vision/a\\\t$AIRFRAME_NAME" "$CMAKELISTS"
  echo ">>> 已在 airframes/CMakeLists.txt 登记 $AIRFRAME_NAME（备份：$BACKUP）"
fi

echo ">>> 机型:   $MODELS_DIR/rc_cessna_down_cam -> $MODEL_SRC"
echo ">>> 世界:   $world_count 个 .sdf 软链进 $WORLDS_DIR"
echo ">>> airframe: $AIRFRAMES_DIR/$AIRFRAME_NAME"
echo ">>> 起 SITL: bash $HERE/run_sitl.sh r2"
echo ">>> （手动起时记得：export GZ_SIM_RESOURCE_PATH=$HERE/worlds/cuadc:\$GZ_SIM_RESOURCE_PATH）"
