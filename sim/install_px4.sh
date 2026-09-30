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

#: 摘掉一条软链。⚠ 卸载是"尽力而为"：只删**软链**，逐项独立处理。
#: 原来是一串裸 ``rm -f`` 配 ``set -e``：只要有一项删不掉（例如目标是实体目录，
#: ``rm -f`` 没有 -r 删不了），脚本立刻退出，**下面的 CMakeLists 还原根本没跑到**
#: ——PX4 树就被留在"装了一半"的状态。实体副本不自动删：它可能是手工拷进去的。
remove_link() {
  if [ -L "$1" ]; then
    if rm -f "$1"; then
      echo ">>> 已摘掉软链 $1"
    else
      echo ">>> 警告：删不掉 $1，请自行确认后处理（卸载继续）" >&2
    fi
  elif [ -e "$1" ]; then
    echo ">>> 警告：$1 是实体副本而不是软链，不自动删（可能是手工拷进去的）。" >&2
    echo "    请自行确认内容后手工处理。卸载继续。" >&2
  fi
}

if [ "$UNINSTALL" = 1 ]; then
  remove_link "$MODELS_DIR/rc_cessna_down_cam"
  for world in "$HERE"/worlds/cuadc/cuadc_recon_strike_*.sdf; do
    remove_link "$WORLDS_DIR/$(basename "$world")"
  done
  remove_link "$AIRFRAMES_DIR/$AIRFRAME_NAME"
  if [ -f "$BACKUP" ]; then
    mv -f "$BACKUP" "$CMAKELISTS"
    echo ">>> 已从备份还原 $CMAKELISTS"
  else
    echo ">>> 没有备份可还原；若 CMakeLists 里还有 $AIRFRAME_NAME，请手动删掉那一行" >&2
  fi
  echo ">>> 已摘掉机型 / 世界 / airframe 软链"
  exit 0
fi

#: 软链前先把"实体副本"清掉（PX4 树里不该有副本，正常应该是软链或不存在）
link_into() {
  if [ -e "$2" ] && [ ! -L "$2" ]; then
    # ⚠ 这一步是**删除**，所以原来那句"（已移进本仓库）"不能只是措辞——它是在
    # 断言"仓库里有第二份"，而出事时这句话会误导人。断言由两处**别的**代码保证：
    #   机型 / airframe：上面第 43 行的早退已经确认仓库侧存在，删掉是安全的；
    #   世界：``$1`` 是 glob 匹配出来的（下面开了 ``nullglob``），按构造就存在。
    # 两处都保证不了的情形根本走不到这里，所以这里**不再重复检查** ``$1``——
    # 加一个到不了的 ``exit 1`` 只会让人误以为删除有守卫。
    echo ">>> 目标是实体副本，**删除**后改软链：$2（仓库侧有对应资产，删掉不丢东西）"
    rm -rf "$2"
  fi
  ln -sfn "$1" "$2"
}

# 1) 机型 + 2) 世界：软链（世界两轮都挂，run_sitl.sh 用 PX4_GZ_WORLD 选一个）
link_into "$MODEL_SRC" "$MODELS_DIR/rc_cessna_down_cam"
# ⚠ nullglob：不加的话 glob 落空时 bash 会**原样吐出模式串**当文件名，于是
#   world_count 被算成 1、下面那个"没有世界文件"的检查永远不报，脚本还会建出
#   一个名字里带 * 的垃圾软链（实测：退出码 0，还报告"1 个 .sdf 软链进 …"）。
#   开了它，glob 落空就展开成空列表，计数与报错才真的对得上。
world_count=0
shopt -s nullglob
for world in "$HERE"/worlds/cuadc/cuadc_recon_strike_*.sdf; do
  link_into "$world" "$WORLDS_DIR/$(basename "$world")"
  world_count=$((world_count + 1))
done
shopt -u nullglob
if [ "$world_count" = 0 ]; then
  echo "FAIL: sim/worlds/cuadc 里没有世界文件——先生成：python -m airdrop.run make-world" >&2
  exit 1
fi

# 3) airframe：软链 + 在 CMakeLists 的 romfs 列表里登记（幂等，先备份）
link_into "$AIRFRAME_SRC" "$AIRFRAMES_DIR/$AIRFRAME_NAME"
if grep -q "^\s*$AIRFRAME_NAME\s*$" "$CMAKELISTS"; then
  echo ">>> airframes/CMakeLists.txt 已登记 $AIRFRAME_NAME（跳过）"
else
  # ⚠ 备份不能是"第一次的那份"。原来用 ``cp -n``（不覆盖已存在的文件），于是：
  #   第一次装 → 备份 = 当时的 PX4 CMakeLists；
  #   PX4 升级、CMakeLists 变了 → 第二次装时 ``cp -n`` **静默不写**，备份仍是旧的；
  #   ``--uninstall`` 把**第一次**的备份 mv 回去 ⇒ 把中间所有 PX4 自己的改动
  #   一起回滚掉（这正是这个备份要避免的事）。
  # 所以：没有备份就新建；备份与当前文件一致就复用（幂等）；**不一致说明 PX4 的
  # 基线变了**，旧的备份已作废——另存为 .stale 再取一份新的（不删旧的，万一要查）。
  if [ -f "$BACKUP" ]; then
    if cmp -s "$BACKUP" "$CMAKELISTS"; then
      echo ">>> 复用已有备份 $BACKUP（与当前 CMakeLists 一致）"
    else
      stale="$BACKUP.stale"
      rm -f "$stale"
      mv -f "$BACKUP" "$stale"
      cp "$CMAKELISTS" "$BACKUP"
      echo ">>> 警告：PX4 的 airframes/CMakeLists.txt 已经变了，旧的备份已作废。" >&2
      echo "      旧备份另存为 $stale；新备份 = 当前 PX4 版本（--uninstall 将还原到这个）。" >&2
    fi
  else
    cp "$CMAKELISTS" "$BACKUP"
  fi
  sed -i "/4006_gz_px4vision/a\\\t$AIRFRAME_NAME" "$CMAKELISTS"
  echo ">>> 已在 airframes/CMakeLists.txt 登记 $AIRFRAME_NAME（备份：$BACKUP）"
fi

echo ">>> 机型:   $MODELS_DIR/rc_cessna_down_cam -> $MODEL_SRC"
echo ">>> 世界:   $world_count 个 .sdf 软链进 $WORLDS_DIR"
echo ">>> airframe: $AIRFRAMES_DIR/$AIRFRAME_NAME"
echo ">>> 起 SITL: bash $HERE/run_sitl.sh r2"
echo ">>> （手动起时记得：export GZ_SIM_RESOURCE_PATH=$HERE/worlds/cuadc:\$GZ_SIM_RESOURCE_PATH）"
