"""``sim/install_px4.sh``（把仿真资产挂进 PX4 树）的回归用例。

⚠ **为什么在 WSL 里测**：这个脚本操作的是 **Linux 上的 PX4 源码树**，软链是它的核心。
Windows 的 Git Bash / MSYS 里 ``ln -s`` 会**退化成复制**（实测 ``link_into`` 建出来的是
实体目录不是软链），于是"软链对不对"这件事根本测不到；而且从 Python 子进程直接调
``bash`` 在这台机器上有三种不同的失败（PATH 上是 WSL 代理桩、反斜杠路径被当转义、
挂载表认不出 ``/c``）。所以这里统一走 ``wsl.exe``，并且让驱动脚本把 ``sim/`` 复制到
WSL 的 ``/tmp``（ext4）上再操作——DrvFs 上软链语义不可靠，PX4 树本身也在 Linux 侧。

没有 WSL 的环境整组跳过（用 ``skipif`` 而不是 marker，所以不受默认 ``-m`` 过滤影响）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "sim" / "install_px4.sh"
SIM_TMP_ROOT = REPO_ROOT / ".sim-script-tmp"

#: PX4 的 airframes/CMakeLists.txt 在我们插入那一行**之前**的样子
CMAKELISTS_ORIGINAL = "px4_add_romfs_files(\n\t4006_gz_px4vision\n)\n"
#: 模拟"PX4 升级、我们那一行被冲掉"之后的样子（自己多了一个条目）
CMAKELISTS_UPGRADED = "px4_add_romfs_files(\n\t4006_gz_px4vision\n\t9999_px4_upgraded\n)\n"

needs_wsl = pytest.mark.skipif(
    shutil.which("wsl") is None, reason="需要 WSL（软链语义只能在 ext4/Linux 上验）"
)

#: 驱动脚本：复制 sim/ 到 WSL 的 /tmp，对一棵假 PX4 树跑各场景，逐条输出 OK/NO。
#: 放在 Python 里是因为断言要集中在一处；**为什么在 WSL 里跑**见模块 docstring。
DRIVER = r"""
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(mktemp -d /tmp/airdrop-simtest.XXXXXX)"
trap 'rm -rf "$BASE"' EXIT
SIM="$BASE/sim"
cp -r "$REPO/sim" "$SIM"

PX4=""; CML=""
ORIGINAL_CML='px4_add_romfs_files(\n\t4006_gz_px4vision\n)\n'
UPGRADED_CML='px4_add_romfs_files(\n\t4006_gz_px4vision\n\t9999_px4_upgraded\n)\n'
new_px4() {
  PX4="$BASE/PX4-Autopilot"; rm -rf "$PX4"
  mkdir -p "$PX4/Tools/simulation/gz/models" "$PX4/Tools/simulation/gz/worlds" \
           "$PX4/ROMFS/px4fmu_common/init.d-posix/airframes"
  CML="$PX4/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt"
  printf "$ORIGINAL_CML" > "$CML"
}
run() { PX4_DIR="$PX4" bash "$SIM/install_px4.sh" "$@" 2>&1; }
ok() { echo "OK $1"; }
no() { echo "NO $1 -- $2"; }
model() { echo "$PX4/Tools/simulation/gz/models/rc_cessna_down_cam"; }

########## 1) 首次安装 + 卸载：软链真建出来、CMakeLists 真还原 ##########
new_px4
out=$(run)
[ -L "$(model)" ] && ok install_creates_symlink || no install_creates_symlink "机型不是软链"
grep -q 4007_gz_rc_cessna_down_cam "$CML" && ok install_registers || no install_registers "没登记"
[ -f "$CML.bak" ] && ok install_backs_up || no install_backs_up "没备份"
out=$(run --uninstall)
[ -e "$(model)" ] || [ -L "$(model)" ] && no uninstall_removes_links "机型软链还在" || ok uninstall_removes_links
if [ "$(cat "$CML")" = "$(printf 'px4_add_romfs_files(\n\t4006_gz_px4vision\n)')" ]; then
  ok uninstall_restores_cmakelists
else
  no uninstall_restores_cmakelists "没还原成 PX4 原样"
fi

########## 2) 幂等重跑：登记不能变两行，软链不能被塞成嵌套 ##########
new_px4
run > /dev/null; run > /dev/null
n=$(grep -c 4007_gz_rc_cessna_down_cam "$CML")
[ "$n" = 1 ] && ok install_is_idempotent || no install_is_idempotent "登记变成 $n 行"
[ -d "$(model)/rc_cessna_down_cam" ] \
  && no install_no_nested_link "出现了嵌套链接（ln 缺 -n）" || ok install_no_nested_link

########## 3) PX4 基线变了之后重装：备份不得停在第一次 ##########
# 要走到"旧备份作废"那条分支，得让 .bak 还在、但当前文件已经和它不同：装一次 →
# 用户/PX4 那边把我们那行冲掉 → 再装一次。（先 --uninstall 的话备份会被 mv 消费掉，
# 根本到不了这条分支——踩过这个坑。）
new_px4
run > /dev/null
printf "$UPGRADED_CML" > "$CML"
out=$(run)
if [ -f "$CML.bak" ] && grep -q 9999_px4_upgraded "$CML.bak"; then
  ok backup_refreshed_after_px4_upgrade
else
  no backup_refreshed_after_px4_upgrade "备份停在第一次了（cp -n 的老行为）"
fi
if [ -f "$CML.bak.stale" ] && ! grep -q 9999_px4_upgraded "$CML.bak.stale"; then
  ok stale_backup_kept
else
  no stale_backup_kept "旧备份没另存成 .stale，或被新内容覆盖了"
fi
run --uninstall > /dev/null
grep -q 9999_px4_upgraded "$CML" \
  && ok uninstall_keeps_px4_own_changes || no uninstall_keeps_px4_own_changes "PX4 自己的条目被回滚了"
grep -q 4007_gz_rc_cessna_down_cam "$CML" \
  && no uninstall_removes_our_entry "我们那行没摘掉" || ok uninstall_removes_our_entry

########## 4) 卸载时遇到实体副本：必须继续，并把 CMakeLists 还原 ##########
# 原来是一串裸 rm -f 配 set -e：只要有一项删不掉（rm -f 没有 -r，删不了目录），
# 脚本立刻退出，**下面的 CMakeLists 还原根本没跑到**，PX4 树留在"装了一半"的状态。
new_px4
run > /dev/null
rm -f "$(model)"; mkdir -p "$(model)"; echo keep > "$(model)/precious.sdf"
out=$(run --uninstall); rc=$?
[ $rc -eq 0 ] && ok uninstall_does_not_abort || no uninstall_does_not_abort "退出码 $rc，中途 abort 了"
[ -f "$(model)/precious.sdf" ] \
  && ok uninstall_keeps_entity_copy || no uninstall_keeps_entity_copy "实体副本被自动删了"
if [ "$(cat "$CML")" = "$(printf 'px4_add_romfs_files(\n\t4006_gz_px4vision\n)')" ]; then
  ok uninstall_restores_even_with_leftovers
else
  no uninstall_restores_even_with_leftovers "有残留时 CMakeLists 没还原"
fi

########## 5) 世界文件缺失：必须报错，且不留垃圾软链 ##########
# 原来没开 nullglob，glob 落空时 bash 原样吐出模式串当文件名，于是 world_count 被
# 算成 1、下面那个"没有世界文件"的检查形同虚设，还会建出一个名字里带 * 的垃圾软链，
# 退出码还是 0。
new_px4
rm -rf "$SIM/worlds"; mkdir -p "$SIM/worlds/cuadc"
out=$(run); rc=$?
[ $rc -ne 0 ] && ok missing_worlds_fails || no missing_worlds_fails "退出码 0，静默装了个不存在的世界"
echo "$out" | grep -q "没有世界文件" \
  && ok missing_worlds_reports_why || no missing_worlds_reports_why "没报出原因：$out"
junk=$(ls "$PX4/Tools/simulation/gz/worlds/" 2>/dev/null | grep -c '\*')
[ "$junk" = 0 ] && ok missing_worlds_leaves_no_junk \
                || no missing_worlds_leaves_no_junk "建了 $junk 个名字里带 * 的垃圾软链"

########## 6) 提示语不得断言代码没做的事 ##########
# 原来 "（已移进本仓库）" 是在断言实体副本的内容在仓库里有第二份，而那一行的代码
# 只是 rm -rf；出事时这句话会误导人。
new_px4
mkdir -p "$(model)"
out=$(run)
echo "$out" | grep -q "已移进本仓库" \
  && no message_matches_action "还在断言『已移进本仓库』" \
  || ok message_matches_action

########## 7) 仓库侧资产缺失：不得删实体副本 ##########
mv "$SIM/vehicles" "$BASE/vehicles.keep"
new_px4
mkdir -p "$(model)"; echo keep > "$(model)/precious.sdf"
out=$(run); rc=$?
[ $rc -ne 0 ] && ok missing_repo_asset_fails || no missing_repo_asset_fails "退出码 0"
[ -f "$(model)/precious.sdf" ] \
  && ok missing_repo_asset_keeps_copy || no missing_repo_asset_keeps_copy "实体副本被删了"
mv "$BASE/vehicles.keep" "$SIM/vehicles"
"""


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉。

    不用 ``tmp_path``：部分受限执行环境里 pytest 的 basetemp 目录建好后不可枚举、
    不可删除（``PermissionError: [WinError 5]``）。放工作区内是为了让 WSL 能通过
    ``/mnt/c/...`` 读到驱动脚本。
    """
    SIM_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = SIM_TMP_ROOT / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _drive_in_wsl(workdir: Path) -> dict[str, str]:
    """在 WSL 里跑驱动脚本，返回 ``{检查名: 结果}``（``"ok"`` 或失败说明）。"""
    stage = workdir / "stage"
    stage.mkdir()
    shutil.copytree(REPO_ROOT / "sim", stage / "sim")
    driver = stage / "driver.sh"
    driver.write_text(DRIVER, encoding="utf-8", newline="\n")

    # 驱动脚本从 /mnt/c 读得到就够；它自己会把 sim/ 复制到 WSL 的 /tmp（ext4）再操作。
    proc = subprocess.run(
        ["wsl", "--exec", "bash", _wsl_path(driver)],
        check=False,  # 退出码自己判（驱动脚本失败要看到它到底报了什么）
        env=dict(os.environ),
        capture_output=True,
        timeout=300,
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    results: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith("OK "):
            results[line[3:].strip()] = "ok"
        elif line.startswith("NO "):
            name, _, why = line[3:].partition(" -- ")
            results[name.strip()] = why.strip() or "失败"
    assert results, (
        f"驱动脚本没输出任何检查结果（rc={proc.returncode}）\nstdout:\n{out}\nstderr:\n{err}"
    )
    return results


def _wsl_path(path: Path) -> str:
    """把 Windows 路径转成 WSL 能吃的 ``/mnt/<盘符>/…`` 形式。

    WSL 的盘符挂载点是 ``/mnt/c``，**不是** MSYS 的 ``/c``（试过 ``/c:/…`` 和
    ``/C:/…``，都是"没有那个文件或目录"）。
    """
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}" + resolved.as_posix()[2:].replace("\\", "/")


@needs_wsl
def test_install_px4_is_syntactically_valid() -> None:
    proc = subprocess.run(
        ["wsl", "--exec", "bash", "-n", _wsl_path(SCRIPT)],
        check=False,
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"bash -n 失败：{proc.stderr.decode('utf-8', 'replace')}"


@needs_wsl
def test_install_px4_lifecycle(workdir: Path) -> None:
    """整条生命周期：装 → 幂等重装 → 升级后重装 → 卸载，逐条核对。"""
    results = _drive_in_wsl(workdir)

    expected = (
        "install_creates_symlink",
        "install_registers",
        "install_backs_up",
        "uninstall_removes_links",
        "uninstall_restores_cmakelists",
        "install_is_idempotent",
        "install_no_nested_link",
        "backup_refreshed_after_px4_upgrade",
        "stale_backup_kept",
        "uninstall_keeps_px4_own_changes",
        "uninstall_removes_our_entry",
        "uninstall_does_not_abort",
        "uninstall_keeps_entity_copy",
        "uninstall_restores_even_with_leftovers",
        "missing_worlds_fails",
        "missing_worlds_reports_why",
        "missing_worlds_leaves_no_junk",
        "message_matches_action",
        "missing_repo_asset_fails",
        "missing_repo_asset_keeps_copy",
    )
    missing = [name for name in expected if name not in results]
    assert not missing, f"驱动脚本没跑到这些检查（脚本可能提前退出了）：{missing}"
    failed = {name: why for name, why in results.items() if why != "ok"}
    assert not failed, "sim/install_px4.sh 行为不对：\n" + "\n".join(
        f"  ✗ {name}：{why}" for name, why in sorted(failed.items())
    )
