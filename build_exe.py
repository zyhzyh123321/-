"""把《分子工坊》打包成单文件 exe（PyInstaller）。

用法：
    python build_exe.py             # 检查环境并打包
    python build_exe.py --dry-run   # 只检查环境、打印将要执行的命令，不真的打包

说明：
    * 入口是 run.py，打包参数为 --onefile --windowed（单文件、不弹控制台窗口）。
    * 打包需要联网安装 pyinstaller；本机未安装时脚本会打印安装命令并退出，
      不会硬着头皮往下跑。
    * 产物输出到项目根目录的 dist/ 下。
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "run.py"
DIST = ROOT / "dist"

# exe 的名字。如果目标系统对中文文件名不友好，改成 "MoleculeWorkshop" 即可。
APP_NAME = "分子工坊"

INSTALL_HINT = "python -m pip install pyinstaller"


def find_pyinstaller() -> list[str] | None:
    """返回可用的 PyInstaller 调用前缀；两者都没有时返回 None。"""
    # 优先用当前解释器自带的模块，避免 PATH 上的 pyinstaller 属于另一个环境
    if importlib.util.find_spec("PyInstaller") is not None:
        return [sys.executable, "-m", "PyInstaller"]
    exe = shutil.which("pyinstaller")
    if exe:
        return [exe]
    return None


def build_command(pyi: list[str]) -> list[str]:
    """组装 PyInstaller 命令行。"""
    return [
        *pyi,
        "--noconfirm",   # 覆盖 dist/ 里已有的产物，不交互提问
        "--clean",       # 打包前清掉缓存的临时文件
        "--onefile",     # 打成单个 exe
        "--windowed",    # GUI 程序：不弹出黑色控制台窗口
        "--name", APP_NAME,
        str(ENTRY),
    ]


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv

    print("=" * 64)
    print(f"《{APP_NAME}》打包脚本 —— 打包需要联网安装 pyinstaller")
    print("=" * 64)

    pyi = find_pyinstaller()
    if pyi is None:
        print("[X] 没有检测到 PyInstaller。")
        print("    打包需要联网安装 pyinstaller，请先执行：")
        print(f"        {INSTALL_HINT}")
        print("    安装完成后重新运行：python build_exe.py")
        return 1
    print(f"[OK] 已检测到 PyInstaller：{' '.join(pyi)}")

    if not ENTRY.is_file():
        print(f"[X] 找不到入口文件：{ENTRY}")
        print("    请确认 run.py 已经在项目根目录下，再重新打包。")
        return 1
    print(f"[OK] 入口文件：{ENTRY}")

    cmd = build_command(pyi)
    print("[*] 将执行：")
    print("    " + " ".join(cmd))

    if dry_run:
        print("[i] --dry-run：只做检查，不真的打包。")
        return 0

    print("[*] 开始打包……（首次打包通常需要一两分钟）")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT))
    except OSError as exc:
        print(f"[X] 启动 PyInstaller 失败：{exc}")
        return 1

    if proc.returncode != 0:
        print(f"[X] 打包失败（PyInstaller 退出码 {proc.returncode}）。")
        return proc.returncode

    exe = DIST / f"{APP_NAME}.exe"
    print("[OK] 打包完成。")
    print(f"     产物：{exe}")
    print("     双击即可运行；单文件 exe 首次启动需要解包，会稍慢一点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
