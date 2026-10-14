"""分子工坊 · 启动入口。

用法：
    python run.py

依赖（本机已安装，无需联网）：
    PySide6 6.12+、numpy 2.x
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 保证从任何工作目录启动都能找到包
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError:
        print("缺少 PySide6。请安装：python -m pip install PySide6 numpy", file=sys.stderr)
        return 1

    # 高 DPI 屏：按系统缩放渲染，避免字糊
    if hasattr(QtGui.QGuiApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("分子工坊")
    app.setApplicationDisplayName("分子工坊 · 化学结构搭建沙盒")

    from molecule_workshop.ui.theme import app_stylesheet, load_fonts, preferred_family
    family = load_fonts()
    app.setStyleSheet(app_stylesheet(preferred_family()))

    from molecule_workshop.ui.main_window import MainWindow
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
