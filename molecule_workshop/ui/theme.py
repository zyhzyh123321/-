"""界面主题：深色科学风格与统一字体。

字体必须显式加载：
    离屏平台下 Qt 找不到系统字体目录（Qt 6 不再自带字体），
    不加载的话所有中文都会变成方块 —— 这一点已实测确认。
"""
from __future__ import annotations

import os

from PySide6 import QtGui, QtWidgets

# 中文字体候选（Windows 自带）
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
    "C:/Windows/Fonts/segoeui.ttf",
]

_loaded_families: list[str] = []


def load_fonts() -> str | None:
    """加载系统中文字体，返回首选字族名。"""
    global _loaded_families
    for path in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        fid = QtGui.QFontDatabase.addApplicationFont(path)
        if fid < 0:
            continue
        for fam in QtGui.QFontDatabase.applicationFontFamilies(fid):
            if fam not in _loaded_families:
                _loaded_families.append(fam)
    for want in ("Microsoft YaHei", "SimHei", "SimSun", "Segoe UI"):
        if want in _loaded_families:
            return want
    return _loaded_families[0] if _loaded_families else None


def preferred_family() -> str:
    for want in ("Microsoft YaHei", "SimHei", "SimSun", "Segoe UI"):
        if want in _loaded_families:
            return want
    return _loaded_families[0] if _loaded_families else "Sans Serif"


# ----------------------------------------------------------------------------
# 配色
# ----------------------------------------------------------------------------

BG_APP = "#080c11"
BG_PANEL = "#111922"
BG_PANEL_SOFT = "#0e1621"
ACCENT = "#45e0d0"
ACCENT_DIM = "#2a8f88"
WARN = "#ffb340"
DANGER = "#ff5f6d"
TEXT = "#e6f0f5"
TEXT_DIM = "#8fa3b0"
TEXT_FAINT = "#5d7080"
BORDER = "rgba(90, 200, 220, 46)"


def app_stylesheet(family: str) -> str:
    return f"""
QWidget {{
    background: {BG_APP};
    color: {TEXT};
    font-family: "{family}";
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {BG_APP}; }}

QFrame#panel, QWidget#panel {{
    background: {BG_PANEL};
    border: 1px solid rgba(90, 200, 220, 46);
    border-radius: 10px;
}}
QFrame#panelSoft, QWidget#panelSoft {{
    background: {BG_PANEL_SOFT};
    border: 1px solid rgba(90, 200, 220, 30);
    border-radius: 8px;
}}

QLabel#sectionLabel {{
    color: {TEXT_FAINT};
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
}}
QLabel#titleLabel {{ font-size: 20px; font-weight: bold; color: {TEXT}; }}
QLabel#subtitleLabel {{ font-size: 11px; color: {TEXT_FAINT}; }}
QLabel#bigValue {{ font-size: 15px; font-weight: bold; color: {TEXT}; }}
QLabel#caption {{ font-size: 10px; color: {TEXT_FAINT}; }}
QLabel#bodyText {{ font-size: 12px; color: #c3d4de; }}
QLabel#dimText {{ font-size: 12px; color: {TEXT_DIM}; }}

QPushButton#elemButton {{
    text-align: left;
    padding: 7px 10px;
    border-radius: 8px;
    border: 1px solid rgba(120, 170, 190, 40);
    background: rgba(255, 255, 255, 8);
    color: {TEXT};
    font-size: 13px;
}}
QPushButton#elemButton:hover {{ background: rgba(69, 224, 208, 32); }}
QPushButton#elemButton:checked {{
    border: 1px solid rgba(69, 224, 208, 150);
    background: rgba(69, 224, 208, 40);
}}

QPushButton#toolButton {{
    text-align: left;
    padding: 7px 10px;
    border-radius: 8px;
    border: 1px solid rgba(120, 170, 190, 30);
    background: rgba(255, 255, 255, 6);
    color: {TEXT_DIM};
}}
QPushButton#toolButton:hover {{ background: rgba(69, 224, 208, 22); color: {TEXT}; }}
QPushButton#toolButton:checked {{
    border: 1px solid rgba(69, 224, 208, 150);
    background: rgba(69, 224, 208, 40);
    color: {TEXT};
    font-weight: bold;
}}

QPushButton {{
    padding: 7px 12px;
    border-radius: 8px;
    border: 1px solid rgba(120, 170, 190, 50);
    background: rgba(255, 255, 255, 8);
    color: {TEXT};
}}
QPushButton:hover {{ background: rgba(69, 224, 208, 28); border-color: rgba(69, 224, 208, 120); }}
QPushButton:pressed {{ background: rgba(69, 224, 208, 55); }}
QPushButton#primaryButton {{
    background: rgba(69, 224, 208, 55);
    border: 1px solid rgba(69, 224, 208, 160);
    font-weight: bold;
}}
QPushButton#primaryButton:hover {{ background: rgba(69, 224, 208, 85); }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: rgba(255, 255, 255, 4); }}

QCheckBox {{ color: {TEXT_DIM}; spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid rgba(120, 170, 190, 70);
    background: rgba(255, 255, 255, 8);
}}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 9px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: rgba(120, 170, 190, 60); border-radius: 4px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: rgba(69, 224, 208, 110); }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {BG_PANEL_SOFT}; color: {TEXT_DIM}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{
    background: {BG_PANEL}; color: {TEXT};
    border: 1px solid rgba(69, 224, 208, 90); padding: 6px; border-radius: 6px;
}}
QProgressBar {{
    border: none; border-radius: 3px; background: rgba(255, 255, 255, 16);
    height: 7px; text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""
