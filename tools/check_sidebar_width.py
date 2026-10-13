"""侧栏宽度鲁棒性验证。

背景：侧栏原本写死 226 px，而内容区可用宽正好等于最宽按钮宽（余量 0）。
只要字体变大一点（125% 显示缩放、系统换字体），文字就会顶出按钮、
和相邻控件叠在一起 —— 用户看到的就是"左侧栏被压缩、文字糊成一团"。

现在宽度由字体度量算出来。这个脚本在多种字体倍数下检查：
  1. 文字 + 图标 + 内边距 是否装得进按钮（装不进就是会叠字）
  2. 按钮是否装得进面板内容区
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets        # noqa: E402

app = QtWidgets.QApplication(sys.argv[:1])
from molecule_workshop.ui.theme import load_fonts, app_stylesheet, preferred_family  # noqa: E402
load_fonts()
FAMILY = preferred_family()

#: 样式表里 #elemButton / #toolButton 的 padding: 7px 10px
CSS_PAD_X = 10
#: 图标与文字之间的间距（QPushButton 自己会留）
ICON_GAP = 8


def build(scale: float):
    css = app_stylesheet(FAMILY)
    if scale != 1.0:
        css = re.sub(r"font-size:\s*(\d+)px",
                     lambda m: f"font-size: {round(int(m.group(1)) * scale)}px", css)
    app.setStyleSheet(css)
    from molecule_workshop.ui.side_panel import SidePanel
    panel = SidePanel()
    panel.show()
    for _ in range(4):
        app.processEvents()
    return panel


def check(panel) -> tuple[bool, list[str]]:
    problems: list[str] = []
    inner = panel.width() - 2 * 14
    for btn in panel.findChildren(QtWidgets.QPushButton):
        if not btn.text() or not btn.isVisible():
            continue
        fm = QtGui.QFontMetrics(btn.font())
        text_w = fm.horizontalAdvance(btn.text())
        icon_w = btn.iconSize().width() + ICON_GAP if not btn.icon().isNull() else 0
        need = text_w + icon_w + 2 * CSS_PAD_X
        if need > btn.width():
            problems.append(
                f"{btn.text().strip()[:16]!r} 按钮 {btn.width()} < 需要 {need}")
        if btn.width() > inner:
            problems.append(
                f"{btn.text().strip()[:16]!r} 按钮 {btn.width()} > 内容区 {inner}")
    return (not problems), problems


def main() -> int:
    print(f"{'字体':>6} {'面板宽':>7} {'内容区':>7} {'最挤按钮需要':>12} {'余量':>6}  结果")
    bad = 0
    for scale in (1.0, 1.1, 1.25, 1.35, 1.5, 1.75, 2.0, 2.5):
        panel = build(scale)
        inner = panel.width() - 2 * 14
        tightest = 0
        for btn in panel.findChildren(QtWidgets.QPushButton):
            if not btn.text() or not btn.isVisible():
                continue
            fm = QtGui.QFontMetrics(btn.font())
            icon_w = btn.iconSize().width() + ICON_GAP if not btn.icon().isNull() else 0
            tightest = max(tightest, fm.horizontalAdvance(btn.text()) + icon_w + 2 * CSS_PAD_X)
        ok, problems = check(panel)
        slack = inner - tightest
        flag = "✓" if ok else "✗"
        print(f"  x{scale:<4} {panel.width():>7} {inner:>7} {tightest:>12} {slack:>6}  {flag}")
        for p in problems:
            print(f"        - {p}")
        if not ok:
            bad += 1
        panel.deleteLater()
    print()
    if bad:
        print(f"{bad} 个字体倍数下会叠字")
        return 1
    print("所有字体倍数下文字都装得下 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
