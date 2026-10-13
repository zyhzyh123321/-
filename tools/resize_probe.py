"""复现「调整窗口时左侧栏文字糊成一团」。

做法：把窗口在多个尺寸之间来回缩放（模拟玩家拖边框），
每一步都渲染一张图，然后和自己「一次到位」渲染的结果逐像素比较。
如果来回缩放后出现差异，就说明是重绘残留（stale paint），
而不是布局算错 —— 两者要修的地方完全不同。

同时精确量出按钮的内边距（chrome），用来校准内容宽度估算。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtGui, QtWidgets        # noqa: E402

app = QtWidgets.QApplication(sys.argv[:1])
from molecule_workshop.ui.theme import load_fonts, app_stylesheet, preferred_family  # noqa: E402
load_fonts()
app.setStyleSheet(app_stylesheet(preferred_family()))
from molecule_workshop.ui.main_window import MainWindow    # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def shot(win: MainWindow) -> QtGui.QImage:
    img = QtGui.QImage(win.width(), win.height(), QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtGui.QColor("#0b1017"))
    p = QtGui.QPainter(img)
    win.render(p, QtCore.QPoint(0, 0))
    p.end()
    return img


def settle(win: MainWindow) -> None:
    for _ in range(4):
        win.layout().activate()
        app.processEvents()


def main() -> int:
    win = MainWindow()
    win.show()
    settle(win)

    # ---- 量按钮 chrome：按钮宽度 - 文字宽度 ----
    print("=== 按钮内边距（chrome）实测 ===")
    for b in win.side.findChildren(QtWidgets.QPushButton):
        if not b.isVisible():
            continue
        fm = QtGui.QFontMetrics(b.font())
        text_w = fm.horizontalAdvance(b.text())
        print(f"  {b.text().strip()[:20]:<22} 宽={b.width():>4} 文字={text_w:>4} "
              f"chrome={b.width() - text_w:>4}  需要(文本+chrome)={text_w + (b.width()-text_w):>4}")
        break
    widest = max((b for b in win.side.findChildren(QtWidgets.QPushButton) if b.isVisible()),
                 key=lambda b: QtGui.QFontMetrics(b.font()).horizontalAdvance(b.text()) + b.width())
    fm = QtGui.QFontMetrics(widest.font())
    icon_w = widest.iconSize().width() + 8 if not widest.icon().isNull() else 0
    need = fm.horizontalAdvance(widest.text()) + icon_w + 20
    inner = win.side.width() - 2 * 14
    print(f"  最挤的按钮 {widest.text().strip()[:20]!r}: 需要 {need} px")
    print(f"  面板内容区可用宽 = {inner} px")
    print(f"  余量 = {inner - need} px  （必须 > 0，否则字体稍大就会叠字）")

    # ---- 基线：一次到位 ----
    win.resize(1114, 804)
    settle(win)
    base = shot(win)
    base.save(str(OUT / "_r_base.png"))

    # ---- 来回缩放：模拟玩家拖边框 ----
    sizes = [(1400, 900), (980, 660), (1114, 804), (900, 620), (1200, 800),
             (1000, 700), (1114, 804), (1300, 850), (950, 640), (1114, 804)]
    for w, h in sizes:
        win.resize(w, h)
        settle(win)
        _ = shot(win)                      # 强制重绘
    after = shot(win)
    after.save(str(OUT / "_r_after_resize.png"))

    # ---- 逐像素比较 ----
    diff = 0
    first = None
    for y in range(min(base.height(), after.height())):
        for x in range(min(base.width(), after.width())):
            if base.pixel(x, y) != after.pixel(x, y):
                diff += 1
                if first is None:
                    first = (x, y)
    total = base.width() * base.height()
    print(f"\n=== 来回缩放前后差异：{diff} 像素 / {total}（{diff/total*100:.3f}%）===")
    print(f"  第一个差异点：{first}")

    # ---- 侧栏区域单独统计 ----
    side_rect = QtCore.QRect(0, 0, win.side.width() + 24, win.height())
    side_diff = 0
    for y in range(side_rect.top(), min(side_rect.bottom(), base.height())):
        for x in range(side_rect.left(), min(side_rect.right(), base.width())):
            if base.pixel(x, y) != after.pixel(x, y):
                side_diff += 1
    print(f"  其中左侧栏区域差异：{side_diff} 像素")
    if side_diff:
        print("  → 左侧栏在来回缩放后发生了变化：这是重绘残留")
    else:
        print("  → 左侧栏完全稳定，没有重绘残留")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
