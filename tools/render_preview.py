"""离屏渲染预览：把界面画成 PNG，供无头环境自查。

这是本项目「能自己看见画面」的关键工具 ——
网页版时代无法截图（浏览器无头被沙箱挡住），只能靠猜，导致大量返工。
现在每改一次渲染，都可以跑一次这个脚本、直接看图上有没有问题。

用法：
    python tools/render_preview.py            # 输出全部场景
    python tools/render_preview.py editor     # 只输出某个场景
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在导入 QtWidgets 之前设置，否则离屏平台不生效
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets          # noqa: E402

from molecule_workshop.chem.constants import ELEMENTS  # noqa: E402
from molecule_workshop.render.camera import Camera     # noqa: E402
from molecule_workshop.render.scene import SceneRenderer, Theme  # noqa: E402
from molecule_workshop.world import World              # noqa: E402

OUT_DIR = ROOT / "out"
W, H = 1000, 660

# 中文字体必须显式加载：离屏平台下 Qt 找不到系统字体目录，
# 不加载的话界面文字会变成方块。
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


def load_fonts() -> str | None:
    family = None
    for path in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        fid = QtGui.QFontDatabase.addApplicationFont(path)
        if fid >= 0:
            fams = QtGui.QFontDatabase.applicationFontFamilies(fid)
            if fams and family is None:
                family = fams[0]
    return family


def new_image() -> tuple[QtGui.QImage, QtGui.QPainter]:
    img = QtGui.QImage(W, H, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QtGui.QColor("#080c11"))
    painter = QtGui.QPainter(img)
    return img, painter


def make_camera(top_down: bool, radius: float = 1.0, fit: bool = False,
                world: World | None = None) -> Camera:
    cam = Camera()
    cam.width, cam.height = W, H
    if top_down:
        cam.azimuth = 0.0
        cam.elevation = 0.0
    else:
        cam.azimuth = 0.55
        cam.elevation = 0.30
    cam.zoom = 1.0
    if fit:
        if world is not None:
            # 按分子在编辑平面上的实际范围自动缩放，否则小分子会缩成一小团
            xs = [a.x for a in world.atoms]
            ys = [a.y for a in world.atoms]
            pts = [(a.pos3 if a.pos3 else (a.x, a.y, 0.0)) for a in world.atoms]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cz = sum(p[2] for p in pts) / len(pts)
            radius = max((math.dist(p, (cx, cy, cz)) for p in pts), default=1.0) + 0.5
        cam.fit(radius, 1.30)
        cam.pan_x = -(sum(a.x for a in world.atoms) / len(world.atoms)) * cam.screen_scale() if world and top_down else 0.0
        cam.pan_y = (sum(a.y for a in world.atoms) / len(world.atoms)) * cam.screen_scale() if world and top_down else 0.0
    return cam


def draw_side_panels(painter: QtGui.QPainter, world: World, family: str | None,
                     tool: str = "鼠标", hint: str = "") -> None:
    """画一个简化版界面框架（工具栏 + 资料面板 + 状态栏），用于检查排版与中文渲染。"""
    font = QtGui.QFont(family or "Sans Serif")

    # ---- 左侧工具栏 ----
    panel = QtCore.QRectF(16, 16, 208, 430)
    painter.setPen(QtGui.QPen(QtGui.QColor(90, 200, 220, 46), 1))
    painter.setBrush(QtGui.QBrush(QtGui.QColor(17, 25, 34, 235)))
    painter.drawRoundedRect(panel, 12, 12)

    painter.setPen(QtGui.QColor("#5d7080"))
    font.setPixelSize(11)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QtCore.QPointF(32, 44), "原 子 盘")

    y = 60
    for el in ["C", "H", "O", "N"]:
        info = ELEMENTS[el]
        row = QtCore.QRectF(28, y, 184, 34)
        painter.setPen(QtGui.QPen(QtGui.QColor(69, 224, 208, 90), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(69, 224, 208, 26)))
        painter.drawRoundedRect(row, 9, 9)
        # 元素色块
        color = QtGui.QColor(info["color"])
        grad = QtGui.QRadialGradient(46 - 5, y + 17 - 5, 1.0, 46, y + 17, 12)
        grad.setColorAt(0.0, color.lighter(190))
        grad.setColorAt(1.0, color.darker(220))
        painter.setBrush(QtGui.QBrush(grad))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(QtCore.QPointF(46, y + 17), 12, 12)
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#e6f0f5"))
        painter.drawText(QtCore.QPointF(68, y + 16), f"{el} {info['name']}")
        font.setPixelSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#5d7080"))
        painter.drawText(QtCore.QPointF(68, y + 29), f"{info['valence']} 个价键")
        y += 40

    # 工具区
    y += 8
    painter.setPen(QtGui.QColor("#5d7080"))
    font.setPixelSize(11)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QtCore.QPointF(32, y), "工 具")
    y += 18
    for name, key in [("鼠标", "V"), ("连键", "B"), ("删除", "X"), ("平移", "Space")]:
        active = (name == tool)
        row = QtCore.QRectF(28, y, 184, 30)
        painter.setPen(QtGui.QPen(QtGui.QColor(69, 224, 208, 140) if active
                                  else QtGui.QColor(120, 170, 190, 46), 1.4 if active else 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(69, 224, 208, 40) if active
                                      else QtGui.QColor(255, 255, 255, 10)))
        painter.drawRoundedRect(row, 8, 8)
        font.setPixelSize(13)
        font.setBold(active)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#e6f0f5") if active else QtGui.QColor("#8fa3b0"))
        painter.drawText(QtCore.QPointF(40, y + 20), name)
        font.setPixelSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#5d7080"))
        painter.drawText(QtCore.QRectF(140, y + 8, 66, 16),
                         QtCore.Qt.AlignmentFlag.AlignRight, key)
        y += 34

    # ---- 右侧资料面板 ----
    info_rect = QtCore.QRectF(W - 320, 16, 304, 470)
    painter.setPen(QtGui.QPen(QtGui.QColor(90, 200, 220, 46), 1))
    painter.setBrush(QtGui.QBrush(QtGui.QColor(17, 25, 34, 238)))
    painter.drawRoundedRect(info_rect, 12, 12)

    parts, total = world.identify()
    x0 = info_rect.x() + 16
    y = info_rect.y() + 32
    if parts:
        main = parts[0]
        rec = main.matched
        font.setPixelSize(19)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#e6f0f5"))
        painter.drawText(QtCore.QPointF(x0, y), rec["names"][0] if rec else main.formula)
        y += 18
        if rec:
            font.setPixelSize(11)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#5d7080"))
            painter.drawText(QtCore.QPointF(x0, y), rec["en"])
            y += 16

        painter.setPen(QtGui.QPen(QtGui.QColor(120, 170, 190, 40), 1))
        painter.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(info_rect.right() - 16, y))
        y += 20

        # 四格信息
        cells = [("分子式", main.formula),
                 ("相对分子质量", f"{main.mass:.2f}" if main.mass else "—"),
                 ("原子数", f"{main.count} 个"),
                 ("状态", "已收录" if rec else "未收录")]
        for i, (k, v) in enumerate(cells):
            cx = x0 + (i % 2) * 140
            cy = y + (i // 2) * 44
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 9)))
            painter.drawRoundedRect(QtCore.QRectF(cx, cy, 130, 38), 8, 8)
            font.setPixelSize(10)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#5d7080"))
            painter.drawText(QtCore.QPointF(cx + 9, cy + 15), k)
            font.setPixelSize(14)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#45e0d0") if k == "状态" and rec else QtGui.QColor("#e6f0f5"))
            painter.drawText(QtCore.QPointF(cx + 9, cy + 32), str(v))
        y += 44 * 2 + 12

        if rec and rec.get("desc"):
            font.setPixelSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#5d7080"))
            painter.drawText(QtCore.QPointF(x0, y), "结 构 特 征")
            y += 16
            font.setPixelSize(12)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#c3d4de"))
            rect = QtCore.QRectF(x0, y, info_rect.width() - 32, 96)
            painter.drawText(rect, QtCore.Qt.TextFlag.TextWordWrap
                             | QtCore.Qt.AlignmentFlag.AlignTop, rec["desc"])
            y += 104

        # 元素组成条
        font.setPixelSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#5d7080"))
        painter.drawText(QtCore.QPointF(x0, y), "元 素 组 成")
        y += 16
        for comp in main.composition:
            font.setPixelSize(11)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#8fa3b0"))
            painter.drawText(QtCore.QPointF(x0, y + 4), f"{comp['name']} {comp['symbol']}")
            bar = QtCore.QRectF(x0 + 84, y, 130, 7)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 18)))
            painter.drawRoundedRect(bar, 3.5, 3.5)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(ELEMENTS[comp["symbol"]]["color"])))
            painter.drawRoundedRect(QtCore.QRectF(bar.x(), bar.y(),
                                                  max(2.0, bar.width() * comp["percent"] / 100.0),
                                                  bar.height()), 3.5, 3.5)
            painter.setPen(QtGui.QColor("#8fa3b0"))
            painter.drawText(QtCore.QRectF(x0 + 220, y - 4, 60, 16),
                             QtCore.Qt.AlignmentFlag.AlignRight,
                             f"{comp['percent']:.1f}%")
            y += 20

    # ---- 状态栏 ----
    bar = QtCore.QRectF(16, H - 46, W - 32, 32)
    painter.setPen(QtGui.QPen(QtGui.QColor(90, 200, 220, 40), 1))
    painter.setBrush(QtGui.QBrush(QtGui.QColor(17, 25, 34, 230)))
    painter.drawRoundedRect(bar, 9, 9)
    font.setPixelSize(12)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QtGui.QColor("#8fa3b0"))
    painter.drawText(QtCore.QPointF(bar.x() + 14, bar.y() + 21),
                     f"工具：{tool}   ·   {hint}")
    painter.setPen(QtGui.QColor("#5d7080"))
    painter.drawText(QtCore.QRectF(bar.right() - 300, bar.y() + 8, 288, 18),
                     QtCore.Qt.AlignmentFlag.AlignRight,
                     f"原子 {world.atom_count}   化学键 {world.bond_count}")


# ----------------------------------------------------------------------------
# 场景构造
# ----------------------------------------------------------------------------


def scene_editor(renderer: SceneRenderer) -> World:
    """编辑模式：搭一个乙醇的骨架（展示键级、插槽、多方向）。"""
    w = World()
    c1 = w.create_atom("C", -1.0, 0.0)[2]
    c2 = w.create_atom("C", 0.0, 0.0)[2]
    o = w.create_atom("O", 1.0, 0.0)[2]
    w.add_bond(c1.uid, c2.uid, 1)
    w.add_bond(c2.uid, o.uid, 1)
    renderer.hover_atom_uid = c2.uid
    renderer.message = "拖动原子把插槽对上就连成键"
    return w


def scene_hover(renderer: SceneRenderer) -> World:
    """拖动吸附预览：正在把第三个碳拖到已有碳的空插槽上（画虚线）。"""
    w = World()
    c1 = w.create_atom("C", -0.5, 0.0)[2]
    c2 = w.create_atom("C", 0.5, 0.0)[2]
    w.add_bond(c1.uid, c2.uid, 1)
    moving = w.create_atom("C", 0.0, 1.0)[2]
    # 把移动原子放到某个插槽附近，触发候选
    w.move_atom(moving.uid, 1.0, 0.55)
    cand = w.find_snap_candidate(moving.uid)
    renderer.preview = cand
    renderer.drag_uid = moving.uid
    renderer.message = "靠近空插槽时出现虚线，松手自动吸附成键"
    return w


def scene_snap_blocked(renderer: SceneRenderer) -> World:
    """不可行吸附：宿主价键已满，虚线应为红色并给出原因。"""
    w = World()
    c = w.create_atom("C", 0.0, 0.0)[2]
    for i, (dx, dy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        h = w.create_atom("H", dx, dy)[2]
        w.add_bond(c.uid, h.uid, 1)
    moving = w.create_atom("C", 0.9, 0.9)[2]
    cand = w.find_snap_candidate(moving.uid)
    if cand is None:
        moving.x, moving.y = 1.0, 0.5
        cand = w.find_snap_candidate(moving.uid)
    renderer.preview = cand
    renderer.drag_uid = moving.uid
    renderer.message = "插槽不足时虚线变红并提示原因"
    return w


def scene_bonds(renderer: SceneRenderer) -> World:
    """键级展示：单键 / 双键 / 三键并排对比。"""
    w = World()
    a1 = w.create_atom("C", -2.2, 0.0)[2]
    b1 = w.create_atom("C", -1.2, 0.0)[2]
    w.add_bond(a1.uid, b1.uid, 1)
    a2 = w.create_atom("C", -0.4, 0.0)[2]
    b2 = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a2.uid, b2.uid, 2)
    a3 = w.create_atom("C", 1.4, 0.0)[2]
    b3 = w.create_atom("C", 2.4, 0.0)[2]
    w.add_bond(a3.uid, b3.uid, 3)
    renderer.hover_bond_index = 1
    renderer.message = "点已有的键可切换键级：单键 → 双键 → 三键"
    return w


def scene_3d(renderer: SceneRenderer, spec: str, name: str) -> World:
    """3D 视图：立起来的分子。"""
    from molecule_workshop.chem.parse_spec import parse_spec

    mol = parse_spec(spec)
    w = World()
    ids = []
    for i, atom in enumerate(mol.atoms):
        ids.append(w.create_atom(atom.el, (i % 8) * 1.6, (i // 8) * 1.6)[2].uid)
    for bond in mol.bonds:
        w.add_bond(ids[bond.a], ids[bond.b], bond.order)
    ok, msg, radius = w.solve_geometry()
    if not ok:
        renderer.message = "求解失败：" + msg
    renderer.message = f"{name} — 拖动旋转视角，滚轮缩放"
    return w


# ----------------------------------------------------------------------------


def render_one(renderer: SceneRenderer, world: World, path: Path, family: str | None,
               camera: Camera, tool: str = "鼠标", hint: str = "",
               panels: bool = True) -> None:
    img, painter = new_image()
    renderer.paint(painter, world, camera, dpr=1.0)
    if panels:
        draw_side_panels(painter, world, family, tool, hint or renderer.message)
    painter.end()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = img.save(str(path))
    print(f"  {'OK ' if ok else 'FAIL'} {path.name}  ({img.width()}x{img.height()})")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    family = load_fonts()
    print(f"中文字体: {family or '未加载（文字可能显示为方块）'}")
    print(f"输出目录: {OUT_DIR}")

    def fresh() -> SceneRenderer:
        return SceneRenderer(Theme())

    if which in ("all", "editor"):
        r = fresh()
        w = scene_editor(r)
        render_one(r, w, OUT_DIR / "01_editor.png", family,
                   make_camera(top_down=True, fit=True, world=w), "鼠标", "拖动原子把插槽对上就连成键")
    if which in ("all", "hover"):
        r = fresh()
        w = scene_hover(r)
        render_one(r, w, OUT_DIR / "02_snap_preview.png", family,
                   make_camera(top_down=True, fit=True, world=w), "鼠标", "松手自动吸附成键")
    if which in ("all", "blocked"):
        r = fresh()
        w = scene_snap_blocked(r)
        render_one(r, w, OUT_DIR / "03_snap_blocked.png", family,
                   make_camera(top_down=True, fit=True, world=w), "鼠标", "插槽不足时虚线变红")
    if which in ("all", "bonds"):
        r = fresh()
        w = scene_bonds(r)
        render_one(r, w, OUT_DIR / "04_bond_orders.png", family,
                   make_camera(top_down=True, fit=True, world=w), "鼠标", "点键切换键级")
    if which in ("all", "tools"):
        r = fresh()
        w = scene_editor(r)
        render_one(r, w, OUT_DIR / "05_tool_delete.png", family,
                   make_camera(top_down=True, fit=True, world=w), "删除", "点原子即删除（含它的所有键）")
    if which in ("all", "3d"):
        for i, (spec, name) in enumerate([
            ("C(-H)(-H)(-H)(-H)", "甲烷"),
            ("C1(-H)=C(-H)-C(-H)=C(-H)-C(-H)=C(-H)-1", "苯"),
            ("C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1", "环己烷"),
        ]):
            r = fresh()
            r.show_grid = False
            w = scene_3d(r, spec, name)
            cam = make_camera(top_down=False, radius=SceneRenderer.content_radius(w), fit=True)
            render_one(r, w, OUT_DIR / f"1{i}_3d_{['methane','benzene','cyclohexane'][i]}.png",
                       family, cam, "鼠标", r.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
