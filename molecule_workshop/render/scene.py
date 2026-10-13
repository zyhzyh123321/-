"""场景渲染：QPainter 合成球棍模型。

绘制顺序与要点：
    1. 背景（径向渐变）
    2. 网格（仅编辑模式，纯参考线，不参与定位）
    3. 键：深色外描边 + 主体 + 细高光，模拟圆柱
    4. 原子：预先烘焙好的光照球体贴图，按深度排序（远的先画）
    5. 交互叠加层（虚线预览键、插槽高亮、选中框）
    6. HUD（分子式、提示）

深度排序用画家算法：把所有图元按深度排一次，再顺次绘制。
球体贴图来自 sprites.SpriteCache（启动时用 numpy 烘好，每帧只 drawImage）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6 import QtCore, QtGui

from ..chem.constants import (
    BOND_RADIUS,
    ELEMENTS,
    VIEW_BUILD_RADIUS_SCALE,
    VIEW_FIT_MARGIN,
    VIEW_TRANSITION_MS,
)
from ..chem.vector import contrast_text, ease_in_out_cubic, hex_to_rgb
from ..world import SnapCandidate, World
from .camera import Camera
from .sprites import SpriteCache

# ----------------------------------------------------------------------------
# 主题
# ----------------------------------------------------------------------------


@dataclass
class Theme:
    bg_inner: str = "#141d28"
    bg_outer: str = "#080c11"
    grid_major: str = "rgba(120,180,200,40)"
    grid_minor: str = "rgba(120,180,200,18)"
    origin: str = "rgba(69,224,208,90)"
    bond_dark: str = "#0a0e14"
    bond_body: str = "#aebecb"
    bond_highlight: str = "#dcebf4"
    bond_hover: str = "#45e0d0"
    slot_free: str = "#45e0d0"
    slot_used: str = "rgba(150,170,185,90)"
    preview_ok: str = "#45e0d0"
    preview_bad: str = "#ff5f6d"
    selection: str = "#ffb340"
    select_box: str = "#5ab4eb"      # 框选矩形（中性蓝，区别于删除工具的橙）
    danger: str = "#ff5f6d"          # 即将断开的键
    text: str = "#e6f0f5"
    text_dim: str = "#8fa3b0"


# ----------------------------------------------------------------------------
# 图元
# ----------------------------------------------------------------------------


@dataclass
class DrawItem:
    depth: float
    kind: str                       # 'bond' | 'atom'
    index: int = 0
    a: int = 0
    b: int = 0


@dataclass
class RenderStats:
    atoms: int = 0
    bonds: int = 0
    draw_ms: float = 0.0


# ----------------------------------------------------------------------------


class SceneRenderer:
    """把 World 画到 QPainter 上。编辑模式与 3D 模式共用。"""

    def __init__(self, theme: Theme | None = None) -> None:
        self.theme = theme or Theme()
        self.sprites = SpriteCache()
        self.show_grid = True
        self.show_hydrogens = True
        self.hover_atom_uid: int | None = None
        self.hover_bond_index: int | None = None
        self.selected_uids: list[int] = []
        self.preview: SnapCandidate | None = None      # 拖动吸附候选（画虚线用）
        self.drag_uid: int | None = None
        self.breaking_uid: int | None = None           # 拖过断键阈值的原子（键画红虚线）
        # 连键工具的实时预览：(起点 uid, 光标世界坐标 x, y, 是否命中可行目标)
        self.bond_preview: tuple[int, float, float, bool] | None = None
        # 从原子盘拖出原子时的幽灵球：(元素符号, 世界坐标 x, y, 落点是否可行)
        self.palette_ghost: tuple[str, float, float, bool] | None = None
        self.marquee: tuple[float, float, float, float] | None = None
        self.message: str = ""
        self.stats = RenderStats()
        # 立起来 / 回到平面的过渡动画进度（0..1），1 表示无动画
        self.anim_t = 1.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def paint(self, painter: QtGui.QPainter, world: World, camera: Camera,
              dpr: float = 1.0) -> None:
        import time

        t0 = time.perf_counter()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        is3d = world.mode == "view3d"

        self._paint_background(painter, camera, is3d)
        if self.show_grid and not is3d:
            self._paint_grid(painter, camera)

        # 拖动吸附的虚线画在原子**下面**：
        # 早先把它和原子一起排序绘制，结果虚线从球体中间穿过去、看起来像划伤了原子。
        # 现在先画落点标记与虚线，再画键和原子，线就只出现在球与球之间的空隙里。
        if not is3d:
            self._paint_snap_line(painter, world, camera)
            self._paint_bond_preview(painter, world, camera)
            self._paint_palette_ghost(painter, camera)

        # 收集可见原子与其屏幕坐标
        visible: dict[int, tuple[float, float, float, Vec3, float]] = {}
        for atom in world.atoms:
            if not self.show_hydrogens and atom.el == "H":
                continue
            pos = atom.pos3 if (is3d and atom.pos3 is not None) else (atom.x, atom.y, 0.0)
            sx, sy, depth = camera.project(pos)
            radius = self._atom_radius(atom.el, camera, is3d)
            visible[atom.uid] = (sx, sy, depth, pos, radius)

        items: list[DrawItem] = []
        for i, bond in enumerate(world.bonds):
            if bond.a not in visible or bond.b not in visible:
                continue
            depth = max(visible[bond.a][2], visible[bond.b][2])
            items.append(DrawItem(depth=depth, kind="bond", index=i, a=bond.a, b=bond.b))
        for uid in visible:
            items.append(DrawItem(depth=visible[uid][2], kind="atom", index=uid))
        items.sort(key=lambda it: it.depth)

        for item in items:
            if item.kind == "bond":
                # 拖动超过断键阈值时，该原子身上的键画成红色虚线，
                # 让玩家在松手前就知道「这些键要断了」
                will_break = (self.breaking_uid is not None
                              and (item.a == self.breaking_uid or item.b == self.breaking_uid))
                self._paint_bond(painter, world, item.index, visible, camera, is3d,
                                 will_break=will_break)
            else:
                self._paint_atom(painter, world, item.index, visible, is3d)

        # 插槽指示（仅编辑模式、非氢原子）
        if not is3d:
            for atom in world.atoms:
                if atom.el == "H" or atom.uid not in visible:
                    continue
                self._paint_slots(painter, world, atom, visible[atom.uid], camera)

        self._paint_interaction(painter, world, camera, visible, is3d)
        self._paint_hud(painter, world, camera)

        self.stats = RenderStats(atoms=len(visible), bonds=len(world.bonds),
                                 draw_ms=(time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------------
    # 背景与网格
    # ------------------------------------------------------------------

    def _paint_background(self, painter: QtGui.QPainter, camera: Camera, is3d: bool) -> None:
        w, h = camera.width, camera.height
        grad = QtGui.QRadialGradient(w * 0.5, h * 0.42, 1.0, w * 0.5, h * 0.5, max(w, h) * 0.78)
        grad.setColorAt(0.0, QtGui.QColor(self.theme.bg_inner))
        grad.setColorAt(1.0, QtGui.QColor(self.theme.bg_outer))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(grad))
        painter.drawRect(QtCore.QRectF(0, 0, w, h))

    def _paint_grid(self, painter: QtGui.QPainter, camera: Camera) -> None:
        """参考网格。

        网格只是视觉参考，**不参与原子定位** ——
        六方向插槽与方格在几何上不兼容（键长=1 时只有 0°/180° 能落在格点上），
        所以拖动时不做任何网格吸附。
        """
        w, h = camera.width, camera.height
        s = camera.screen_scale()
        if s <= 0:
            return
        step = 0.5
        while step * s < 14:
            step *= 2
        while step * s > 52:
            step /= 2

        corners = [camera.screen_to_plane(0, 0), camera.screen_to_plane(w, 0),
                   camera.screen_to_plane(0, h), camera.screen_to_plane(w, h)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if not math.isfinite(min_x) or (max_x - min_x) / step > 400:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for major in (True, False):
            st = step * 2 if major else step
            painter.setPen(QtGui.QPen(QtGui.QColor(self.theme.grid_major if major else self.theme.grid_minor), 1))
            k = math.ceil(min_x / st)
            while k * st <= max_x:
                x = k * st
                p1 = camera.project((x, min_y, 0.0))
                p2 = camera.project((x, max_y, 0.0))
                painter.drawLine(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1]))
                k += 1
            k = math.ceil(min_y / st)
            while k * st <= max_y:
                y = k * st
                p1 = camera.project((min_x, y, 0.0))
                p2 = camera.project((max_x, y, 0.0))
                painter.drawLine(QtCore.QPointF(p1[0], p1[1]), QtCore.QPointF(p2[0], p2[1]))
                k += 1
        painter.restore()

        o = camera.project((0.0, 0.0, 0.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(self.theme.origin), 1.4))
        painter.drawEllipse(QtCore.QPointF(o[0], o[1]), 3.5, 3.5)

    # ------------------------------------------------------------------
    # 原子与键
    # ------------------------------------------------------------------

    def _atom_radius(self, el: str, camera: Camera, is3d: bool) -> float:
        info = ELEMENTS.get(el, {"radius": 0.5})
        factor = 1.0 if is3d else VIEW_BUILD_RADIUS_SCALE
        return info["radius"] * factor * camera.screen_scale()

    def _paint_atom(self, painter: QtGui.QPainter, world: World, uid: int,
                    visible: dict, is3d: bool) -> None:
        sx, sy, _depth, _pos, radius = visible[uid]
        if radius < 1.0:
            return
        atom = world.atom(uid)
        if atom is None:
            return
        info = ELEMENTS.get(atom.el, {"color": "#808080", "radius": 0.5})

        diameter = int(max(6, radius * 2))
        sprite = self.sprites.sphere(info["color"], diameter)
        target = QtCore.QRectF(sx - radius, sy - radius, radius * 2, radius * 2)
        painter.drawImage(target, sprite)

        # 元素符号
        font_px = radius * (1.15 if atom.el == "H" else 1.0)
        if font_px >= 7.5:
            painter.save()
            painter.setPen(QtGui.QColor(contrast_text(info["color"])))
            f = painter.font()
            f.setPixelSize(max(7, int(font_px)))
            f.setBold(True)
            painter.setFont(f)
            painter.drawText(QtCore.QRectF(sx - radius, sy - radius, radius * 2, radius * 2),
                             QtCore.Qt.AlignmentFlag.AlignCenter, atom.el)
            painter.restore()

        # 高亮环
        if uid == self.drag_uid:
            pen_color, width = self.theme.selection, 2.6
        elif uid == self.hover_atom_uid:
            pen_color, width = self.theme.preview_ok, 2.0
        elif uid in self.selected_uids:
            pen_color, width = self.theme.selection, 2.2
        else:
            pen_color = None
            width = 0
        if pen_color:
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(pen_color), width))
            painter.drawEllipse(QtCore.QPointF(sx, sy), radius + 3.0, radius + 3.0)

    def _paint_bond(self, painter: QtGui.QPainter, world: World, bond_index: int,
                    visible: dict, camera: Camera, is3d: bool,
                    will_break: bool = False) -> None:
        bond = world.bonds[bond_index]
        ax, ay, _ad, _ap, ra = visible[bond.a]
        bx, by, _bd, _bp, rb = visible[bond.b]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux

        scale = camera.screen_scale()
        # 键宽随缩放变化，并限制在合理范围
        base_width = max(1.6, BOND_RADIUS * 2.0 * scale)
        if will_break:
            body_color = self.theme.danger
        elif bond_index == self.hover_bond_index:
            body_color = self.theme.bond_hover
        else:
            body_color = self.theme.bond_body

        orders = [0] if bond.order == 1 else ([0, 1] if bond.order == 2 else [0, 1, 2])
        count = len(orders)
        spread = 0.0 if count == 1 else base_width * 1.5

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        for i, k in enumerate(orders):
            off = (k - (count - 1) / 2.0) * spread
            # 键从球面伸出一点，让连接处看起来是"插进去"的
            x1 = ax + ux * ra * 0.80 + nx * off
            y1 = ay + uy * ra * 0.80 + ny * off
            x2 = bx - ux * rb * 0.80 + nx * off
            y2 = by - uy * rb * 0.80 + ny * off

            line_w = base_width * (0.92 if count == 1 else 0.60)
            # 外描边：让键在球体之间界限清晰
            painter.setPen(QtGui.QPen(QtGui.QColor(self.theme.bond_dark),
                                      line_w + max(2.4, line_w * 0.5),
                                      QtCore.Qt.PenStyle.SolidLine,
                                      QtCore.Qt.PenCapStyle.RoundCap))
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
            # 主体（将要断开的键画成虚线，提示玩家）
            body_pen = QtGui.QPen(QtGui.QColor(body_color), line_w,
                                  QtCore.Qt.PenStyle.DashLine if will_break
                                  else QtCore.Qt.PenStyle.SolidLine,
                                  QtCore.Qt.PenCapStyle.RoundCap)
            if will_break:
                body_pen.setDashPattern([5, 3.5])
            painter.setPen(body_pen)
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
            # 高光条（只对单键且未将要断开时；多键画高光会显脏）
            if count == 1 and line_w > 3.0 and not will_break:
                hl = QtGui.QColor(self.theme.bond_highlight)
                hl.setAlpha(120)
                ox, oy = -line_w * 0.24, -line_w * 0.28
                painter.setPen(QtGui.QPen(hl, line_w * 0.34,
                                          QtCore.Qt.PenStyle.SolidLine,
                                          QtCore.Qt.PenCapStyle.RoundCap))
                painter.drawLine(QtCore.QPointF(x1 + ox, y1 + oy),
                                 QtCore.QPointF(x2 + ox, y2 + oy))
        painter.restore()

    def _paint_slots(self, painter: QtGui.QPainter, world: World, atom,
                     vis_entry: tuple, camera: "Camera") -> None:
        """价键插槽指示：空插槽画在**真实空闲方向**上，已占的画暗点。

        早先这里是「取 SLOT_DIRS_2D 的前 valence 个方向」——
        与实际空闲方向**不一致**（碳连了一根键后，画出来的仍是 0/60/120/180），
        玩家照着亮点拖反而连不上。现在改由 world.free_slot_points() 提供，
        和吸附逻辑用的是同一套方向，所见即所得。
        """
        sx, sy, _d, _p, radius = vis_entry
        info = ELEMENTS.get(atom.el)
        if info is None:
            return
        dot_r = max(1.8, min(4.6, radius * 0.20))
        ring = radius + dot_r * 2.2 + 2.5

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # 已占方向：暗色小点
        for ux, uy in world.occupied_directions(atom.uid):
            cx = sx + ux * ring
            cy = sy - uy * ring
            painter.setBrush(QtGui.QBrush(QtGui.QColor(150, 170, 185, 70)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(QtCore.QPointF(cx, cy), dot_r * 0.72, dot_r * 0.72)

        # 空插槽：亮点 + 光晕。
        # 圆点直接画在 free_slot_points 给出的世界坐标上（可能因为插槽太密被外推过），
        # 所以「亮点在哪」和「松手会吸到哪」完全一致 —— 所见即所得。
        for slot_index, ((_dx, _dy), point) in enumerate(world.free_slot_points(atom.uid)):
            px, py, _pd = camera.project((point[0], point[1], 0.0))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(self.theme.slot_free)))
            painter.setPen(QtGui.QPen(QtGui.QColor(69, 224, 208, 90), 2.0))
            painter.drawEllipse(QtCore.QPointF(px, py), dot_r, dot_r)

            # 插槽被吸附候选选中时加一圈光晕（用 slot_index 精确匹配，
            # 不再比较浮点方向 —— 那样会因精度问题偶尔高亮错插槽）
            if (self.preview is not None
                    and self.preview.host_uid == atom.uid
                    and self.preview.slot_index == slot_index):
                color = (self.theme.preview_ok if self.preview.feasible
                         else self.theme.preview_bad)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.4))
                painter.drawEllipse(QtCore.QPointF(px, py), dot_r + 4.5, dot_r + 4.5)
        painter.restore()

    # ------------------------------------------------------------------
    # 交互叠加层
    # ------------------------------------------------------------------

    def _paint_snap_line(self, painter: QtGui.QPainter, world: World,
                         camera: Camera) -> None:
        """拖动吸附预览：从宿主原子到理想落点的虚线 + 落点标记。

        这一步在画原子之前调用，所以虚线只出现在球体之间的空隙里，
        不会从球中间穿过去（那样看着像把原子划伤了）。
        """
        cand = self.preview
        if cand is None:
            return
        host = world.atom(cand.host_uid)
        if host is None:
            return

        color = self.theme.preview_ok if cand.feasible else self.theme.preview_bad
        hx, hy, _ = camera.project((host.x, host.y, 0.0))
        px, py, _ = camera.project((cand.point[0], cand.point[1], 0.0))

        radius = self._atom_radius(cand.host_el, camera, False)
        # 从球面外缘起画
        dx, dy = px - hx, py - hy
        length = math.hypot(dx, dy)
        if length > 1e-6:
            ux, uy = dx / length, dy / length
            start_offset = min(radius * 0.98, length * 0.5)
            hx, hy = hx + ux * start_offset, hy + uy * start_offset

        painter.save()
        pen = QtGui.QPen(QtGui.QColor(color), 2.6, QtCore.Qt.PenStyle.DashLine)
        pen.setDashPattern([6, 4])
        painter.setPen(pen)
        painter.drawLine(QtCore.QPointF(hx, hy), QtCore.QPointF(px, py))

        if cand.feasible:
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(69, 224, 208, 50)))
            painter.drawEllipse(QtCore.QPointF(px, py), 12.0, 12.0)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.2))
            painter.drawLine(QtCore.QPointF(px - 5, py), QtCore.QPointF(px + 5, py))
            painter.drawLine(QtCore.QPointF(px, py - 5), QtCore.QPointF(px, py + 5))
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.0,
                                      QtCore.Qt.PenStyle.DashLine))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QtCore.QPointF(px, py), 12.0, 12.0)
        painter.restore()

    def _paint_bond_preview(self, painter: QtGui.QPainter, world: World,
                            camera: Camera) -> None:
        """连键工具的实时预览：从起点原子拉一条虚线到光标。

        早先按住连键时画面上毫无反馈（`_bond_cursor` 被赋值了却从没被绘制），
        玩家不知道自己在连什么。现在：
            虚线终点在命中可行插槽时**吸附到那个插槽**并在目标蓝点外圈高亮，
            否则就跟着光标走、用红色表示"连不上"。
        """
        pv = self.bond_preview
        if pv is None:
            return
        src_uid, wx, wy, feasible = pv
        src = world.atom(src_uid)
        if src is None:
            return

        color = self.theme.preview_ok if feasible else self.theme.preview_bad
        sx, sy, _ = camera.project((src.x, src.y, 0.0))
        ex, ey, _ = camera.project((wx, wy, 0.0))

        # 从球面外缘起画，避免线穿过球体
        radius = self._atom_radius(src.el, camera, False)
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length > 1e-6:
            ux, uy = dx / length, dy / length
            sx, sy = sx + ux * min(radius * 0.98, length * 0.5), \
                sy + uy * min(radius * 0.98, length * 0.5)

        painter.save()
        pen = QtGui.QPen(QtGui.QColor(color), 2.6, QtCore.Qt.PenStyle.DashLine)
        pen.setDashPattern([7, 5])
        painter.setPen(pen)
        painter.drawLine(QtCore.QPointF(sx, sy), QtCore.QPointF(ex, ey))

        # 终点标记：十字准星
        painter.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QtCore.QPointF(ex, ey), 7.0, 7.0)
        painter.drawLine(QtCore.QPointF(ex - 11, ey), QtCore.QPointF(ex + 11, ey))
        painter.drawLine(QtCore.QPointF(ex, ey - 11), QtCore.QPointF(ex, ey + 11))
        painter.restore()

    def _paint_palette_ghost(self, painter: QtGui.QPainter, camera: Camera) -> None:
        """从原子盘拖出原子时的落点预览：半透明球。

        可行性用颜色区分（能放 / 插槽已满），让玩家在松手前就知道能不能放。
        """
        ghost = self.palette_ghost
        if ghost is None:
            return
        el, wx, wy, feasible = ghost
        info = ELEMENTS.get(el)
        if info is None:
            return

        gx, gy, _ = camera.project((wx, wy, 0.0))
        radius = info["radius"] * VIEW_BUILD_RADIUS_SCALE * camera.screen_scale()
        color = QtGui.QColor(info["color"])
        color.setAlpha(140)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtGui.QBrush(color))
        ring = self.theme.preview_ok if feasible else self.theme.preview_bad
        painter.setPen(QtGui.QPen(QtGui.QColor(ring), 2.4,
                                  QtCore.Qt.PenStyle.SolidLine))
        painter.drawEllipse(QtCore.QPointF(gx, gy), radius, radius)
        painter.restore()

    def _paint_interaction(self, painter: QtGui.QPainter, world: World, camera: Camera,
                           visible: dict, is3d: bool) -> None:
        # 吸附失败原因的提示文字（画在原子之上，保证可读）
        cand = self.preview
        if cand is not None and not cand.feasible and cand.reason and not is3d:
            px, py, _ = camera.project((cand.point[0], cand.point[1], 0.0))
            painter.save()
            painter.setPen(QtGui.QColor(self.theme.preview_bad))
            f = painter.font()
            f.setPixelSize(12)
            painter.setFont(f)
            painter.drawText(QtCore.QPointF(px + 16, py - 12), cand.reason)
            painter.restore()

        # 框选矩形
        if self.marquee is not None:
            x, y, w, h = self.marquee
            painter.save()
            # 中性蓝：区别于「删除工具」的橙色框选，避免误以为一定会删
            painter.setPen(QtGui.QPen(QtGui.QColor(self.theme.select_box), 1.6,
                                      QtCore.Qt.PenStyle.DashLine))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(90, 180, 235, 38)))
            painter.drawRect(QtCore.QRectF(x, y, w, h))
            painter.restore()

    def _paint_hud(self, painter: QtGui.QPainter, world: World, camera: Camera) -> None:
        """画面内的空态提示。

        注意：这里**不再**绘制状态提示文字。
        窗口底部已有状态栏负责显示提示，画面内再画一遍会和状态栏叠字
        （第一版预览图里出现的就是这个：两行字重在一起、互相糊掉）。
        视图只负责画分子，文字交给界面层。
        """
        if world.atoms:
            return
        painter.save()
        painter.setPen(QtGui.QColor(self.theme.text_dim))
        f = painter.font()
        f.setPixelSize(15)
        painter.setFont(f)
        painter.drawText(
            QtCore.QRectF(0, 0, camera.width, camera.height),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "拖出原子放置：按住左侧原子盘里的元素拖到画布",
        )
        painter.restore()

    # ------------------------------------------------------------------
    # 便捷：几何中心（用于视图适配）
    # ------------------------------------------------------------------

    @staticmethod
    def content_radius(world: World) -> float:
        if not world.atoms:
            return 1.0
        pts = []
        for atom in world.atoms:
            if atom.pos3 is not None:
                pts.append(atom.pos3)
            else:
                pts.append((atom.x, atom.y, 0.0))
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        return max((math.dist(p, (cx, cy, cz)) for p in pts), default=1.0)
