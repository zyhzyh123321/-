"""画布控件：所有鼠标/键盘交互与渲染都挂在这里。

交互设计（这是本次重写改动最大的部分）：

  工具区（工具按钮或快捷键切换）
      鼠标 V   选中 / 拖动原子 / 框选
      连键 B   从原子拖出一条虚线连到另一个原子
      删除 X   点原子删原子，点键删键，拖动框选批量删
      平移 ␣   拖动画布

  拖动原子
      * **不做网格吸附** —— 位置完全跟手（旧版的「生硬」就来自强制对齐到 0.5 网格，
        而且六方向插槽与方格在几何上并不兼容）
      * 靠近别的原子的空插槽时出现**虚线预览**并高亮目标插槽
      * 松开鼠标时**精确吸附**到理想键位并自动成键
      * 插槽不够时虚线变红并给出原因（例如「碳的 4 个价键已用满」）
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from ..chem.constants import (
    CANVAS_MIN_HEIGHT,
    CANVAS_MIN_WIDTH,
    DRAG_BREAK_DIST,
    DRAG_RESTORE_EPS,
    DRAG_ROTATE_MIN_PX,
    ELEMENTS,
    MARQUEE_MIN_PX,
    RELAX_MAX_ATOMS,
    SLOT_PICK_RADIUS_PX,
    SLOT_SNAP_RADIUS,
    VIEW_FIT_MARGIN,
    VIEW_TRANSITION_MS,
)
from ..chem.vector import ease_in_out_cubic, normalize2
from ..world import SnapCandidate, World
from .camera import Camera
from .scene import SceneRenderer

TOOLS = ("鼠标", "连键", "删除", "平移")
TOOL_KEYS = {"V": "鼠标", "B": "连键", "D": "删除", "Space": "平移"}

TOOL_HINTS = {
    "鼠标": "左键点选/拖动原子，拖空白框选；Ctrl+点击多选；右键出菜单",
    "连键": "按住原子上的蓝点拖到另一个蓝点，松手连键；连完自动摆到标准键长",
    "删除": "点原子/键删除，拖动可框选批量删除",
    "平移": "拖动画布平移视图（中键与 Shift+中键同样可以）",
}

#: 原子盘拖放用的 MIME 类型
ELEMENT_MIME = "application/x-mol-element"


class CanvasWidget(QtWidgets.QWidget):
    """自绘画布。负责输入、相机操作与把渲染委托给 SceneRenderer。

    工具分工（v4）：
        鼠标 V —— 左键点选 / 拖动 / 框选；Ctrl+点击多选
        连键 B —— 按住原子上的**蓝点**拖到另一个蓝点，松手连键
        删除 D —— 点删除 + 框选删除
        平移 Space —— 拖动画布

    放原子：**从左侧原子盘按住拖到画布**（v4 起唯一的放置方式），
    或右键菜单「在此放置」。左键在画布上不再放原子。

    鼠标与相机：
        左键           选择 / 拖动原子（带键时绕锚点旋转）/ 框选
        Ctrl + 左键    多选（加入或移出选择）
        X / Delete     删除所选
        + / -          鼠标指向的键升/降一级
        右键           上下文菜单（成键 / 升键级 / 删除 / 在此放置）
        中键           3D 旋转，2D 平移
        Shift + 中键   平移视图（两种模式都可）
        滚轮           以光标为中心缩放
    """

    message_changed = QtCore.Signal(str)
    world_changed = QtCore.Signal()
    #: 请求主窗口弹出上下文菜单（在画布坐标系里）
    context_menu_requested = QtCore.Signal(QtCore.QPoint, int)

    def __init__(self, world: World, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.world = world
        self.renderer = SceneRenderer()
        self.camera = Camera()
        self.tool = "鼠标"
        self.selected_element = "C"

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(CANVAS_MIN_WIDTH, CANVAS_MIN_HEIGHT)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        # 相机尺寸必须在收到第一个 resizeEvent 之前就有值，
        # 否则相机宽高为 0，命中测试与坐标换算全部失效
        # （表现为"点原子选不中、点空白清不掉选择"）。
        self._sync_camera()

        # 交互状态
        self._dragging_uid: int | None = None
        self._drag_offset = (0.0, 0.0)
        self._drag_start_pos: tuple[float, float] | None = None
        self._drag_start_snapshot: dict | None = None
        self._drag_broken = False
        #: 拖动带键原子时，绕哪个邻居旋转（None = 没有键，走自由移动）
        self._drag_anchor_uid: int | None = None
        #: 旋转用的基准姿态（每次 move 都从它出发重算，避免误差累积）
        self._drag_rot_snapshot: dict[int, tuple[float, float]] = {}
        #: 当前吸附候选（松手时要用；也在 move 里决定是否进入断键状态）
        self._drag_snap: SnapCandidate | None = None
        self._orbiting = False
        self._panning = False
        self._last_mouse = QtCore.QPointF()
        self._press_pos: QtCore.QPointF | None = None
        self._bond_from_uid: int | None = None
        self._bond_from_slot: int = -1
        self._bond_cursor: tuple[float, float] | None = None
        self._marquee_start: QtCore.QPointF | None = None
        self._preview_atom: tuple[float, float] | None = None
        self._ctrl_click = False
        self.setAcceptDrops(True)        # 接收从原子盘拖来的元素

        # 立起来 / 回到平面的过渡动画
        self._anim_from: list[tuple[float, float, float]] = []
        self._anim_to: list[tuple[float, float, float]] = []
        self._anim_uids: list[int] = []
        self._anim_elapsed = 0.0
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._tick_anim)

        self._fps_timer = QtCore.QTimer(self)
        self._fps_timer.setInterval(600)
        self._fps_timer.timeout.connect(self._report_fps)
        self._fps_timer.start()

        self._msg_timer = QtCore.QTimer(self)
        self._msg_timer.setSingleShot(True)
        self._msg_timer.timeout.connect(lambda: self.message_changed.emit(""))

    # ------------------------------------------------------------------
    # 尺寸与坐标
    # ------------------------------------------------------------------

    def _sync_camera(self) -> None:
        dpr = self.devicePixelRatioF()
        self.camera.width = self.width()
        self.camera.height = self.height()
        self.camera.dpr = dpr

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:      # noqa: N802
        super().resizeEvent(event)
        self._sync_camera()

    def showEvent(self, event: QtGui.QShowEvent) -> None:          # noqa: N802
        super().showEvent(event)
        self._sync_camera()
        self._fit_view(animate=False)

    def _screen_pos(self, event) -> tuple[float, float]:
        p = event.position() if hasattr(event, "position") else event.localPos()
        return p.x(), p.y()

    # ------------------------------------------------------------------
    # 提示
    # ------------------------------------------------------------------

    def say(self, text: str, ms: int = 2600) -> None:
        self.message_changed.emit(text)
        self._msg_timer.start(ms)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:        # noqa: N802
        self._sync_camera()
        painter = QtGui.QPainter(self)
        try:
            self.renderer.paint(painter, self.world, self.camera,
                                self.devicePixelRatioF())
        finally:
            painter.end()

    def _report_fps(self) -> None:
        self.update()

    @property
    def fps(self) -> float:
        ms = self.renderer.stats.draw_ms
        return 1000.0 / ms if ms > 0.01 else 0.0

    # ------------------------------------------------------------------
    # 命中测试
    # ------------------------------------------------------------------

    def _atom_radius_px(self, el: str) -> float:
        info = ELEMENTS.get(el, {"radius": 0.5})
        factor = 1.0 if self.world.mode == "view3d" else 1.25
        return info["radius"] * factor * self.camera.screen_scale()

    def atom_at(self, sx: float, sy: float) -> int | None:
        """返回命中的原子 uid（取屏幕距离最近且在球内的）。"""
        best_uid = None
        best_d = 1e18
        for atom in self.world.atoms:
            if not self.renderer.show_hydrogens and atom.el == "H":
                continue
            pos = atom.pos3 if (self.world.mode == "view3d" and atom.pos3) else (atom.x, atom.y, 0.0)
            px, py, _d = self.camera.project(pos)
            r = self._atom_radius_px(atom.el)
            d = math.hypot(px - sx, py - sy)
            # 放宽 4px，小原子也好点中
            if d <= max(r, 7.0) + 4.0 and d < best_d:
                best_d = d
                best_uid = atom.uid
        return best_uid

    def slot_at(self, sx: float, sy: float) -> tuple[int, int] | None:
        """命中原子的**空插槽蓝点**，返回 (原子 uid, 插槽索引)；没命中返回 None。

        连键工具支持"按住蓝点拖到另一个蓝点"就靠它。
        命中半径比蓝点本身大一圈（+6 px），否则那么小的点很难点中。
        """
        best = None
        best_d = 1e18
        for atom in self.world.atoms:
            if not self.renderer.show_hydrogens and atom.el == "H":
                continue
            if self.world.free_valence(atom.uid) <= 0:
                continue
            for idx, (_d, point) in enumerate(self.world.free_slot_points(atom.uid)):
                px, py, _ = self.camera.project((point[0], point[1], 0.0))
                d = math.hypot(px - sx, py - sy)
                if d <= SLOT_PICK_RADIUS_PX and d < best_d:
                    best_d = d
                    best = (atom.uid, idx)
        return best

    def bond_at(self, sx: float, sy: float) -> int | None:
        """返回命中的键下标。"""
        best_idx = None
        best_d = 1e18
        for i, bond in enumerate(self.world.bonds):
            a = self.world.atom(bond.a)
            b = self.world.atom(bond.b)
            if a is None or b is None:
                continue
            pa = self.camera.project(a.pos3 if (self.world.mode == "view3d" and a.pos3) else (a.x, a.y, 0.0))
            pb = self.camera.project(b.pos3 if (self.world.mode == "view3d" and b.pos3) else (b.x, b.y, 0.0))
            d = _point_segment_dist(sx, sy, pa[0], pa[1], pb[0], pb[1])
            if d < 8.0 and d < best_d:
                best_d = d
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # 鼠标
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:    # noqa: N802
        self.setFocus()
        sx, sy = self._screen_pos(event)
        self._last_mouse = QtCore.QPointF(sx, sy)
        self._press_pos = QtCore.QPointF(sx, sy)
        button = event.button()
        mods = event.modifiers()
        self._ctrl_click = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)

        # ---- 中键：3D 旋转 / 2D 平移；Shift+中键 一律平移 ----
        if button == QtCore.Qt.MouseButton.MiddleButton:
            if (self.world.mode == "view3d"
                    and not (mods & QtCore.Qt.KeyboardModifier.ShiftModifier)):
                self._orbiting = True
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                self._panning = True
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            return

        # ---- 右键：交给主窗口弹上下文菜单 ----
        if button == QtCore.Qt.MouseButton.RightButton:
            uid = self.atom_at(sx, sy)
            if uid is not None and uid not in self.renderer.selected_uids:
                # 右键点在没选中的原子上 → 先选中它，菜单才作用于"这个原子"
                self.renderer.selected_uids = [uid]
                self.update()
            bi = None if uid is not None else self.bond_at(sx, sy)
            self.context_menu_requested.emit(QtCore.QPoint(int(sx), int(sy)),
                                             -1 if bi is None else bi)
            return

        uid = self.atom_at(sx, sy)

        # ---- 平移工具 ----
        if self.tool == "平移":
            self._panning = True
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            return

        # ---- 放置工具已移除：放原子请从左侧原子盘拖出来 ----
        # （保留这个分支只为了给出清晰的提示，避免玩家以为点空白坏了）

        # ---- 删除工具 ----
        if self.tool == "删除":
            if uid is not None:
                self._delete_atom(uid)
                return
            bi = self.bond_at(sx, sy)
            if bi is not None:
                self._delete_bond(bi)
                return
            self._marquee_start = QtCore.QPointF(sx, sy)
            return

        # ---- 连键工具：从原子（优先从蓝点）拖到目标 ----
        if self.tool == "连键":
            slot_hit = self.slot_at(sx, sy)
            if slot_hit is not None:
                self._bond_from_uid, self._bond_from_slot = slot_hit
                self.say("拖到另一个原子的蓝点上松手即可连键", 2200)
            elif uid is not None:
                self._bond_from_uid = uid
                self._bond_from_slot = -1
                self.say("拖到另一个原子（或它的蓝点）上松手即可连键", 2200)
            else:
                self.say("连键请从原子上的蓝点开始拖；想放原子请从左侧原子盘拖出来", 2800)
            if self._bond_from_uid is not None:
                wp = self.camera.screen_to_plane(sx, sy)
                self.renderer.bond_preview = (self._bond_from_uid, wp[0], wp[1], False)
                self.update()
            return

        # ---- 鼠标工具（默认）：选择 / 拖动 / 框选 ----
        if uid is not None:
            # 选中逻辑：Ctrl 点击切换选中；普通点击若不在选择里则改为只选它
            if self._ctrl_click:
                if uid in self.renderer.selected_uids:
                    self.renderer.selected_uids.remove(uid)
                else:
                    self.renderer.selected_uids.append(uid)
                self.update()
                return
            if uid not in self.renderer.selected_uids:
                self.renderer.selected_uids = [uid]

            if self.world.mode == "view3d":
                self._orbiting = True
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
                return

            atom = self.world.atom(uid)
            if atom is None:
                return
            # 记录起始状态：拖动可能"断键"或需要回退（带键原子），都需要它
            self._drag_start_snapshot = self.world.snapshot()
            self.world.push_undo()
            self._dragging_uid = uid
            self._drag_start_pos = (atom.x, atom.y)
            self._drag_broken = False
            self._drag_snap = None
            # 带键的原子走"绕锚点旋转"：锚点取第一个邻居（点住哪根键的哪一端）
            anchors = self.world.neighbors(uid)
            self._drag_anchor_uid = anchors[0] if anchors else None
            self._drag_rot_snapshot = self.world.snapshot_positions()
            wp = self.camera.screen_to_plane(sx, sy)
            self._drag_offset = (atom.x - wp[0], atom.y - wp[1])
            self.renderer.drag_uid = uid
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.world_changed.emit()
            self.update()
            return

        # 点空白：取消选择（**不再放原子**，这是本次修的重点）
        if self.world.mode == "view3d":
            self._orbiting = True
            return
        self.renderer.selected_uids = []
        self._marquee_start = QtCore.QPointF(sx, sy)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:     # noqa: N802
        sx, sy = self._screen_pos(event)
        dx = sx - self._last_mouse.x()
        dy = sy - self._last_mouse.y()
        self._last_mouse = QtCore.QPointF(sx, sy)

        if self._panning:
            self.camera.pan(dx, dy)
            self.update()
            return

        if self._orbiting:
            self.camera.orbit(dx, dy)
            self.update()
            return

        # ---- 拖动原子 ----
        # 三种情况：
        #   a) 有键 + 未超阈值 → 绕锚点**整体旋转**（只改方向，键长一丝不变）
        #   b) 超过阈值       → 标记为"将断键"，键画红虚线，松手才真断
        #   c) 没有键         → 自由跟手移动
        if self._dragging_uid is not None:
            wp = self.camera.screen_to_plane(sx, sy)
            desired = (wp[0] + self._drag_offset[0], wp[1] + self._drag_offset[1])
            atom = self.world.atom(self._dragging_uid)
            if atom is None:
                self._dragging_uid = None
                return
            # 鼠标从按下点移动了多少像素（决定要不要进入"绕锚旋转"）
            moved_px = (math.hypot(sx - self._press_pos.x(), sy - self._press_pos.y())
                        if self._press_pos is not None else 0.0)

            anchors = self._drag_anchor_neighbors()

            # 拖动意图判断：玩家是不是在**把两个相连的原子往一起拉**？
            # 旋转模式保持键长恒定，所以"拉近"这个动作在旋转下永远做不到 ——
            # 原子与锚点的距离恒为键长。要支持"拉近就升键"，
            # 就必须在这个手势下允许键长缩短、走自由移动。
            pulling_together = False
            if anchors and self._drag_anchor_uid is not None:
                anchor_a = self.world.atom(self._drag_anchor_uid)
                if anchor_a is not None:
                    cur_d = math.dist((atom.x, atom.y), (anchor_a.x, anchor_a.y))
                    want_d = math.dist(desired, (anchor_a.x, anchor_a.y))
                    if want_d < cur_d - 1e-6 and want_d <= SLOT_SNAP_RADIUS + 0.4:
                        pulling_together = True

            if self._drag_broken:
                # 已经进入「将断键」状态：让原子自己跟手走，好让玩家看清要拖到哪
                self.world.move_atom(self._dragging_uid, desired[0], desired[1])
            elif pulling_together:
                # 拉近相连的邻居：允许缩短，好触发升键预览
                self.world.move_atom(self._dragging_uid, desired[0], desired[1])
            elif anchors and moved_px >= DRAG_ROTATE_MIN_PX:
                # 有键：绕锚点旋转，键长恒定。
                # 锚点取"离目标最近"的那个邻居 —— 多键原子（比如碳）往哪侧拖，
                # 就绕那一侧的邻居转，玩家的意图能被正确理解。
                anchor_uid = self._drag_anchor_uid or anchors[0]
                if len(anchors) > 1:
                    anchor_uid = min(
                        anchors,
                        key=lambda u: math.dist(
                            (self.world.atom(u).x, self.world.atom(u).y), desired))
                anchor = self.world.atom(anchor_uid)
                if anchor is not None and self._drag_start_pos:
                    base = math.atan2(self._drag_start_pos[1] - anchor.y,
                                      self._drag_start_pos[0] - anchor.x)
                    want = math.atan2(desired[1] - anchor.y, desired[0] - anchor.x)
                    delta = want - base
                    # 先回到起始姿态再整体旋转，避免误差累积
                    self.world.restore_geometry_positions(self._drag_rot_snapshot)
                    self.world.rotate_group(self._dragging_uid, anchor_uid, delta)
            else:
                # 没有键：自由移动
                self.world.move_atom(self._dragging_uid, desired[0], desired[1])

            atom = self.world.atom(self._dragging_uid)

            # 断键阈值必须按**鼠标实际拖了多远**来量，不能按原子位移量。
            #
            # 原因：带键的原子是绕锚点旋转的，它与锚点的距离恒定不变 ——
            # 无论拖多远，原子到起点的距离永远 ≤ 键长，阈值就永远触发不了。
            # （实测：把原子从 (1,0) 绕原点拖到 (4,0) 的方向，
            #   它其实停在 (1,0)，位移还是 0，键怎么也断不了。）
            # 用光标位移做判据则三种模式都自洽：
            #   自由移动时 光标位移 = 原子位移；旋转时它就是玩家"想拖多远"的意图。
            moved_from_start = (math.hypot(sx - self._press_pos.x(), sy - self._press_pos.y())
                                / max(1e-6, self.camera.screen_scale())
                                if self._press_pos is not None else 0.0)

            # 拖回阈值内 → **取消**断键状态，恢复成绕锚旋转
            if self._drag_broken and moved_from_start <= DRAG_BREAK_DIST:
                self._drag_broken = False
                self.renderer.breaking_uid = None
                if anchors and self._drag_anchor_uid is not None:
                    self.world.restore_geometry_positions(self._drag_rot_snapshot)
                    anchor = self.world.atom(self._drag_anchor_uid)
                    if anchor is not None and self._drag_start_pos:
                        base = math.atan2(self._drag_start_pos[1] - anchor.y,
                                          self._drag_start_pos[0] - anchor.x)
                        want = math.atan2(desired[1] - anchor.y, desired[0] - anchor.x)
                        self.world.rotate_group(self._dragging_uid,
                                                self._drag_anchor_uid, want - base)
                else:
                    self.world.restore_geometry_positions(self._drag_rot_snapshot)
                self.say("已取消断键", 1600)

            # 超过断键阈值 → 标记"将断键"（松手才真正断开）。
            # **只对带键的原子生效**：没有键就没有可断的东西，
            # 而且这条分支会把原子从"自由移动"抢成"跟手走"，
            # 结果是一个孤立原子被拖去远处连键时（距离超过 1.85 键长很常见）
            # 反而永远吸不上 —— 实测就是这样丢掉了拖到插槽吸附的能力。
            if (anchors and not self._drag_broken and self._drag_start_pos
                    and moved_from_start > DRAG_BREAK_DIST
                    and not self._drag_snap):
                self.world.restore_geometry_positions(self._drag_rot_snapshot)
                self._drag_broken = True
                self.world.move_atom(self._dragging_uid, desired[0], desired[1])
                if self.world.neighbors(self._dragging_uid):
                    self.say("松手即断开这些键（拖回阈值内可取消）", 2600)

            # 吸附预览：只在没进入断键状态时给。
            # allow_bonded_host=True 让"把两个已相连的原子拖到一起"能升键；
            # World 那边只在**确实比开始时更近**的情况下才给升键候选，
            # 所以普通平移不会误升键。
            cand = None
            if not self._drag_broken:
                cand = self.world.find_snap_candidate(
                    self._dragging_uid, allow_bonded_host=True,
                    pull_from=self._drag_start_pos)
            self._drag_snap = cand
            self.renderer.preview = cand
            self.renderer.breaking_uid = self._dragging_uid if self._drag_broken else None
            if cand is not None:
                if cand.upgrade_order:
                    if cand.feasible:
                        self.say(f"松手可将这根键升为"
                                 f"{'双键' if cand.upgrade_order == 2 else '三键'}", 1400)
                    else:
                        self.say(cand.reason, 1800)
                elif cand.feasible:
                    host = self.world.atom(cand.host_uid)
                    hname = ELEMENTS[cand.host_el]["name"] if host else ""
                    self.say(f"松手即可与{hname}连成单键", 1200)
                else:
                    self.say(cand.reason, 1600)
            self.world_changed.emit()
            self.update()
            return

        # ---- 连键工具：实时虚线预览 ----
        if self._bond_from_uid is not None:
            src = self._bond_from_uid
            target = self.atom_at(sx, sy)
            self.renderer.hover_atom_uid = target
            _tgt, end, feasible, _cand = self._bond_target(src, sx, sy)
            self.renderer.bond_preview = (src, end[0], end[1], feasible)
            self.update()
            return

        # ---- 框选 ----
        if self._marquee_start is not None:
            x0, y0 = self._marquee_start.x(), self._marquee_start.y()
            self.renderer.marquee = (min(x0, sx), min(y0, sy), abs(sx - x0), abs(sy - y0))
            self.update()
            return

        # ---- 悬停高亮 ----
        uid = self.atom_at(sx, sy)
        self.renderer.hover_atom_uid = uid
        if uid is None:
            self.renderer.hover_bond_index = self.bond_at(sx, sy)
        else:
            self.renderer.hover_bond_index = None

        if self.tool in ("鼠标", "连键") and self.world.mode == "build" and uid is None:
            self._preview_atom = self.camera.screen_to_plane(sx, sy)
        else:
            self._preview_atom = None
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        sx, sy = self._screen_pos(event)

        if self._panning:
            self._panning = False
            self._restore_cursor()
            return
        if self._orbiting:
            self._orbiting = False
            self._restore_cursor()
            return

        # ---- 拖动结束：吸附 / 断键 / 保留旋转后的姿态 ----
        if self._dragging_uid is not None:
            uid = self._dragging_uid
            # 一定要用**拖动过程中记下的那个候选**：松手时重新算会按默认参数
            # 调用（不允许已相连的宿主），把升键候选和吸附目标都丢掉。
            cand = self._drag_snap
            broken = self._drag_broken
            start_pos = self._drag_start_pos
            atom = self.world.atom(uid)

            self._dragging_uid = None
            self._drag_anchor_uid = None
            self._drag_rot_snapshot = {}
            self._drag_snap = None
            self.renderer.drag_uid = None
            self.renderer.preview = None
            self.renderer.breaking_uid = None
            self._drag_broken = False
            self._drag_start_pos = None

            if broken:
                # 真正把键断掉。
                # 注意：拖动过程中只标记 `_drag_broken`（好让键显示成红色虚线、
                # 并允许玩家拖回阈值内取消），**不能在拖动中就断键** ——
                # 否则"拖回取消"就没法还原了。
                n = self.world.break_all_bonds(uid)
                self.say(f"已断开 {n} 根键，原子停在当前位置（Ctrl+Z 可恢复）", 3000)
            elif cand is not None and cand.upgrade_order:
                # 把两个已相连的原子拖到一起 = 升键
                if cand.feasible:
                    self.bump_bond_order_at_hover_by_atoms(uid, cand.host_uid,
                                                           cand.upgrade_order)
                else:
                    self._restore_drag(uid, start_pos)
                    self.say(cand.reason, 2800)
            elif cand is not None and cand.feasible:
                ok, msg = self.world.apply_snap(uid, cand)
                if ok:
                    host = self.world.atom(cand.host_uid)
                    hname = ELEMENTS[cand.host_el]["name"] if host else ""
                    self.say(f"已与{hname}成键", 2000)
                else:
                    self.say(msg, 2600)
            elif cand is not None and not cand.feasible:
                # 目标插槽不够 → 提示原因。
                # 带键的原子退回原位（它只是旋转，不能凭空换位置）；
                # 没键的原子按「拖到哪放到哪」，留在原地。
                if bool(self.world.neighbors(uid)):
                    self._restore_drag(uid, start_pos)
                self.say(cand.reason, 2600)
            else:
                # 没有吸附、也没断键：
                #   有键的原子 → 保留旋转后的姿态（玩家要的就是改方向）
                #   没键的原子 → **拖到哪放到哪**（v5 起不再恢复原位）
                has_bonds = bool(self.world.neighbors(uid))
                if not has_bonds and atom is not None and start_pos is not None:
                    moved = math.dist((atom.x, atom.y), start_pos)
                    if moved <= DRAG_RESTORE_EPS and self._drag_start_snapshot is not None:
                        # 原地按下松开 = 什么都没改 → 弹掉这次压入的撤销点，
                        # 别让 Ctrl+Z 看起来像「没反应」
                        self.world.pop_last_undo_if_unchanged(self._drag_start_snapshot)

            self._drag_start_snapshot = None
            self._restore_cursor()
            self.world_changed.emit()
            self.update()
            return

        # ---- 连键工具：松手时连键并摆正到标准键长 ----
        if self._bond_from_uid is not None:
            src = self._bond_from_uid
            slot = self._bond_from_slot
            self._bond_from_uid = None
            self._bond_from_slot = -1
            self._bond_cursor = None
            self.renderer.bond_preview = None

            target, _end, feasible, cand = self._bond_target(src, sx, sy)
            if target is None or target == src:
                self.say("连键请拖到另一个原子（或它的蓝点）上松手", 2400)
                self.update()
                return
            if not feasible:
                self.say((cand.reason if cand is not None else "")
                         or "这个位置连不上：对方价键已满", 2800)
                self.update()
                return

            self.world.push_undo()
            if self.world.bond_index(src, target) >= 0:
                # 已经相连：连键工具是**明确的**成键意图，
                # 所以直接升键，不要求"拖得更近"（那是鼠标拖动的判据）。
                self.bump_bond_order_at_hover_by_atoms(src, target, 0)
            else:
                ok, msg, moved = self.world.connect_atoms(src, target, 1)
                if ok:
                    self.say("已连成单键（两端已摆到标准键长；右键可改键级）", 2600)
                else:
                    self.say(msg, 3000)
            self.world_changed.emit()
            self.update()
            return

        # ---- 框选结束 ----
        if self._marquee_start is not None:
            rect = self.renderer.marquee
            start = self._marquee_start
            self._marquee_start = None
            self.renderer.marquee = None
            if rect and rect[2] >= MARQUEE_MIN_PX and rect[3] >= MARQUEE_MIN_PX:
                self._apply_marquee(rect)
            elif start is not None:
                # 位移太小 → 当成"点空白"，选择已在 press 时清空，这里不用再做事
                pass
            self.update()

    def _drag_anchor_neighbors(self) -> list[int]:
        """拖动中的原子当前连着哪些邻居（用来判断能不能"绕锚旋转"）。"""
        if self._dragging_uid is None:
            return []
        return self.world.neighbors(self._dragging_uid)

    # ------------------------------------------------------------------
    # 从原子盘拖出原子
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:      # noqa: N802
        if event.mimeData().hasFormat(ELEMENT_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:        # noqa: N802
        if not event.mimeData().hasFormat(ELEMENT_MIME):
            event.ignore()
            return
        el = bytes(event.mimeData().data(ELEMENT_MIME)).decode("utf-8")
        if el not in ELEMENTS:
            event.ignore()
            return
        pos = event.position() if hasattr(event, "position") else event.posF()
        wp = self.camera.screen_to_plane(pos.x(), pos.y())
        self.renderer.palette_ghost = (el, wp[0], wp[1], self._palette_drop_ok(el, wp))
        self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:      # noqa: N802
        self.renderer.palette_ghost = None
        self.update()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:                # noqa: N802
        self.renderer.palette_ghost = None
        if not event.mimeData().hasFormat(ELEMENT_MIME):
            event.ignore()
            return
        el = bytes(event.mimeData().data(ELEMENT_MIME)).decode("utf-8")
        if el not in ELEMENTS:
            event.ignore()
            return
        if self.world.mode == "view3d":
            self.say("3D 模式下不能改结构，点「重新编辑」回到平面", 2600)
            event.ignore()
            return

        pos = event.position() if hasattr(event, "position") else event.posF()
        wp = self.camera.screen_to_plane(pos.x(), pos.y())

        # 落点靠近某个空插槽 → 直接吸附成键（这是"拖出来就能连上"的关键）
        cand = self._palette_candidate(wp)
        if cand is not None and not cand.feasible:
            self.say(cand.reason, 2800)
            event.ignore()
            return

        self.world.push_undo()
        if cand is not None:
            # 先放原子，再把它吸附到插槽点上
            ok, msg, atom = self.world.create_atom(el, cand.point[0], cand.point[1])
            if not ok or atom is None:
                self.say(msg, 2600)
                event.ignore()
                return
            bok, bmsg = self.world.add_bond(atom.uid, cand.host_uid, 1)
            if bok:
                host = self.world.atom(cand.host_uid)
                hname = ELEMENTS[host.el]["name"] if host else ""
                self.say(f"{ELEMENTS[el]['name']}已连到{hname}上", 2400)
            else:
                self.say(bmsg, 2800)
        else:
            ok, msg, _atom = self.world.create_atom(el, wp[0], wp[1])
            if not ok:
                self.say(msg, 2600)
                event.ignore()
                return
            self.say(f"放了 1 个{ELEMENTS[el]['name']}原子", 2000)

        self.selected_element = el
        self.world_changed.emit()
        self.update()
        event.acceptProposedAction()

    def _palette_candidate(self, wp: tuple[float, float]):
        """给"从原子盘拖出的落点"找一个插槽候选。

        借用 World.find_snap_candidate：临时建一个不可见的探针原子来问它。
        直接建又删会污染 uid 与撤销历史，所以这里手写一份轻量匹配。
        """
        from ..chem.constants import SLOT_SNAP_RADIUS
        best = None
        best_d = 1e18
        for host in self.world.atoms:
            if self.world.free_valence(host.uid) <= 0:
                continue
            for idx, (_d, point) in enumerate(self.world.free_slot_points(host.uid)):
                d = math.hypot(point[0] - wp[0], point[1] - wp[1])
                if d <= SLOT_SNAP_RADIUS and d < best_d:
                    best_d = d
                    best = (host, idx, point)
        if best is None:
            # 没有空插槽候选：看看是不是落在某个已满原子上（要是就拒绝）
            for host in self.world.atoms:
                if math.hypot(host.x - wp[0], host.y - wp[1]) <= 0.45:
                    if self.world.free_valence(host.uid) <= 0:
                        info = ELEMENTS.get(host.el, {})
                        return SnapCandidate(
                            host_uid=host.uid, host_el=host.el, point=wp,
                            direction=(0.0, 0.0), distance=0.0, angle_deviation=0.0,
                            feasible=False,
                            reason=f"{info.get('name', host.el)}的价键已用满，放不下新原子",
                        )
            return None
        host, _idx, point = best
        return SnapCandidate(
            host_uid=host.uid, host_el=host.el, point=point,
            direction=(0.0, 0.0), distance=best_d, angle_deviation=0.0,
            feasible=True, slot_index=_idx,
        )

    def _palette_drop_ok(self, el: str, wp: tuple[float, float]) -> bool:
        cand = self._palette_candidate(wp)
        return cand is None or cand.feasible

    def _restore_drag(self, uid: int, start_pos: tuple[float, float] | None) -> None:
        """把原子放回拖动前的位置，并弹掉这次拖动压入的撤销点。

        现在只用于**带键原子**的两种失败回退（吸附到已满宿主、升键失败）：
        它只是旋转未果，退回原位不产生变化，撤销点要一起弹掉。
        没键的原子不再走这里 —— 拖到哪放到哪（v5）。
        """
        if start_pos is not None:
            atom = self.world.atom(uid)
            if atom is not None:
                atom.x, atom.y = start_pos
        if self.world.undo_depth() and self._drag_start_snapshot is not None:
            # 只有当栈顶就是本次拖动压入的那条时才弹，避免误删别人的撤销点
            self.world.pop_last_undo_if_unchanged(self._drag_start_snapshot)
        self.world.invalidate_geometry()

    def leaveEvent(self, event) -> None:                            # noqa: N802
        self.renderer.hover_atom_uid = None
        self.renderer.hover_bond_index = None
        self._preview_atom = None
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:         # noqa: N802
        sx, sy = self._screen_pos(event)
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.camera.zoom_at(1.12 if delta > 0 else 1 / 1.12, sx, sy)
        self.update()

    def _restore_cursor(self) -> None:
        if self.tool == "平移":
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        elif self.tool == "删除":
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------
    # 编辑动作
    # ------------------------------------------------------------------

    def _place_atom(self, sx: float, sy: float) -> None:
        if self.world.mode == "view3d":
            self.say("3D 模式下不能改结构，点「重新编辑」回到平面", 2400)
            return
        wp = self.camera.screen_to_plane(sx, sy)
        self.world.push_undo()
        ok, msg, atom = self.world.create_atom(self.selected_element, wp[0], wp[1])
        if not ok:
            self.say(msg, 2600)
            return
        self.say(f"放了 1 个{ELEMENTS[self.selected_element]['name']}原子"
                 f"（拖动它靠近别的原子可连键）", 2200)
        self.world_changed.emit()
        self.update()

    def _delete_atom(self, uid: int) -> None:
        atom = self.world.atom(uid)
        if atom is None:
            return
        name = ELEMENTS[atom.el]["name"]
        self.world.push_undo()
        self.world.remove_atom(uid)
        self.say(f"已删除{name}原子（连带它的化学键）", 2000)
        self.world_changed.emit()
        self.update()

    def _delete_bond(self, index: int) -> None:
        self.world.push_undo()
        bond = self.world.bonds[index]
        self.world.bonds.pop(index)
        self.world.invalidate_geometry()
        a = self.world.atom(bond.a)
        b = self.world.atom(bond.b)
        if a and b:
            self.say(f"已删除 {a.el}—{b.el} 键", 2000)
        self.world_changed.emit()
        self.update()

    def _atoms_in_rect(self, rect: tuple[float, float, float, float]) -> list[int]:
        """矩形（屏幕坐标）内的原子 uid 列表。"""
        x, y, w, h = rect
        hit = []
        for atom in self.world.atoms:
            if not self.renderer.show_hydrogens and atom.el == "H":
                continue
            pos = atom.pos3 if (self.world.mode == "view3d" and atom.pos3) else (atom.x, atom.y, 0.0)
            px, py, _ = self.camera.project(pos)
            if x <= px <= x + w and y <= py <= y + h:
                hit.append(atom.uid)
        return hit

    def _apply_marquee(self, rect: tuple[float, float, float, float]) -> None:
        """框选：鼠标工具下是「选中」，删除工具下是「删除」。

        这是 v3 的改动点 —— 鼠标模式的框选不再删原子，
        而是把框内的原子加入选择集合（Ctrl 状态决定是否追加）。
        """
        hit = self._atoms_in_rect(rect)
        if self.tool == "删除":
            self.world.push_undo()
            for uid in hit:
                self.world.remove_atom(uid)
            if hit:
                self.say(f"框选删除了 {len(hit)} 个原子", 2200)
            self.world_changed.emit()
            return

        if self._ctrl_click:
            for uid in hit:
                if uid not in self.renderer.selected_uids:
                    self.renderer.selected_uids.append(uid)
        else:
            self.renderer.selected_uids = list(hit)
        if hit:
            self.say(f"已选中 {len(self.renderer.selected_uids)} 个原子"
                     f"（X 删除，右键出菜单）", 2600)
        else:
            self.renderer.selected_uids = []
        self.world_changed.emit()

    def delete_selection(self) -> int:
        """删除当前选中的原子，返回删除数量。供 X 键与右键菜单调用。"""
        uids = list(self.renderer.selected_uids)
        if not uids:
            return 0
        self.world.push_undo()
        removed = 0
        for uid in uids:
            if self.world.remove_atom(uid):
                removed += 1
        self.renderer.selected_uids = []
        if removed:
            self.say(f"已删除所选的 {removed} 个原子", 2200)
        self.world_changed.emit()
        self.update()
        return removed

    def bond_selection(self) -> tuple[bool, str]:
        """把选中的两个原子连成单键。供右键菜单调用。"""
        uids = list(self.renderer.selected_uids)
        if len(uids) != 2:
            return False, "请先选中正好两个原子再成键"
        self.world.push_undo()
        ok, msg = self.world.add_bond(uids[0], uids[1], 1)
        if ok:
            self.say("已连成单键（在键上右键可切换键级）", 2400)
        else:
            self.say(msg, 3000)
        self.world_changed.emit()
        self.update()
        return ok, msg

    def _bond_target(self, src_uid: int, sx: float, sy: float
                     ) -> tuple[int | None, tuple[float, float], bool, SnapCandidate | None]:
        """连键工具的落点解析：返回 (目标原子, 预览终点, 是否可行, 吸附候选)。

        **move 与 release 共用这一份逻辑**，否则会出现
        「预览说可行、松手却没连上」这种最让人困惑的不一致。

        顺序很关键：先看光标下有没有别的原子，有就以它为目标；
        只有在没命中原子时才回头看空插槽。
        反过来的话，从某个原子自己的槽位起拖时会命中它自己的槽点，
        目标被解析成"起点自己"，于是变成一次自连、什么也不做 ——
        实测就是这样导致"拖到已相连的原子上不升键"。
        """
        wp = self.camera.screen_to_plane(sx, sy)
        hover = self.atom_at(sx, sy)
        if hover is not None and hover != src_uid:
            existing = self.world.bond_index(src_uid, hover)
            if existing >= 0:
                cur = self.world.bonds[existing].order
                if cur < 3:
                    ok, _m, _r = self.world.free_slots_for_order(existing, cur + 1)
                    host = self.world.atom(hover)
                    cand = SnapCandidate(
                        host_uid=hover, host_el=host.el if host else "",
                        point=(host.x, host.y) if host else wp,
                        direction=(0.0, 0.0), distance=0.0, angle_deviation=0.0,
                        feasible=ok, reason="" if ok else "插槽不够，无法升键",
                        slot_index=-1, upgrade_order=cur + 1)
                    return hover, cand.point, ok, cand
                return hover, wp, False, None
            tgt = self.world.atom(hover)
            if tgt is not None and self.world.free_valence(hover) > 0:
                dirs = self.world.free_slot_directions(hover)
                src = self.world.atom(src_uid)
                if dirs and src is not None:
                    rel = normalize2(src.x - tgt.x, src.y - tgt.y)
                    best = max(dirs, key=lambda d: d[0] * rel[0] + d[1] * rel[1])
                    return hover, (tgt.x + best[0], tgt.y + best[1]), True, None
            return hover, wp, False, None

        # 没命中别的原子 → 看空插槽
        slot_hit = self.slot_at(sx, sy)
        if slot_hit is not None and slot_hit[0] != src_uid:
            tgt = self.world.atom(slot_hit[0])
            if tgt is not None:
                pts = self.world.free_slot_points(tgt.uid)
                if 0 <= slot_hit[1] < len(pts):
                    return slot_hit[0], pts[slot_hit[1]][1], True, None
        return None, wp, False, None

    def bump_bond_order_at_hover_by_atoms(self, a_uid: int, b_uid: int,
                                         target_order: int = 0) -> bool:
        """把 a、b 之间的键升到 target_order（0 = 自动升一级）。供菜单与拖动升键共用。"""
        bi = self.world.bond_index(a_uid, b_uid)
        if bi < 0:
            return False
        cur = self.world.bonds[bi].order
        want = target_order or min(3, cur + 1)
        if want <= cur:
            return False
        self.world.push_undo()
        ok, msg = self.world.set_bond_order(bi, want)
        if ok:
            order = self.world.bonds[bi].order
            name = {1: "单键", 2: "双键", 3: "三键"}[order]
            extra = ""
            if self.world.last_removed_hydrogens:
                extra = f"（自动移除了 {len(self.world.last_removed_hydrogens)} 个氢腾出插槽）"
            self.say(f"键级已升为{name}{extra}", 2800)
        else:
            self.say(msg, 3200)
        self.world_changed.emit()
        self.update()
        return ok

    def bump_bond_order_at_hover(self, delta: int) -> bool:
        """把鼠标指向的那根键升/降一级（快捷键 +/-）。

        左键不再承担改键级（双击已移除），改键级统一走：
        右键菜单，或者把鼠标悬在键上按 + / -。
        """
        bi = self.renderer.hover_bond_index
        if bi is None:
            self.say("把鼠标移到一根键上，再按 + / - 改键级", 2400)
            return False
        self.world.push_undo()
        cur = self.world.bonds[bi].order
        target = max(1, min(3, cur + delta))
        ok, msg = self.world.set_bond_order(bi, target)
        if ok:
            order = self.world.bonds[bi].order
            name = {1: "单键", 2: "双键", 3: "三键"}[order]
            extra = ""
            if self.world.last_removed_hydrogens:
                extra = f"（自动移除了 {len(self.world.last_removed_hydrogens)} 个氢腾出插槽）"
            self.say(f"键级改为{name}{extra}", 2600)
        else:
            self.say(msg, 3000)
        self.world_changed.emit()
        self.update()
        return ok

    # ------------------------------------------------------------------
    # 立起来 / 回到平面
    # ------------------------------------------------------------------

    def stand_up(self) -> None:
        if self.world.is_empty():
            self.say("先放几个原子吧", 2400)
            return
        if self.world.mode == "view3d":
            return
        fill = self.world.auto_fill_hydrogens()
        if self.world.atom_count > RELAX_MAX_ATOMS:
            self.say(f"原子太多（{self.world.atom_count} 个），超过 {RELAX_MAX_ATOMS} 个后不再自动算构型", 4200)
            return
        self.say("正在按真实键角计算立体构型…", 1500)
        QtWidgets.QApplication.processEvents()

        before_undo = self.world._snapshot()
        ok, msg, radius = self.world.solve_geometry()
        if not ok:
            self.say(msg, 4000)
            return
        self.world._undo.append(before_undo)

        # 动画：从编辑平面插值到求解结果
        self._anim_uids = [a.uid for a in self.world.atoms]
        self._anim_from = []
        for atom in self.world.atoms:
            a = self.world.atom(atom.uid)
            self._anim_from.append((a.x, -a.y, 0.0))
        self._anim_to = [a.pos3 or (0.0, 0.0, 0.0) for a in self.world.atoms]
        self._anim_elapsed = 0.0
        self.camera.reset(top_down=False)
        self.camera.fit(radius, VIEW_FIT_MARGIN)
        self._anim_timer.start()

        note = f"，自动补了 {fill['added']} 个氢" if fill.get("added") else ""
        self.say(f"已按真实键角立起来{note}，拖动可旋转视角", 4000)
        self.world_changed.emit()
        self.update()

    def back_to_edit(self) -> None:
        self.world.mode = "build"
        for atom in self.world.atoms:
            atom.pos3 = None
        self._anim_timer.stop()
        self.camera.reset(top_down=True)
        self.camera.pan_x = self.camera.pan_y = 0.0
        self._fit_view(animate=False)
        self.renderer.preview = None
        self.say("回到平面拼装（原来的布局没有被打乱）", 2600)
        self.world_changed.emit()
        self.update()

    def _tick_anim(self) -> None:
        self._anim_elapsed += self._anim_timer.interval()
        t = min(1.0, self._anim_elapsed / max(1, VIEW_TRANSITION_MS))
        k = ease_in_out_cubic(t)
        for i, uid in enumerate(self._anim_uids):
            atom = self.world.atom(uid)
            if atom is None:
                continue
            f = self._anim_from[i]
            to = self._anim_to[i]
            atom.pos3 = (f[0] + (to[0] - f[0]) * k,
                         f[1] + (to[1] - f[1]) * k,
                         f[2] + (to[2] - f[2]) * k)
        if t >= 1.0:
            self._anim_timer.stop()
            for i, uid in enumerate(self._anim_uids):
                atom = self.world.atom(uid)
                if atom is not None:
                    atom.pos3 = self._anim_to[i]
            self._fit_view(animate=True)
        self.update()

    def _fit_view(self, animate: bool = True) -> None:
        if not self.world.atoms:
            return
        radius = SceneRenderer.content_radius(self.world)
        if self.world.mode == "view3d":
            self.camera.fit(radius, VIEW_FIT_MARGIN)
        else:
            # 平面模式下按编辑范围缩放，居中显示
            xs = [a.x for a in self.world.atoms]
            ys = [a.y for a in self.world.atoms]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 1.5
            self.camera.fit(span / 2.0, 1.1)
            cx = (max(xs) + min(xs)) / 2.0
            cy = (max(ys) + min(ys)) / 2.0
            s = self.camera.screen_scale()
            self.camera.pan_x = -cx * s
            self.camera.pan_y = cy * s
        self.update()

    # ------------------------------------------------------------------
    # 键盘
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:        # noqa: N802
        key = event.key()
        text = event.text().upper()

        if key == QtCore.Qt.Key.Key_Escape:
            if self._bond_from_uid is not None:
                self._bond_from_uid = None
                self._bond_cursor = None
                self.say("已取消连键", 1600)
                self.update()
                return
            self.renderer.selected_uids = []
            self.update()
            return

        # 工具快捷键：V 鼠标 / B 连键 / D 删除 / Space 平移
        if text in ("V", "B"):
            self.set_tool(TOOL_KEYS[text])
            return
        if key == QtCore.Qt.Key.Key_Space:
            self.set_tool("平移")
            return
        if text == "D":
            self.set_tool("删除")
            return
        if text == "X" or key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            # 优先删选中的；没有选中就删鼠标指向的那个
            if self.renderer.selected_uids:
                self.delete_selection()
            else:
                uid = self.renderer.hover_atom_uid
                if uid is not None:
                    self._delete_atom(uid)
            return
        # +/- 改键级：鼠标悬在哪根键上就改哪根（左键不再承担改键级）
        if text in ("+", "=") or key == QtCore.Qt.Key.Key_Plus:
            self.bump_bond_order_at_hover(+1)
            return
        if text in ("-", "_") or key == QtCore.Qt.Key.Key_Minus:
            self.bump_bond_order_at_hover(-1)
            return
        if text == "H":
            self.renderer.show_hydrogens = not self.renderer.show_hydrogens
            self.say("显示氢原子" if self.renderer.show_hydrogens else "隐藏氢原子", 1600)
            self.update()
            return
        if text == "G":
            self.renderer.show_grid = not self.renderer.show_grid
            self.update()
            return
        super().keyPressEvent(event)

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            return
        self.tool = tool
        cursors = {
            "鼠标": QtCore.Qt.CursorShape.ArrowCursor,
            "连键": QtCore.Qt.CursorShape.PointingHandCursor,
            "删除": QtCore.Qt.CursorShape.PointingHandCursor,
            "平移": QtCore.Qt.CursorShape.OpenHandCursor,
        }
        self.setCursor(cursors.get(tool, QtCore.Qt.CursorShape.CrossCursor))
        self.say(TOOL_HINTS.get(tool, ""), 3200)
        self.update()


def _point_segment_dist(px: float, py: float, x1: float, y1: float,
                        x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
