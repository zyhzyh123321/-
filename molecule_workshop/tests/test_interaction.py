"""交互与界面回归测试。

覆盖四块：
    A. 空插槽几何 —— 「氢原子互相重叠」与「吸附到已满碳」的修复
    B. 画布交互   —— 鼠标模式不再放原子、选择/多选/框选、X 删除所选、放置工具
    C. 拖动与断键 —— 超阈值断键、自由原子拖到哪放到哪、撤销/重做
    D. 界面        —— 窄窗口自适应、分子选择器、撤销重做按钮状态

Qt 相关测试用 offscreen 平台，必须在导入 PySide6 **之前**设置环境变量；
并且显式加载中文字体，否则离屏环境里所有文字都是方块。

运行：
    python -m pytest molecule_workshop/tests/test_interaction.py -q -p no:cacheprovider
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

# ---- 必须在导入 PySide6 之前设置离屏平台 ----
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molecule_workshop.chem.constants import (  # noqa: E402
    DRAG_BREAK_DIST,
    SLOT_MIN_SEPARATION,
    SLOT_SNAP_RADIUS,
)
from molecule_workshop.world import SnapCandidate, World  # noqa: E402


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    from molecule_workshop.ui.theme import load_fonts
    load_fonts()
    return app


@pytest.fixture
def canvas(qapp):
    from molecule_workshop.render.canvas import CanvasWidget
    world = World()
    widget = CanvasWidget(world)
    widget.resize(900, 640)
    widget.show()
    qapp.processEvents()
    return widget


# ----------------------------------------------------------------------------
# 事件构造帮手
# ----------------------------------------------------------------------------


def _mouse(kind, x, y, button=None, buttons=None, mods=None):
    from PySide6 import QtCore, QtGui
    if button is None:
        button = QtCore.Qt.MouseButton.LeftButton
    if buttons is None:
        buttons = button if kind != QtCore.QEvent.Type.MouseMove else QtCore.Qt.MouseButton.NoButton
    if mods is None:
        mods = QtCore.Qt.KeyboardModifier.NoModifier
    pt = QtCore.QPointF(x, y)
    return QtGui.QMouseEvent(kind, pt, pt, button, buttons, mods)


def click(widget, x, y, mods=None, button=None):
    from PySide6 import QtCore
    b = button or QtCore.Qt.MouseButton.LeftButton
    widget.mousePressEvent(_mouse(QtCore.QEvent.Type.MouseButtonPress, x, y, b, b, mods))
    widget.mouseReleaseEvent(_mouse(QtCore.QEvent.Type.MouseButtonRelease, x, y, b, b, mods))


def press_at(canvas, x, y, button=None):
    from PySide6 import QtCore
    b = button or QtCore.Qt.MouseButton.LeftButton
    canvas.mousePressEvent(_mouse(QtCore.QEvent.Type.MouseButtonPress, x, y, b, b))


def move_to(canvas, x, y):
    from PySide6 import QtCore
    canvas.mouseMoveEvent(_mouse(QtCore.QEvent.Type.MouseMove, x, y,
                                 QtCore.Qt.MouseButton.LeftButton))


def release_at(canvas, x, y, button=None):
    from PySide6 import QtCore
    b = button or QtCore.Qt.MouseButton.LeftButton
    canvas.mouseReleaseEvent(_mouse(QtCore.QEvent.Type.MouseButtonRelease, x, y,
                                    b, QtCore.Qt.MouseButton.NoButton))


def drag(widget, x0, y0, x1, y1, steps=3):
    from PySide6 import QtCore
    b = QtCore.Qt.MouseButton.LeftButton
    widget.mousePressEvent(_mouse(QtCore.QEvent.Type.MouseButtonPress, x0, y0, b, b))
    for i in range(1, steps + 1):
        t = i / steps
        widget.mouseMoveEvent(_mouse(QtCore.QEvent.Type.MouseMove,
                                     x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, b))
    widget.mouseReleaseEvent(_mouse(QtCore.QEvent.Type.MouseButtonRelease, x1, y1, b, b))


def key(widget, qt_key, text=""):
    from PySide6 import QtCore, QtGui
    ev = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, qt_key,
                         QtCore.Qt.KeyboardModifier.NoModifier, text)
    widget.keyPressEvent(ev)


def screen_of(canvas, atom_or_x, y=None):
    """把世界坐标（或原子）换算成屏幕坐标。"""
    if y is None:
        x = atom_or_x.x
        y = atom_or_x.y
    else:
        x = atom_or_x
    sx, sy, _ = canvas.camera.project((x, y, 0.0))
    return sx, sy


def drop_element(canvas, symbol, x, y):
    """模拟「从原子盘拖一个原子到画布」的 drop 事件。"""
    from PySide6 import QtCore, QtGui
    from molecule_workshop.render.canvas import ELEMENT_MIME
    mime = QtCore.QMimeData()
    mime.setData(ELEMENT_MIME, symbol.encode("utf-8"))
    canvas.dropEvent(QtGui.QDropEvent(
        QtCore.QPointF(x, y), QtCore.Qt.DropAction.CopyAction, mime,
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QEvent.Type.Drop))


def drag_bond(canvas, src_uid, dst_uid, steps=3):
    """模拟连键工具：从 src 的某个空插槽拖到 dst 的球上。"""
    from PySide6 import QtCore
    w = canvas.world
    _d, point = w.free_slot_points(src_uid)[0]
    sx, sy = screen_of(canvas, point[0], point[1])
    dst = w.atom(dst_uid)
    ex, ey = screen_of(canvas, dst)
    canvas.mousePressEvent(_mouse(QtCore.QEvent.Type.MouseButtonPress, sx, sy,
                                  QtCore.Qt.MouseButton.LeftButton,
                                  QtCore.Qt.MouseButton.LeftButton))
    for i in range(1, steps + 1):
        t = i / steps
        canvas.mouseMoveEvent(_mouse(QtCore.QEvent.Type.MouseMove,
                                     sx + (ex - sx) * t, sy + (ey - sy) * t,
                                     QtCore.Qt.MouseButton.LeftButton))
    canvas.mouseReleaseEvent(_mouse(QtCore.QEvent.Type.MouseButtonRelease, ex, ey,
                                    QtCore.Qt.MouseButton.LeftButton,
                                    QtCore.Qt.MouseButton.NoButton))


def angle_between(d1, d2) -> float:
    c = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
    return math.degrees(math.acos(c))


# ============================================================================
# A. 空插槽几何
# ============================================================================


def test_lone_carbon_has_four_slots_at_90_degrees():
    """单个碳有 4 个空插槽，间距 90°（十字形）。

    这是修「氢原子互相重叠」的核心参数：一律用 60° 时 4 个空位刚好占满
    一圈（360/60 = 6 个位置里的 4 个），相邻碳上的空位就会落到同一处。
    """
    w = World()
    c = w.create_atom("C", 0, 0)[2]
    dirs = w.free_slot_directions(c.uid)
    assert len(dirs) == 4
    for i in range(len(dirs)):
        gap = angle_between(dirs[i], dirs[(i + 1) % len(dirs)])
        assert abs(gap - 90.0) < 1e-6, f"第 {i} 个夹角是 {gap}°，应为 90°"


def test_lone_oxygen_has_two_opposite_slots():
    """裸氧 2 个空插槽，成 180° 对开。"""
    w = World()
    o = w.create_atom("O", 0, 0)[2]
    dirs = w.free_slot_directions(o.uid)
    assert len(dirs) == 2
    assert abs(angle_between(dirs[0], dirs[1]) - 180.0) < 1e-6


def test_saturated_carbon_has_no_slots():
    """满价碳（4 根键）没有空插槽。"""
    w = World()
    c = w.create_atom("C", 0, 0)[2]
    for i in range(4):
        h = w.create_atom("H", math.cos(i * math.pi / 2), math.sin(i * math.pi / 2))[2]
        assert w.add_bond(c.uid, h.uid, 1)[0]
    assert w.free_slot_directions(c.uid) == []


def test_ethane_carbon_slots_are_evenly_spread():
    """乙烷的碳（1 根键）有 3 个空插槽，两两 120°。

    只有一根键时最宽空档几乎就是整圈；若按"空档中点 ± 60°"排布，
    3 个插槽会挤在 120° 范围内，相邻碳上的落点又会撞在一起。
    """
    w = World()
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    w.add_bond(a.uid, b.uid, 1)
    dirs = w.free_slot_directions(a.uid)
    assert len(dirs) == 3
    gaps = [angle_between(dirs[i], dirs[(i + 1) % 3]) for i in range(3)]
    assert all(abs(g - 120.0) < 1e-6 for g in gaps), f"夹角应为 120°，实得 {gaps}"


def test_adjacent_carbon_slots_do_not_coincide():
    """相邻两个碳的空插槽落点**不能重合**（玩家报的重叠问题）。

    未做跨原子避让时，两边的空位落点会完全重合（距离 0.000 键长），
    玩家拖过去分不清要连哪一个。
    """
    w = World()
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    w.add_bond(a.uid, b.uid, 1)
    pts_a = [pt for _d, pt in w.free_slot_points(a.uid)]
    pts_b = [pt for _d, pt in w.free_slot_points(b.uid)]
    assert pts_a and pts_b
    nearest = min(math.dist(p, q) for p in pts_a for q in pts_b)
    assert nearest > 0.5, f"两侧空位最近距离只有 {nearest:.3f} 键长，会重叠"


def test_slot_min_separation_respected_in_chain():
    """丙烷链上所有空插槽落点两两不能挨得太近。"""
    w = World()
    u = [w.create_atom("C", i * 1.5, 0)[2].uid for i in range(3)]
    w.add_bond(u[0], u[1], 1)
    w.add_bond(u[1], u[2], 1)
    pts = []
    for uid in u:
        pts.extend(pt for _d, pt in w.free_slot_points(uid))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.dist(pts[i], pts[j])
            assert d > 0.3, f"两个插槽落点相距仅 {d:.3f} 键长"


def test_feasible_host_beats_full_host():
    """探针落到未满碳的空位上时，必须吸附到**未满**的碳。

    这是「拖到指定位置放不下去、被识别成连到了已满的碳」的直接回归测试。
    排序键里 feasible 必须在第一位，否则距离更近的已满碳会抢走吸附。
    """
    w = World()
    full = w.create_atom("C", 0, 0)[2]
    for i in range(4):
        h = w.create_atom("H", math.cos(i * math.pi / 2), math.sin(i * math.pi / 2))[2]
        w.add_bond(full.uid, h.uid, 1)
    free_c = w.create_atom("C", 2.0, 0)[2]

    for _d, pt in w.free_slot_points(free_c.uid):
        probe = w.create_atom("C", pt[0], pt[1])[2]
        cand = w.find_snap_candidate(probe.uid)
        assert cand is not None, f"空位 {pt} 上居然没有候选"
        assert cand.host_uid == free_c.uid, (
            f"空位 {pt} 的候选吸附到了 uid={cand.host_uid}（已满是 {full.uid}）")
        assert cand.feasible
        w.remove_atom(probe.uid)


def test_snap_candidate_reports_infeasible_with_reason():
    """拖到已满原子的空档方向时，候选存在但标记为不可行并给出原因。"""
    w = World()
    c = w.create_atom("C", 0, 0)[2]
    for i in range(4):
        h = w.create_atom("H", math.cos(i * math.pi / 2), math.sin(i * math.pi / 2))[2]
        w.add_bond(c.uid, h.uid, 1)
    # 45° 方向是"最宽空档"的中点（四个氢分别占 0/90/180/270）
    probe = w.create_atom("C", 0.6, 0.6)[2]
    cand = w.find_snap_candidate(probe.uid)
    assert cand is not None
    assert cand.feasible is False
    assert "用满" in cand.reason


def test_slot_index_matches_returned_slot():
    """候选里的 slot_index 必须指向真正命中的那个插槽（界面靠它高亮）。"""
    w = World()
    host = w.create_atom("C", 0, 0)[2]
    slots = w.free_slot_points(host.uid)
    for idx, (_d, pt) in enumerate(slots):
        probe = w.create_atom("C", pt[0], pt[1])[2]
        cand = w.find_snap_candidate(probe.uid)
        assert cand is not None
        assert cand.slot_index == idx, f"落点 {pt} 应命中插槽 {idx}，实得 {cand.slot_index}"
        w.remove_atom(probe.uid)


# ============================================================================
# B. 画布交互
# ============================================================================


def test_mouse_tool_click_on_empty_does_not_place_atom(canvas):
    """鼠标模式点空白**只放**——什么都不放，原子数不变。

    注意点的位置要真的落在空白处：画布中心放着原子时（半径约 23 px），
    点中心会命中那个原子而不是空白。
    """
    click(canvas, 60, 60)
    assert canvas.world.atom_count == 0


def test_mouse_tool_click_empty_clears_selection(canvas):
    w = canvas.world
    atom = w.create_atom("C", 0, 0)[2]
    canvas.renderer.selected_uids = [atom.uid]
    click(canvas, 60, 60)            # 远离中心那个原子，确保是空白
    assert canvas.renderer.selected_uids == []


def test_drop_element_creates_atom(canvas):
    """从原子盘拖到画布 → 在落点创建一个原子（v4 起唯一的放置方式）。"""
    drop_element(canvas, "C", 300, 260)
    assert canvas.world.atom_count == 1
    assert canvas.world.atoms[0].el == "C"
    drop_element(canvas, "O", 520, 260)
    assert canvas.world.atom_count == 2
    assert canvas.world.atoms[1].el == "O"


def test_drop_on_free_slot_bonds_immediately(canvas):
    """拖到某个空插槽上 → 直接吸附成键，且键长就是标准键长。"""
    w = canvas.world
    host = w.create_atom("C", 0.0, 0.0)[2]
    canvas._fit_view(animate=False)
    _d, point = w.free_slot_points(host.uid)[0]
    sx, sy = screen_of(canvas, point[0], point[1])
    drop_element(canvas, "C", sx, sy)
    assert w.atom_count == 2
    assert w.bond_count == 1
    mover = w.atoms[1]
    assert abs(w.bond_length_of(host.uid, mover.uid) - 1.0) < 1e-6


def test_drop_on_saturated_atom_is_rejected(canvas):
    """落在已满原子的球上 → 拒绝放置，原子数不变。"""
    w = canvas.world
    full = w.create_atom("C", 0.0, 0.0)[2]
    for i in range(4):
        h = w.create_atom("H", math.cos(i * math.pi / 2), math.sin(i * math.pi / 2))[2]
        w.add_bond(full.uid, h.uid, 1)
    canvas._fit_view(animate=False)
    sx, sy = screen_of(canvas, full.x, full.y)
    drop_element(canvas, "C", sx, sy)
    assert w.atom_count == 5, "已满原子上不该再放原子"


def test_drop_ignores_non_element_mime(canvas):
    """只认元素 MIME，别的拖放（比如文本）一律忽略。"""
    from PySide6 import QtCore, QtGui
    mime = QtCore.QMimeData()
    mime.setText("hello")
    canvas.dropEvent(QtGui.QDropEvent(
        QtCore.QPointF(300, 260), QtCore.Qt.DropAction.CopyAction, mime,
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QEvent.Type.Drop))
    assert canvas.world.atom_count == 0


def test_mouse_tool_click_atom_selects_it(canvas):
    w = canvas.world
    atom = w.create_atom("C", 0, 0)[2]
    canvas._fit_view(animate=False)
    sx, sy = screen_of(canvas, w.atom(atom.uid))
    click(canvas, sx, sy)
    assert canvas.renderer.selected_uids == [atom.uid]


def test_ctrl_click_multi_selects_and_toggles(canvas):
    """Ctrl+点击：加入选择；再 Ctrl+点击同一个则移出。"""
    from PySide6 import QtCore
    w = canvas.world
    a = w.create_atom("C", -1.0, 0)[2]
    b = w.create_atom("C", 1.0, 0)[2]
    canvas._fit_view(animate=False)
    mods = QtCore.Qt.KeyboardModifier.ControlModifier

    click(canvas, *screen_of(canvas, w.atom(a.uid)))
    click(canvas, *screen_of(canvas, w.atom(b.uid)), mods=mods)
    assert set(canvas.renderer.selected_uids) == {a.uid, b.uid}

    click(canvas, *screen_of(canvas, w.atom(b.uid)), mods=mods)
    assert canvas.renderer.selected_uids == [a.uid]


def test_marquee_selects_atoms_inside(canvas):
    """左键拖空白 = 框选（不再删原子）。"""
    w = canvas.world
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    c = w.create_atom("C", 8.0, 0)[2]
    canvas._fit_view(animate=False)
    ax, ay = screen_of(canvas, w.atom(a.uid))
    bx, by = screen_of(canvas, w.atom(b.uid))
    drag(canvas, ax - 40, ay - 60, bx + 40, by + 60)
    assert set(canvas.renderer.selected_uids) == {a.uid, b.uid}
    assert c.uid not in canvas.renderer.selected_uids
    assert w.atom_count == 3, "框选不应该删原子"


def test_x_key_deletes_selection(canvas):
    """X 删除选中的原子（不再是"切到删除工具"）。"""
    from PySide6 import QtCore
    w = canvas.world
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas.renderer.selected_uids = [a.uid]
    key(canvas, QtCore.Qt.Key.Key_X, "x")
    assert w.atom_count == 1
    assert w.atom(b.uid) is not None
    assert w.bond_count == 0, "删原子应连带删键"


def test_mouse_tool_click_on_atom_does_not_add_another(canvas):
    """鼠标模式点在已有原子上只做选择，不会叠一个新原子上去。"""
    w = canvas.world
    atom = w.create_atom("C", 0, 0)[2]
    canvas.set_tool("鼠标")
    canvas._fit_view(animate=False)
    click(canvas, *screen_of(canvas, w.atom(atom.uid)))
    assert w.atom_count == 1
    assert canvas.renderer.selected_uids == [atom.uid]


def test_right_click_emits_context_menu_request(canvas):
    """右键发出上下文菜单请求，并且会先选中点到的原子。"""
    from PySide6 import QtCore
    w = canvas.world
    atom = w.create_atom("C", 0, 0)[2]
    canvas._fit_view(animate=False)
    got = []
    canvas.context_menu_requested.connect(lambda pos, bi: got.append((pos, bi)))
    sx, sy = screen_of(canvas, w.atom(atom.uid))
    canvas.mousePressEvent(_mouse(QtCore.QEvent.Type.MouseButtonPress, sx, sy,
                                  QtCore.Qt.MouseButton.RightButton,
                                  QtCore.Qt.MouseButton.RightButton))
    assert len(got) == 1
    assert got[0][1] == -1
    assert canvas.renderer.selected_uids == [atom.uid]


# ============================================================================
# C. 拖动与断键
# ============================================================================


def test_drag_beyond_threshold_breaks_all_bonds(canvas):
    """拖得够远 → 该原子身上的键全部断开，原子停在当前位置。"""
    w = canvas.world
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    ax, ay = screen_of(canvas, w.atom(a.uid))
    scale = canvas.camera.screen_scale()
    far = DRAG_BREAK_DIST + 0.5
    drag(canvas, ax, ay, ax - far * scale, ay)
    assert w.bond_count == 0, "超过阈值应断键"
    assert w.atom_count == 2, "断键不该删原子"
    assert w.atom(a.uid) is not None


def test_drag_beyond_threshold_keeps_atom_at_new_position(canvas):
    """断键后原子停在拖到的位置（不弹回原处）。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    ax, ay = screen_of(canvas, w.atom(a.uid))
    scale = canvas.camera.screen_scale()
    far = DRAG_BREAK_DIST + 0.6
    drag(canvas, ax, ay, ax - far * scale, ay)
    moved = w.atom(a.uid)
    assert moved.x < -DRAG_BREAK_DIST, f"原子应停在远处，实得 x={moved.x:.2f}"


def test_small_drag_of_bonded_atom_rotates_and_keeps_length(canvas):
    """带键原子小幅拖动（未达阈值）→ 绕邻居旋转：只改方向，键长不变。

    v4 的改动点：早先这里是把原子「恢复原位」，玩家没法微调方向；
    现在改为刚性旋转 —— 键长由几何保证恒定。
    """
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    length_before = w.bond_length_of(a.uid, b.uid)
    bx, by = screen_of(canvas, w.atom(b.uid))
    scale = canvas.camera.screen_scale()
    drag(canvas, bx, by, bx, by - 0.4 * scale)       # 往垂直方向拖，远小于阈值

    assert abs(w.bond_length_of(a.uid, b.uid) - length_before) < 1e-9, "键长必须不变"
    assert w.bond_count == 1, "未超阈值不该断键"
    anchor = w.atom(a.uid)
    assert abs(anchor.x) < 1e-9 and abs(anchor.y) < 1e-9, "锚点不该移动"
    moved = w.atom(b.uid)
    assert abs(moved.y) > 0.1, f"方向应该变了，实得 y={moved.y:.3f}"


def test_free_atom_stays_where_dropped(canvas):
    """**没有键**的原子：拖到哪放到哪，松手就停在那里（v5 起不再回位）。"""
    w = canvas.world
    lone = w.create_atom("C", 0.0, 0.0)[2]
    canvas._fit_view(animate=False)
    lx, ly = screen_of(canvas, w.atom(lone.uid))
    scale = canvas.camera.screen_scale()
    drag(canvas, lx, ly, lx + 0.6 * scale, ly)
    moved = w.atom(lone.uid)
    assert abs(moved.x - 0.6) < 0.02, \
        f"孤立原子应停在拖到的位置（x≈0.6），实得 ({moved.x:.3f}, {moved.y:.3f})"


def test_free_atom_drag_leaves_undo_entry(canvas):
    """孤立原子拖到新位置后**要**留下撤销点（位置确实变了）。"""
    w = canvas.world
    lone = w.create_atom("C", 0.0, 0.0)[2]
    w.clear_history()
    canvas._fit_view(animate=False)
    lx, ly = screen_of(canvas, w.atom(lone.uid))
    scale = canvas.camera.screen_scale()
    drag(canvas, lx, ly, lx + 0.6 * scale, ly)
    assert w.undo_depth() == 1, f"真实拖动应留下一个撤销点，实得 {w.undo_depth()}"


def test_free_atom_click_without_move_leaves_no_undo_entry(canvas):
    """原地按下松开（没移动）= 什么都没改 → 不留撤销点。"""
    w = canvas.world
    lone = w.create_atom("C", 0.0, 0.0)[2]
    w.clear_history()
    canvas._fit_view(animate=False)
    lx, ly = screen_of(canvas, w.atom(lone.uid))
    press_at(canvas, lx, ly)
    release_at(canvas, lx, ly)
    moved = w.atom(lone.uid)
    assert abs(moved.x) < 1e-9 and abs(moved.y) < 1e-9, "位置不该变"
    assert w.undo_depth() == 0, f"原地点击不该留撤销点，实得 {w.undo_depth()}"


def test_bonded_small_drag_keeps_undo_entry(canvas):
    """带键原子旋转后**要**留下撤销点（方向确实变了，不是空操作）。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.clear_history()
    canvas._fit_view(animate=False)
    bx, by = screen_of(canvas, w.atom(b.uid))
    scale = canvas.camera.screen_scale()
    drag(canvas, bx, by, bx, by - 0.4 * scale)
    assert w.undo_depth() == 1, f"旋转应留下一个撤销点，实得 {w.undo_depth()}"


def test_break_then_undo_restores_bonds_and_position(canvas):
    """断键后一次撤销 → 键和位置都回到拖动前。"""
    w = canvas.world
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.clear_history()
    canvas._fit_view(animate=False)
    ax, ay = screen_of(canvas, w.atom(a.uid))
    scale = canvas.camera.screen_scale()
    far = DRAG_BREAK_DIST + 0.5
    drag(canvas, ax, ay, ax - far * scale, ay)
    assert w.bond_count == 0

    assert w.undo() is True
    assert w.bond_count == 1, "撤销应恢复键"
    restored = w.atom(a.uid)
    assert abs(restored.x - (-0.5)) < 1e-6, f"撤销应恢复位置，实得 x={restored.x:.3f}"


def test_break_then_redo_breaks_again(canvas):
    """断键 → 撤销 → 重做，结果应再次断开。"""
    w = canvas.world
    a = w.create_atom("C", -0.5, 0)[2]
    b = w.create_atom("C", 0.5, 0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.clear_history()
    canvas._fit_view(animate=False)
    ax, ay = screen_of(canvas, w.atom(a.uid))
    scale = canvas.camera.screen_scale()
    drag(canvas, ax, ay, ax - (DRAG_BREAK_DIST + 0.5) * scale, ay)
    w.undo()
    assert w.bond_count == 1
    assert w.redo() is True
    assert w.bond_count == 0, "重做应再次断开"


def test_drag_near_slot_snaps_and_bonds(canvas):
    """拖到别的原子空插槽附近松手 → 精确吸附并成键。"""
    w = canvas.world
    host = w.create_atom("C", 0.0, 0.0)[2]
    mover = w.create_atom("C", 3.0, 0.0)[2]
    canvas._fit_view(animate=False)
    scale = canvas.camera.screen_scale()
    _d, target = w.free_slot_points(host.uid)[0]
    mx, my = screen_of(canvas, w.atom(mover.uid))
    tx, ty = canvas.camera.project((target[0], target[1], 0.0))[:2]
    drag(canvas, mx, my, tx, ty)
    assert w.bond_count == 1, "应吸附成键"
    pos = w.atom(mover.uid)
    assert math.dist((pos.x, pos.y), target) < 0.02, "应精确落在插槽点上"


# ============================================================================
# C2. 连键工具（v4：蓝点起拖 + 实时预览 + 自动摆正）
# ============================================================================


def test_link_tool_press_on_slot_records_source(canvas):
    """按住原子上的蓝点开始拖 → 记录起点与槽位索引。"""
    w = canvas.world
    a = w.create_atom("C", -1.5, 0.0)[2]
    w.create_atom("C", 1.5, 0.0)[2]
    canvas._fit_view(animate=False)
    canvas.set_tool("连键")
    _d, point = w.free_slot_points(a.uid)[0]
    sx, sy = screen_of(canvas, point[0], point[1])
    press_at(canvas, sx, sy)
    assert canvas._bond_from_uid == a.uid
    assert canvas._bond_from_slot == 0


def test_link_tool_shows_live_preview(canvas):
    """拖动过程中必须有预览（早先 `_bond_cursor` 被赋值却从没画出来）。"""
    w = canvas.world
    a = w.create_atom("C", -1.5, 0.0)[2]
    b = w.create_atom("C", 1.5, 0.0)[2]
    canvas._fit_view(animate=False)
    canvas.set_tool("连键")
    _d, point = w.free_slot_points(a.uid)[0]
    sx, sy = screen_of(canvas, point[0], point[1])
    press_at(canvas, sx, sy)
    bx, by = screen_of(canvas, w.atom(b.uid))
    move_to(canvas, bx, by)
    pv = canvas.renderer.bond_preview
    assert pv is not None, "应该有预览"
    assert pv[0] == a.uid
    assert pv[3] is True, "落在有空插槽的目标上应判定可行"


def test_link_tool_normalises_bond_length(canvas):
    """松手后两端自动摆到**标准键长**（修「距离异常也能连上」）。"""
    w = canvas.world
    a = w.create_atom("C", -2.4, 0.0)[2]
    b = w.create_atom("C", 2.4, 0.0)[2]
    canvas._fit_view(animate=False)
    canvas.set_tool("连键")
    drag_bond(canvas, a.uid, b.uid)
    assert w.bond_count == 1, "应该连上了"
    assert abs(w.bond_length_of(a.uid, b.uid) - 1.0) < 1e-6, \
        f"键长应被摆正到 1.0，实得 {w.bond_length_of(a.uid, b.uid):.4f}"


def test_link_tool_refuses_saturated_target(canvas):
    """目标原子插槽已满 → 不成键，也不挪动任何原子。"""
    w = canvas.world
    a = w.create_atom("C", -2.5, 0.0)[2]
    full = w.create_atom("C", 2.5, 0.0)[2]
    for i in range(4):
        h = w.create_atom("H", 2.5 + math.cos(i * math.pi / 2) * 0.9,
                          0.0 + math.sin(i * math.pi / 2) * 0.9)[2]
        w.add_bond(full.uid, h.uid, 1)
    canvas._fit_view(animate=False)
    canvas.set_tool("连键")
    before = (w.atom(a.uid).x, w.atom(a.uid).y)
    drag_bond(canvas, a.uid, full.uid)
    assert w.bond_count == 4, "不该新增键"
    assert (w.atom(a.uid).x, w.atom(a.uid).y) == before, "被拖的原子不该被挪走"


def test_link_tool_to_bonded_atom_upgrades_order(canvas):
    """连键拖到**已相连**的原子上 → 升键（单 → 双）。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    canvas.set_tool("连键")
    drag_bond(canvas, a.uid, b.uid)
    assert w.bonds[w.bond_index(a.uid, b.uid)].order == 2, "应升为双键"


# ============================================================================
# C3. 拖动带键原子（v4：绕锚旋转 / 超阈值标红 / 拖回取消）
# ============================================================================


def test_rotation_keeps_all_bond_lengths(canvas):
    """绕锚旋转时**所有**键长都不变（含支链内部）。"""
    w = canvas.world
    c = w.create_atom("C", 0.0, 0.0)[2]
    o = w.create_atom("O", 1.0, 0.0)[2]
    h = w.create_atom("H", 1.0, 1.0)[2]
    w.add_bond(c.uid, o.uid, 1)
    w.add_bond(o.uid, h.uid, 1)
    canvas._fit_view(animate=False)
    before = (w.bond_length_of(c.uid, o.uid), w.bond_length_of(o.uid, h.uid))

    ox, oy = screen_of(canvas, w.atom(o.uid))
    scale = canvas.camera.screen_scale()
    drag(canvas, ox, oy, ox, oy - 0.5 * scale)

    after = (w.bond_length_of(c.uid, o.uid), w.bond_length_of(o.uid, h.uid))
    assert abs(after[0] - before[0]) < 1e-9, "C—O 键长不该变"
    assert abs(after[1] - before[1]) < 1e-9, "O—H 键长不该变（支链整体走）"


def test_drag_over_threshold_marks_break_before_release(canvas):
    """拖过阈值时**先标红**，此时键还没断 —— 松手才断。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    cx, cy = screen_of(canvas, w.atom(b.uid))
    far_x, far_y = screen_of(canvas, 4.0, 0.0)
    press_at(canvas, cx, cy)
    move_to(canvas, far_x, far_y)
    assert w.bond_count == 1, "未松手时键不该断"
    assert canvas.renderer.breaking_uid == b.uid, "应处于将断键状态"
    release_at(canvas, far_x, far_y)
    assert w.bond_count == 0, "松手后才断"


def test_drag_back_within_threshold_cancels_break(canvas):
    """拖出去再拖回阈值内 → 取消断键，键长恢复标准值。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    cx, cy = screen_of(canvas, w.atom(b.uid))
    press_at(canvas, cx, cy)
    far_x, far_y = screen_of(canvas, 4.0, 0.0)
    move_to(canvas, far_x, far_y)
    assert canvas.renderer.breaking_uid == b.uid
    back_x, back_y = screen_of(canvas, 0.6, 0.6)
    move_to(canvas, back_x, back_y)
    assert canvas.renderer.breaking_uid is None, "拖回后应取消断键"
    assert abs(w.bond_length_of(a.uid, b.uid) - 1.0) < 1e-9, "键长应恢复"
    release_at(canvas, back_x, back_y)
    assert w.bond_count == 1, "松手后键还在"


def test_pulling_bonded_atoms_together_upgrades(canvas):
    """把两个已相连的原子**拖到一起** → 升键。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    cx, cy = screen_of(canvas, w.atom(b.uid))
    press_at(canvas, cx, cy)
    nx, ny = screen_of(canvas, 0.75, 0.0)
    move_to(canvas, nx, ny)
    cand = canvas._drag_snap
    assert cand is not None and cand.upgrade_order == 2, "应给出升键候选"
    release_at(canvas, nx, ny)
    assert w.bonds[w.bond_index(a.uid, b.uid)].order == 2, "应升为双键"


def test_plain_move_does_not_accidentally_upgrade(canvas):
    """普通平移（没有拉近）不应冒出升键候选。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    cx, cy = screen_of(canvas, w.atom(b.uid))
    press_at(canvas, cx, cy)
    # 往**远离** a 的方向拖（角度变化，距离不缩短）
    nx, ny = screen_of(canvas, 0.866, 0.5)
    move_to(canvas, nx, ny)
    cand = canvas._drag_snap
    assert cand is None or cand.upgrade_order == 0, \
        f"不该给升键候选，实得 {getattr(cand, 'upgrade_order', None)}"
    release_at(canvas, nx, ny)
    assert w.bonds[w.bond_index(a.uid, b.uid)].order == 1, "键级不该变"


# ============================================================================
# C4. 键级：双键 / 三键（v4 修「根本做不出双键」）
# ============================================================================


def test_ethane_to_ethylene_by_upgrading_bond():
    """乙烷（补满氢）升双键 → 自动脱掉 2 个氢，识别为乙烯。

    这是最常见的操作路径，也曾经完全走不通：
    补满氢后两个碳的插槽都是 0，升键被化合价校验直接拒绝。
    """
    w = World()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.auto_fill_hydrogens()
    assert w.atom_count == 8, "先确认是乙烷"

    bi = w.bond_index(a.uid, b.uid)
    ok, msg = w.cycle_bond_order(bi)
    assert ok, f"升双键应成功，实得 {msg!r}"
    assert w.bonds[bi].order == 2
    assert len(w.last_removed_hydrogens) == 2, "应自动脱掉 2 个氢"
    assert w.atom_count == 6, f"应是乙烯 6 个原子，实得 {w.atom_count}"

    parts, _ = w.identify()
    assert parts[0].matched is not None
    assert parts[0].matched["names"][0] == "乙烯"
    assert parts[0].formula == "C2H4"


def test_ethylene_to_acetylene_by_upgrading_bond():
    """再升三键 → 乙炔。"""
    w = World()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.auto_fill_hydrogens()
    bi = w.bond_index(a.uid, b.uid)
    w.cycle_bond_order(bi)
    ok, msg = w.cycle_bond_order(bi)
    assert ok, f"升三键应成功，实得 {msg!r}"
    assert w.bonds[bi].order == 3
    assert w.atom_count == 4, f"应是乙炔 4 个原子，实得 {w.atom_count}"
    parts, _ = w.identify()
    assert parts[0].matched["names"][0] == "乙炔"
    assert parts[0].formula == "C2H2"


def test_upgrade_never_removes_manual_hydrogens():
    """手动放的氢**不会**被自动移除；腾不出插槽时明确报错。"""
    w = World()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    for i in range(3):
        h = w.create_atom("H", -1.5, i - 1.0)[2]
        w.add_bond(a.uid, h.uid, 1)
    for i in range(4):
        h = w.create_atom("H", 1.5, i - 1.5)[2]
        w.add_bond(b.uid, h.uid, 1)
    before_count = w.atom_count

    bi = w.bond_index(a.uid, b.uid)
    ok, msg = w.cycle_bond_order(bi)
    assert not ok, "插槽不够时应拒绝"
    assert "插槽" in msg, f"错误信息要说清原因，实得 {msg!r}"
    assert w.atom_count == before_count, "手动氢一个都不能删"
    assert w.bonds[bi].order == 1, "键级不该被改动"


def test_downgrade_does_not_auto_add_hydrogen():
    """降键不自动补氢（避免玩家没意识到结构被改了）。"""
    w = World()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.auto_fill_hydrogens()
    bi = w.bond_index(a.uid, b.uid)
    w.cycle_bond_order(bi)                 # 单 → 双，脱 2 个氢
    assert w.atom_count == 6
    w.cycle_bond_order(bi)                 # 双 → 三，再脱 2 个
    assert w.atom_count == 4
    w.set_bond_order(bi, 1)                # 三 → 单
    assert w.bonds[bi].order == 1
    assert w.atom_count == 4, "降键不该自动补氢"


def test_bump_bond_order_at_hover_uses_hovered_bond(canvas):
    """+ / - 快捷键：改的是鼠标指向的那根键。"""
    w = canvas.world
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    # 中点就是键所在的位置
    mx, my = screen_of(canvas, 0.0, 0.0)
    canvas.renderer.hover_bond_index = canvas.bond_at(mx, my)
    assert canvas.renderer.hover_bond_index is not None, "应命中键"
    assert canvas.bump_bond_order_at_hover(+1) is True
    assert w.bonds[0].order == 2


def _h_dirs(w, uid) -> list[float]:
    """某个原子上所有氢相对它的方向（度）。"""
    host = w.atom(uid)
    out = []
    for n in w.neighbors(uid):
        other = w.atom(n)
        if other is not None and other.el == "H":
            out.append(math.degrees(math.atan2(other.y - host.y, other.x - host.x)))
    return sorted(out)


def _axis_dir(w, uid, other_uid) -> float:
    a, b = w.atom(uid), w.atom(other_uid)
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def test_hydrogens_follow_hybridisation_on_upgrade():
    """升键后氢要跟着杂化态重摆：乙炔共线、乙烯 ±120°。

    这是「乙炔画出来是折线」的修复：补氢时氢的位置是按当时的插槽算的，
    键级一变插槽方向就变了，氢却留在原地。实测残留方向 -32° 和 86°，
    四个原子完全看不出共线。
    """
    # --- 乙炔：sp，氢应与 C≡C 共线 ---
    w = World()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.auto_fill_hydrogens()
    bi = w.bond_index(a.uid, b.uid)
    w.set_bond_order(bi, 3)
    assert w.atom_count == 4, "应是乙炔"
    axis_ab = _axis_dir(w, a.uid, b.uid)
    axis_ba = _axis_dir(w, b.uid, a.uid)
    ha = _h_dirs(w, a.uid)
    hb = _h_dirs(w, b.uid)
    assert len(ha) == 1 and len(hb) == 1
    # a 上的氢应与 a→b 相反，b 上的氢应与 b→a 相反（即都朝外、四原子共线）
    assert abs(abs(((ha[0] - axis_ab + 180) % 360) - 180) - 180) < 1.0, \
        f"乙炔左端 C-H 应共线，实测 {ha[0]:.1f}° vs 轴 {axis_ab:.1f}°"
    assert abs(abs(((hb[0] - axis_ba + 180) % 360) - 180) - 180) < 1.0, \
        f"乙炔右端 C-H 应共线，实测 {hb[0]:.1f}° vs 轴 {axis_ba:.1f}°"

    # --- 乙烯：sp²，两个氢与 C=C 各成 120° ---
    w2 = World()
    a2 = w2.create_atom("C", -0.6, 0.0)[2]
    b2 = w2.create_atom("C", 0.6, 0.0)[2]
    w2.add_bond(a2.uid, b2.uid, 1)
    w2.auto_fill_hydrogens()
    bi2 = w2.bond_index(a2.uid, b2.uid)
    w2.set_bond_order(bi2, 2)
    assert w2.atom_count == 6, "应是乙烯"
    axis = _axis_dir(w2, a2.uid, b2.uid)
    ha2 = _h_dirs(w2, a2.uid)
    assert len(ha2) == 2, f"乙烯每个碳带 2 个氢，实测 {len(ha2)}"
    for h in ha2:
        delta = abs(((h - axis + 180) % 360) - 180)
        assert abs(delta - 120.0) < 1.0, \
            f"乙烯 C-H 应与 C=C 成 120°，实测 {delta:.1f}°"


def test_double_click_no_longer_changes_bond_order(canvas):
    """双击不再改键级（左键只保留拖动/选择/框选）。"""
    from PySide6 import QtCore
    w = canvas.world
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    canvas._fit_view(animate=False)
    mx, my = screen_of(canvas, 0.0, 0.0)
    canvas.mouseDoubleClickEvent(_mouse(QtCore.QEvent.Type.MouseButtonDblClick, mx, my,
                                        QtCore.Qt.MouseButton.LeftButton,
                                        QtCore.Qt.MouseButton.LeftButton))
    assert w.bonds[0].order == 1, "双击不该改键级"





def test_undo_redo_depth_tracking():
    w = World()
    for i in range(3):
        w.push_undo()
        w.create_atom("C", i * 2.0, 0)
    assert w.undo_depth() == 3
    assert w.can_undo() and not w.can_redo()
    w.undo()
    w.undo()
    assert w.undo_depth() == 1 and w.redo_depth() == 2
    assert w.can_redo()
    w.redo()
    assert w.undo_depth() == 2 and w.redo_depth() == 1


def test_redo_cleared_by_new_action():
    """重做后再做新操作 → 重做栈清空（标准语义）。"""
    w = World()
    w.push_undo()
    w.create_atom("C", 0, 0)
    w.undo()
    assert w.can_redo()
    w.push_undo()
    w.create_atom("O", 2, 0)
    assert not w.can_redo(), "新操作应清空重做栈"


def test_break_all_bonds_reports_count():
    w = World()
    c = w.create_atom("C", 0, 0)[2]
    for i in range(3):
        h = w.create_atom("H", math.cos(i * 2), math.sin(i * 2))[2]
        w.add_bond(c.uid, h.uid, 1)
    assert w.break_all_bonds(c.uid) == 3
    assert w.bond_count == 0
    assert w.atom_count == 4, "断键不该删原子"


def test_break_all_bonds_handles_multiple_orders():
    """双键也要一起断掉（按确认的行为：断开该原子全部键）。"""
    w = World()
    a = w.create_atom("C", 0, 0)[2]
    b = w.create_atom("C", 1.0, 0)[2]
    h1 = w.create_atom("H", -1.0, 0)[2]
    w.add_bond(a.uid, b.uid, 2)
    w.add_bond(a.uid, h1.uid, 1)
    assert w.break_all_bonds(a.uid) == 2
    assert w.bond_count == 0


# ============================================================================
# E. 资料面板：逐个查看分子
# ============================================================================


def test_empty_canvas_hint_says_drag_out(canvas):
    """空画布中央提示应为「拖出原子放置」，不再让玩家去点画布空白。"""
    import inspect
    from molecule_workshop.render.scene import SceneRenderer
    src = inspect.getsource(SceneRenderer._paint_hud)
    assert "拖出原子放置" in src, "中央提示应是新文案"
    assert "点左侧元素按钮" not in src, "旧文案不该残留"


def test_help_dialog_is_scrollable_and_current(window, qapp):
    """操作说明弹窗：必须可滚动，内容与当前交互一致。"""
    dlg = window.build_help_dialog()
    dlg.show()
    qapp.processEvents()
    viewer = dlg.help_viewer
    # 可滚动：内容高度超过视口
    assert viewer.verticalScrollBar().maximum() > 0, \
        "帮助内容应比视口高（有滚动条），否则下半截看不到"
    text = viewer.toPlainText()
    for word in ("拖出原子放置", "蓝点", "双键", "自动脱", "两种方式", "Tab"):
        assert word in text, f"帮助缺少关键说明：{word}"
    for stale in ("恢复原位", "点击左侧", "点左侧元素按钮", "放置 A", "两根手指"):
        assert stale not in text, f"帮助里还有过期说法：{stale}"
    dlg.close()


# ============================================================================
# C5. 测试报告回归（P1 / P2 残留 / P3 / P4 / P5）
# ============================================================================


def test_bonded_atom_click_leaves_no_undo_entry(canvas):
    """带键原子原地单击 = 什么都没改 → 不留撤销点（P2 残留修复）。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.clear_history()
    canvas._fit_view(animate=False)
    bx, by = screen_of(canvas, w.atom(b.uid))
    press_at(canvas, bx, by)
    release_at(canvas, bx, by)
    assert w.bond_count == 1, "键不该动"
    assert w.undo_depth() == 0, f"带键原子单击不该留撤销点，实得 {w.undo_depth()}"


def test_break_cancel_drag_leaves_no_undo_entry(canvas):
    """拖远触发断键 → 拖回原位取消 → 松手：世界没变，不留撤销点（P2 同源残留）。"""
    w = canvas.world
    a = w.create_atom("C", 0.0, 0.0)[2]
    b = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    w.clear_history()
    canvas._fit_view(animate=False)
    bx, by = screen_of(canvas, w.atom(b.uid))
    press_at(canvas, bx, by)
    far_x, far_y = screen_of(canvas, 4.0, 0.0)
    move_to(canvas, far_x, far_y)
    assert canvas.renderer.breaking_uid == b.uid, "应先进入将断键状态"
    # 拖回**按下时的位置**：旋转量回到 0，世界恢复起始姿态
    move_to(canvas, bx, by)
    assert canvas.renderer.breaking_uid is None, "拖回应取消断键"
    release_at(canvas, bx, by)
    assert w.bond_count == 1, "键还在"
    bb = w.atom(b.uid)
    assert abs(bb.x - 1.0) < 1e-9 and abs(bb.y) < 1e-9, "原子应回到起始姿态"
    assert w.undo_depth() == 0, f"断键取消后不该留撤销点，实得 {w.undo_depth()}"


def test_undo_after_stand_up_restores_top_down_camera(canvas):
    """3D 立起来 + 转视角后撤销 → 相机回到正视（P5 修复：相机入快照）。"""
    w = canvas.world
    c1 = w.create_atom("C", 0.0, 0.0)[2]
    c2 = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(c1.uid, c2.uid, 1)
    w.clear_history()
    canvas._fit_view(animate=False)
    canvas.stand_up()
    assert w.mode == "view3d"
    canvas._anim_timer.stop()
    # 玩家在 3D 里转视角
    canvas.camera.orbit(120, 80)
    assert abs(canvas.camera.azimuth) > 0.1, "先确认视角确实转了"

    assert w.undo() is True
    assert w.mode == "build", "撤销应回到平面模式"
    assert abs(canvas.camera.azimuth) < 1e-9 and abs(canvas.camera.elevation) < 1e-9, \
        f"撤销后相机应回到正视，实得 az={canvas.camera.azimuth:.3f} el={canvas.camera.elevation:.3f}"


def test_snapshot_carries_camera_state(canvas):
    """快照应带相机状态（P5 的机制保证）。"""
    canvas.camera.orbit(60, 40)
    snap = canvas.world.snapshot()
    assert "camera" in snap, "快照里应有相机状态"
    canvas.camera.reset(top_down=True)
    canvas.world.restore(snap)
    assert abs(canvas.camera.azimuth - 60 * 0.008) < 1e-9, "restore 应恢复相机方位角"


def test_stand_up_refuses_when_too_many_atoms(canvas):
    """补氢后超上限：先试算、直接拒绝 —— 一个氢都不补、世界不动（P1 修复）。"""
    w = canvas.world
    prev_uid = None
    for i in range(70):
        atom = w.create_atom("C", float(i), 0.0)[2]
        if prev_uid is not None:
            w.add_bond(prev_uid, atom.uid, 1)
        prev_uid = atom.uid
    assert w.atom_count == 70
    w.clear_history()
    canvas._fit_view(animate=False)
    canvas.stand_up()
    assert w.atom_count == 70, f"超限时不该补氢，实得 {w.atom_count} 个原子"
    assert w.mode == "build", "不应进入 3D"
    assert w.undo_depth() == 0, "拒绝操作不该留撤销点"


def test_stand_up_clears_redo(canvas):
    """立起来是新的编辑操作 → 清空重做栈（P3 修复）。"""
    w = canvas.world
    # 先制造一条重做记录：创建氧 → 撤销
    w.push_undo()
    oxy = w.create_atom("O", 5.0, 5.0)[2]
    assert w.undo() is True
    assert w.redo_depth() == 1, "先确认重做栈有 1 条"

    # 再搭个乙烷立起来
    c1 = w.create_atom("C", 0.0, 0.0)[2]
    c2 = w.create_atom("C", 1.0, 0.0)[2]
    w.add_bond(c1.uid, c2.uid, 1)
    canvas._fit_view(animate=False)
    canvas.stand_up()
    assert w.mode == "view3d"
    assert w.redo_depth() == 0, f"立起来后重做栈应清空，实得 {w.redo_depth()}"
    assert w.undo_depth() == 1, f"立起来应留下一个撤销点，实得 {w.undo_depth()}"


def test_saturated_moving_atom_snap_is_infeasible():
    """满价原子拖到空插槽：候选不可行（P4 修复：feasible 要查 moving 自己）。"""
    w = World()
    host = w.create_atom("C", 0.0, 0.0)[2]
    full = w.create_atom("C", 0.9, 0.9)[2]
    for ang in range(4):
        h = w.create_atom("H", full.x + math.cos(ang * math.pi / 2),
                          full.y + math.sin(ang * math.pi / 2))[2]
        w.add_bond(full.uid, h.uid, 1)
    assert w.free_valence(full.uid) == 0
    # 把满价碳放到宿主插槽点附近
    _d, point = w.free_slot_points(host.uid)[0]
    w.move_atom(full.uid, point[0], point[1])
    cand = w.find_snap_candidate(full.uid)
    assert cand is not None, "应该给出候选（用于提示原因）"
    assert cand.feasible is False, "满价原子不该吸附"
    assert "价键已用满" in cand.reason, f"原因要说清，实得 {cand.reason!r}"


def test_apply_snap_failure_restores_position():
    """apply_snap 成键失败 → 位置回滚（P4 修复）。"""
    w = World()
    host = w.create_atom("C", 0.0, 0.0)[2]
    full = w.create_atom("C", 3.0, 0.0)[2]
    for ang in range(4):
        h = w.create_atom("H", full.x + math.cos(ang * math.pi / 2),
                          full.y + math.sin(ang * math.pi / 2))[2]
        w.add_bond(full.uid, h.uid, 1)
    before = (full.x, full.y)
    # 构造一个"宿主可行"的候选（模拟拖到插槽点、且 moving 检查被绕过的情况）
    _d, point = w.free_slot_points(host.uid)[0]
    fake = SnapCandidate(host_uid=host.uid, host_el="C", point=point,
                         direction=(0.0, 0.0), distance=0.1, angle_deviation=0.0,
                         feasible=True, slot_index=0)
    ok, _msg = w.apply_snap(full.uid, fake)
    assert ok is False, "满价原子成键必须失败"
    assert abs(w.atom(full.uid).x - before[0]) < 1e-9, \
        f"失败后位置应回滚，实得 x={w.atom(full.uid).x:.4f}"
    assert abs(w.atom(full.uid).y - before[1]) < 1e-9
    assert w.bond_count == 4, "不该新增键"


def test_info_panel_selector_lists_all_molecules(qapp):
    from molecule_workshop.ui.info_panel import InfoPanel
    w = World()
    w.create_atom("C", -4, 0)
    w.auto_fill_hydrogens()
    w.create_atom("O", 4, 0)
    w.auto_fill_hydrogens()

    panel = InfoPanel()
    panel.update_from_world(w)
    assert panel.selector_box.isVisible() or len(panel._selector_buttons) == 2
    assert len(panel._selector_buttons) == 2


def test_info_panel_can_switch_between_molecules(qapp):
    from molecule_workshop.ui.info_panel import InfoPanel
    w = World()
    w.create_atom("C", -4, 0)
    w.auto_fill_hydrogens()
    w.create_atom("O", 4, 0)
    w.auto_fill_hydrogens()

    panel = InfoPanel()
    panel.update_from_world(w)
    first = panel.cells["formula"].text()
    # 换到第二个分子
    keys = [panel._part_key(p) for p in [x for x in w.identify()[0] if not x.loose_atom]]
    panel.select_molecule(keys[1])
    panel.update_from_world(w)
    second = panel.cells["formula"].text()
    assert first != second, "切换分子后分子式应该变了"
    assert second == "H2O", f"第二个分子应是水，实得 {second}"


def test_info_panel_falls_back_when_selected_molecule_removed(qapp):
    """选中的分子被删掉后回落到另一个，不报错。"""
    from molecule_workshop.ui.info_panel import InfoPanel
    w = World()
    w.create_atom("C", -4, 0)
    w.auto_fill_hydrogens()
    w.create_atom("O", 4, 0)
    w.auto_fill_hydrogens()
    panel = InfoPanel()
    panel.update_from_world(w)
    parts = [p for p in w.identify()[0] if not p.loose_atom]
    panel.select_molecule(panel._part_key(parts[1]))
    # 把水删掉
    for a in list(w.atoms):
        if a.x > 0:
            w.remove_atom(a.uid)
    panel.update_from_world(w)          # 不应抛异常
    assert panel.cells["formula"].text() == "CH4"


def test_info_panel_single_molecule_hides_selector(qapp):
    from molecule_workshop.ui.info_panel import InfoPanel
    w = World()
    w.create_atom("C", 0, 0)
    w.auto_fill_hydrogens()
    panel = InfoPanel()
    panel.update_from_world(w)
    assert not panel.selector_box.isVisible()


def test_info_panel_unknown_molecule_shows_status(qapp):
    from molecule_workshop.ui.info_panel import InfoPanel
    w = World()
    # 新戊烷：库外
    c = w.create_atom("C", 0, 0)[2]
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        n = w.create_atom("C", dx, dy)[2]
        w.add_bond(c.uid, n.uid, 1)
    w.auto_fill_hydrogens()
    panel = InfoPanel()
    panel.update_from_world(w)
    assert panel.cells["status"].text() == "未收录"
    assert panel.cells["formula"].text() == "C5H12"


# ============================================================================
# F. 布局回归
# ============================================================================


@pytest.fixture
def window(qapp):
    from molecule_workshop.ui.main_window import MainWindow
    win = MainWindow()
    win.resize(1280, 820)
    win.show()
    qapp.processEvents()
    return win


def test_window_minimum_matches_layout_needs(window, qapp):
    """缩到窗口下限时，三栏都不能被压扁。

    v5 起这个下限是**算**出来的（侧栏宽度随字体变化），不再是常数。
    这里不去复算那个公式 —— 直接缩到下限，然后量实际结果，
    这样即使以后布局改了，测试依然能抓住"画布被挤没"这类问题。
    """
    from molecule_workshop.chem.constants import CANVAS_MIN_WIDTH
    window.resize(window.minimumWidth(), window.minimumHeight())
    for _ in range(3):
        window.layout().activate()
        qapp.processEvents()
    assert window.side.width() >= 200, f"侧栏被压到 {window.side.width()} px"
    assert window.canvas.width() >= CANVAS_MIN_WIDTH, \
        f"画布被挤到 {window.canvas.width()} px（下限 {CANVAS_MIN_WIDTH}）"


def test_sidebar_never_squeezes_its_buttons(window):
    """侧栏不能把自己里面的按钮压扁 —— 压扁就会叠字（本次修的问题）。

    早先侧栏写死 226 px，而内容区可用宽正好等于最宽按钮宽（余量 0 px），
    字体稍微大一点就会顶出按钮、文字糊成一团。
    """
    from PySide6 import QtGui, QtWidgets
    inner = window.side.width() - 2 * 14
    for btn in window.side.findChildren(QtWidgets.QPushButton):
        if not btn.text() or not btn.isVisible():
            continue
        fm = QtGui.QFontMetrics(btn.font())
        icon_w = btn.iconSize().width() + 8 if not btn.icon().isNull() else 0
        need = fm.horizontalAdvance(btn.text()) + icon_w + 20
        assert btn.width() >= need, \
            f"{btn.text().strip()[:14]!r} 需要 {need} px，实际只有 {btn.width()} px"
        assert btn.width() <= inner, \
            f"{btn.text().strip()[:14]!r} 比内容区还宽（{btn.width()} > {inner}）"


def test_narrow_window_auto_collapses_info_panel(window, qapp):
    window.resize(1000, 700)
    qapp.processEvents()
    assert not window.info_scroll.isVisible(), "窄窗口应自动收起资料面板"
    assert window.canvas.width() >= 360, f"画布被挤到 {window.canvas.width()} px"


def test_wide_window_restores_info_panel(window, qapp):
    window.resize(1000, 700)
    qapp.processEvents()
    assert not window.info_scroll.isVisible()
    window.resize(1280, 820)
    qapp.processEvents()
    assert window.info_scroll.isVisible(), "宽度恢复后资料面板应自动展开"


def test_manual_hide_is_not_overridden_by_resize(window, qapp):
    """玩家手动取消勾选后，resize 不应把资料面板又自动勾回来。"""
    window.resize(1280, 820)
    qapp.processEvents()
    window._set_info_visible(False)          # 等同于玩家取消勾选
    assert not window.info_scroll.isVisible()
    # 先变窄再变宽，都不应该擅自展开
    window.resize(1000, 700)
    qapp.processEvents()
    assert not window.info_scroll.isVisible()
    window.resize(1400, 860)
    qapp.processEvents()
    assert not window.info_scroll.isVisible(), "玩家手动收起后不该被自动展开"


def test_canvas_minimum_size_is_reasonable(window):
    from molecule_workshop.chem.constants import CANVAS_MIN_HEIGHT, CANVAS_MIN_WIDTH
    assert window.canvas.minimumWidth() == CANVAS_MIN_WIDTH
    assert window.canvas.minimumHeight() == CANVAS_MIN_HEIGHT


def test_tool_list_has_four_tools(window):
    from molecule_workshop.render.canvas import TOOLS
    assert TOOLS == ("鼠标", "连键", "删除", "平移")
    assert set(window.side.tool_buttons.keys()) == set(TOOLS)


def test_history_buttons_reflect_stack(window, qapp):
    w = window.world
    assert not window.side.btn_undo.isEnabled()
    assert not window.side.btn_redo.isEnabled()
    w.push_undo()
    w.create_atom("C", 0, 0)
    window._on_world_changed()
    qapp.processEvents()
    assert window.side.btn_undo.isEnabled()
    assert "可撤销 1 步" in window.side.lbl_history.text()

    window._on_undo()
    qapp.processEvents()
    assert window.side.btn_redo.isEnabled()


def test_status_bar_shows_molecule_when_panel_collapsed(window, qapp):
    """资料面板收起时，状态栏要给出简版分子信息，别让信息彻底消失。"""
    w = window.world
    w.create_atom("C", 0, 0)
    w.auto_fill_hydrogens()
    window._set_info_visible(False)
    window._on_world_changed()
    qapp.processEvents()
    assert "CH4" in window.lbl_count.text()
