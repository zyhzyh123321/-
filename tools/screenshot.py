"""整窗离屏截图：真正创建 MainWindow 并出图。

和 render_preview.py 的区别：
    render_preview.py 是我手搭的简化框架图，用来快速调渲染；
    本脚本跑的是**真实主窗口**（真工具栏、真资料面板、真交互状态），
    所以它验证的是「玩家实际会看到的界面」。

用法：
    python tools/screenshot.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets      # noqa: E402

OUT_DIR = ROOT / "out"


def shoot(window: QtWidgets.QWidget, name: str) -> None:
    """抓窗口图像并保存。"""
    app = QtWidgets.QApplication.instance()
    for _ in range(3):
        app.processEvents()
    pix = window.grab()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    ok = pix.save(str(path))
    print(f"  {'OK ' if ok else 'FAIL'} {name}  {pix.width()}x{pix.height()}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv[:1])

    from molecule_workshop.ui.theme import app_stylesheet, load_fonts, preferred_family
    family = load_fonts()
    app.setStyleSheet(app_stylesheet(preferred_family()))
    print(f"字体: {family}")

    from molecule_workshop.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(1280, 820)
    win.show()
    app.processEvents()

    # ---- 1. 空界面 ----
    shoot(win, "20_app_empty.png")

    # ---- 2. 搭乙醇骨架（模拟玩家点击） ----
    w = win.world
    c1 = w.create_atom("C", -1.0, 0.0)[2]
    c2 = w.create_atom("C", 0.0, 0.0)[2]
    o = w.create_atom("O", 1.0, 0.0)[2]
    w.add_bond(c1.uid, c2.uid, 1)
    w.add_bond(c2.uid, o.uid, 1)
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.renderer.hover_atom_uid = c2.uid
    win.canvas.say("拖动原子把插槽对上就连成键", 5000)
    shoot(win, "21_app_build.png")

    # ---- 3. 拖动吸附预览（虚线） ----
    w.clear()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    moving = w.create_atom("C", 1.0, 0.62)[2]
    cand = w.find_snap_candidate(moving.uid)
    win.canvas.renderer.preview = cand
    win.canvas.renderer.drag_uid = moving.uid
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("松手即可与碳连成单键", 5000)
    shoot(win, "22_app_snap_preview.png")

    # ---- 4. 立起来看 3D：苯 ----
    w.clear()
    win.canvas.renderer.preview = None
    win.canvas.renderer.drag_uid = None
    from molecule_workshop.chem.parse_spec import parse_spec

    mol = parse_spec("C1(-H)=C(-H)-C(-H)=C(-H)-C(-H)=C(-H)-1")
    ids = []
    for i, atom in enumerate(mol.atoms):
        ids.append(w.create_atom(atom.el, (i % 6) * 1.4, (i // 6) * 1.4)[2].uid)
    for bond in mol.bonds:
        w.add_bond(ids[bond.a], ids[bond.b], bond.order)
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.side.set_mode("build")
    shoot(win, "23_app_benzene_flat.png")

    win.canvas.stand_up()
    win.canvas._anim_elapsed = 10_000        # 跳过动画，直接看终态
    win.canvas._tick_anim()
    win.side.set_mode(w.mode)
    app.processEvents()
    win._on_world_changed()
    win.canvas.say("拖动旋转视角，滚轮缩放", 5000)
    shoot(win, "24_app_benzene_3d.png")

    # ---- 5. 环己烷 3D（椅式） ----
    w.clear()
    mol = parse_spec("C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1")
    ids = []
    for i, atom in enumerate(mol.atoms):
        ids.append(w.create_atom(atom.el, (i % 6) * 1.4, (i // 6) * 1.4)[2].uid)
    for bond in mol.bonds:
        w.add_bond(ids[bond.a], ids[bond.b], bond.order)
    win.canvas.stand_up()
    win.canvas._anim_elapsed = 10_000
    win.canvas._tick_anim()
    win.side.set_mode(w.mode)
    app.processEvents()
    win._on_world_changed()
    shoot(win, "25_app_cyclohexane_3d.png")

    # ---- 6. 未收录分子 ----
    w.clear()
    win.canvas.back_to_edit()
    c = w.create_atom("C", 0.0, 0.0)[2]
    for i in range(4):
        n = w.create_atom("C", [1.0, -1.0, 1.0, -1.0][i], [1.0, 1.0, -1.0, -1.0][i])[2]
        w.add_bond(c.uid, n.uid, 1)
    w.auto_fill_hydrogens()
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("库外分子也会给出分子式与元素组成", 5000)
    shoot(win, "26_app_unknown.png")

    # ---- 7. 删除工具 ----
    win.side.tool_buttons["删除"].click()
    win.canvas.renderer.hover_atom_uid = w.atoms[2].uid
    win.canvas.say("点原子即删除（连带它的所有键）", 5000)
    shoot(win, "27_app_tool_delete.png")

    # ---- 8. 选中 + 空插槽不再重叠（本次修的重点） ----
    w.clear()
    win.canvas.back_to_edit()
    win.side.tool_buttons["鼠标"].click()
    c1 = w.create_atom("C", -0.5, 0.0)[2]
    c2 = w.create_atom("C", 0.5, 0.0)[2]
    w.add_bond(c1.uid, c2.uid, 1)
    win.canvas.renderer.selected_uids = [c1.uid]
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("相邻碳的空插槽已错开，不会叠在一起", 5000)
    shoot(win, "28_app_slots_spread.png")

    # ---- 9. 拖动将断键（红色虚线） ----
    w.clear()
    a = w.create_atom("C", -0.6, 0.0)[2]
    b = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(a.uid, b.uid, 1)
    far = w.create_atom("H", 1.6, 0.0)[2]
    w.add_bond(b.uid, far.uid, 1)
    # 手动构造"拖到将断键"的画面状态
    win.canvas.renderer.breaking_uid = a.uid
    win.canvas.renderer.drag_uid = a.uid
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("拖得够远时键会变成红色虚线，松手即断开", 5000)
    shoot(win, "29_app_will_break.png")
    win.canvas.renderer.breaking_uid = None
    win.canvas.renderer.drag_uid = None

    # ---- 10. 多分子选择器 ----
    w.clear()
    m = w.create_atom("C", -4.0, 0.0)[2]
    w.auto_fill_hydrogens()
    o = w.create_atom("O", 4.0, 0.0)[2]
    w.auto_fill_hydrogens()
    c3 = w.create_atom("C", 0.0, 3.0)[2]
    c4 = w.create_atom("C", 1.0, 3.0)[2]
    w.add_bond(c3.uid, c4.uid, 1)
    w.auto_fill_hydrogens()
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("画布上多个分子时，可在右上角逐个切换查看", 5000)
    shoot(win, "30_app_multi_molecule.png")

    # ---- 11. 窄窗口布局 ----
    win.resize(980, 660)
    app.processEvents()
    win._on_world_changed()
    shoot(win, "31_app_narrow.png")
    win.resize(1280, 820)
    app.processEvents()

    # ---- 12. 连键工具：蓝点起拖的虚线预览 ----
    w.clear()
    win.canvas.back_to_edit()
    win.side.tool_buttons["连键"].click()
    a1 = w.create_atom("C", -1.4, -0.6)[2]
    a2 = w.create_atom("C", 1.4, -0.6)[2]
    win.canvas._fit_view(animate=False)
    _d, slot = w.free_slot_points(a1.uid)[0]
    import math as _m
    sx, sy = win.canvas.camera.project((slot[0], slot[1], 0.0))[:2]
    win.canvas._bond_from_uid = a1.uid
    win.canvas._bond_from_slot = 0
    # 预览终点指向 a2 的插槽
    _d2, slot2 = w.free_slot_points(a2.uid)[0]
    win.canvas.renderer.bond_preview = (a1.uid, slot2[0], slot2[1], True)
    win._on_world_changed()
    win.canvas.say("按住蓝点拖到另一个蓝点，虚线预览会吸附过去", 5000)
    shoot(win, "32_app_bond_preview.png")
    win.canvas.renderer.bond_preview = None

    # ---- 13. 从原子盘拖出原子时的幽灵球 ----
    win.side.tool_buttons["鼠标"].click()
    gx, gy = win.canvas.camera.project((0.2, 0.3, 0.0))[:2]
    win.canvas.renderer.palette_ghost = ("O", 0.2, 0.3, True)
    win.canvas.say("从原子盘拖出来时，落点会显示一个半透明预览球", 5000)
    shoot(win, "33_app_palette_drag.png")
    win.canvas.renderer.palette_ghost = None

    # ---- 14. 双键 / 三键（乙烷 → 乙烯 → 乙炔）----
    w.clear()
    c5 = w.create_atom("C", -0.6, 0.0)[2]
    c6 = w.create_atom("C", 0.6, 0.0)[2]
    w.add_bond(c5.uid, c6.uid, 1)
    w.auto_fill_hydrogens()
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("乙烷 C2H6：右键点 C—C 键可以升成双键", 5000)
    shoot(win, "34_app_ethane.png")

    bi = w.bond_index(c5.uid, c6.uid)
    w.push_undo()
    w.set_bond_order(bi, 2)          # 自动脱掉 2 个氢
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("升双键后自动脱氢 → 乙烯 C2H4", 5000)
    shoot(win, "35_app_ethylene.png")

    w.push_undo()
    w.set_bond_order(bi, 3)
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("再升三键 → 乙炔 C2H2", 5000)
    shoot(win, "36_app_acetylene.png")

    # ---- 15. 带键拖动：绕锚旋转成 90° ----
    w.clear()
    b1 = w.create_atom("C", 0.0, 0.0)[2]
    b2 = w.create_atom("C", 1.2, 0.0)[2]
    w.add_bond(b1.uid, b2.uid, 1)
    w.move_atom(b2.uid, 0.0, 1.2)
    win.canvas._fit_view(animate=False)
    win._on_world_changed()
    win.canvas.say("拖带键的原子只会改方向，键长一点不变", 5000)
    shoot(win, "37_app_rotate.png")

    print("\n输出目录:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
