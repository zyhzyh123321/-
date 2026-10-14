"""主窗口：把画布、左侧工具栏与右侧资料面板组装起来。

布局是**响应式**的：
    左面板固定 226（本来就窄），画布可伸缩，右侧资料面板可折叠。
    窗口宽度小于 NARROW_WIDTH 时自动收起资料面板，把空间让给画布 ——
    早先三栏都是固定宽度（合计约 1094 px），而窗口最小宽度设成 1000，
    窄窗口下画布会被挤到低于自己的最小尺寸、出现错位。
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..world import World
from ..chem.constants import (
    CANVAS_MIN_WIDTH,
    ELEMENTS,
    INFO_PANEL_MIN_WIDTH,
    INFO_PANEL_WIDTH,
    NARROW_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from ..render.canvas import CanvasWidget
from ..render.scene import SceneRenderer, Theme
from .info_panel import InfoPanel
from .side_panel import SidePanel
from . import theme as th

HELP_HTML = """
<h3 style="color:#45e0d0; margin-bottom:6px;">放原子</h3>
<p><b>从左侧原子盘拖出原子放置</b>：按住元素按钮拖到画布，松手即可放一个原子 ——
这是唯一的放置方式。拖到某个原子周围的蓝点附近松手，会直接吸附成键。</p>
<p>右键画布空白处也能「在此放置」。</p>
<h3 style="color:#45e0d0; margin-bottom:6px;">四个工具</h3>
<table cellpadding="4">
<tr><td><b>鼠标 V</b></td><td>左键点原子选中、拖动移动；拖空白<b>框选</b>；
    Ctrl+点击可多选；按 <b>X</b> 删除所选</td></tr>
<tr><td><b>连键 B</b></td><td>按住原子上的<b>蓝点</b>拖到另一个蓝点，松手连键；
    拖动时有虚线预览，连完两端自动摆到标准键长</td></tr>
<tr><td><b>删除 D</b></td><td>点原子/键删除；拖动框选批量删除</td></tr>
<tr><td><b>平移 Space</b></td><td>拖动画布平移视图</td></tr>
</table>
<h3 style="color:#45e0d0; margin-bottom:6px;">改键级（双键 / 三键）</h3>
<p>两种方式：<b>右键点那根键</b>，菜单里直接选「改为双键 / 三键 / 单键」；
或者把鼠标悬在键上按 <b>+</b> / <b>-</b>。</p>
<p>升键需要额外插槽，程序会<b>自动脱掉自动补上的氢</b>腾出位置，
并在提示里告诉你脱了几个（Ctrl+Z 可撤回）。例如乙烷 C—C 升双键就变成乙烯。</p>
<h3 style="color:#45e0d0; margin-bottom:6px;">拖动原子的三种结果</h3>
<table cellpadding="4">
<tr><td><b>带键的原子</b></td><td>绕相连的那个原子<b>整体旋转</b> ——
    只改方向，<b>键长一丝不变</b>，身后的基团跟着一起转</td></tr>
<tr><td><b>没有键的原子</b></td><td>跟手自由移动，<b>拖到哪放到哪</b>（松手就停在那里）</td></tr>
<tr><td><b>拖得足够远</b></td><td>键先变成<b>红色虚线</b>提示，松手才真正断开；
    拖回阈值内可以取消</td></tr>
</table>
<p>把两个已经相连的原子<b>拖到一起</b>，可以把键升一级（单 → 双 → 三）。</p>
<h3 style="color:#45e0d0; margin-bottom:6px;">鼠标与相机</h3>
<table cellpadding="4">
<tr><td><b>右键</b></td><td>弹出菜单：成键 / 改键级 / 删除 / 在此放置 / 撤销</td></tr>
<tr><td><b>中键</b></td><td>3D 模式下旋转视角；2D 模式下平移视图</td></tr>
<tr><td><b>Shift+中键</b></td><td>平移视图（两种模式都可以）</td></tr>
<tr><td><b>滚轮</b></td><td>以光标为中心缩放</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>撤销 / 重做（Ctrl+Shift+Z 也可以重做）</td></tr>
</table>
<h3 style="color:#45e0d0; margin-bottom:6px;">其它快捷键</h3>
<table cellpadding="4">
<tr><td><b>Tab</b></td><td>立起来看 3D / 重新编辑（来回切换）</td></tr>
<tr><td><b>H</b></td><td>显示 / 隐藏氢原子</td></tr>
<tr><td><b>G</b></td><td>显示 / 隐藏参考网格</td></tr>
</table>
<h3 style="color:#45e0d0; margin-bottom:6px;">化合价规则</h3>
<p>碳 4 根键、氢 1 根、氧 2 根、氮 3 根。双键占 2 个插槽，三键占 3 个。<br>
原子周围的小圆点就是它的插槽：<b>亮的还空着、暗的已经被占用</b>；
亮点的位置就是松手会吸附到的位置。插槽用满就连不上了 ——
这是化学事实，也是这个游戏的核心玩法。</p>
<h3 style="color:#45e0d0; margin-bottom:6px;">3D 是怎么来的</h3>
<p>不是随手摆的：先按杂化态决定键角（单键 109.47°、双键 120°、三键 180°），
再用位置约束求解器把键长键角精确收敛。环己烷的椅式构象是求解器自己「折」出来的。</p>
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("分子工坊 · 化学结构搭建沙盒")
        self.resize(1280, 820)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.world = World()
        self._info_visible = True
        self._auto_collapsed = False
        self._user_choice = False       # 玩家是否手动决定过资料面板的显隐

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(12)

        self.side = SidePanel()
        layout.addWidget(self.side)

        self.canvas = CanvasWidget(self.world)
        layout.addWidget(self.canvas, 1)

        # ---- 右侧资料面板：放进 QSplitter，可拖动调宽，也可整体收起 ----
        self.info = InfoPanel()
        info_scroll = QtWidgets.QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_scroll.setWidget(self.info)
        info_scroll.setMinimumWidth(INFO_PANEL_MIN_WIDTH)
        self.info_scroll = info_scroll

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(0)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(info_scroll)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1280 - 226 - INFO_PANEL_WIDTH - 40, INFO_PANEL_WIDTH])
        layout.addWidget(self.splitter, 1)

        self._build_statusbar()
        self._connect()
        # 侧栏宽度是按字体算出来的，所以窗口下限也得跟着算 ——
        # 否则在大字体/高缩放的机器上，900 px 的下限会让画布被挤到没有。
        self.setMinimumSize(self._min_window_width(), WINDOW_MIN_HEIGHT)

        self.canvas.say("从左侧原子盘按住拖到画布即可放原子；连键请从蓝点起拖", 6000)
        self.info.update_from_world(self.world)
        self._on_world_changed()

    # ------------------------------------------------------------------

    def _min_window_width(self) -> int:
        """按当前字体算出窗口最小宽度。

        组成（与 _build 里的布局一一对应）：
            左右外边距 12+12 + 侧栏 + 间距 12
            + [画布下限 + 间距 12 + 资料面板下限]  ← 都在 QSplitter 里
        侧栏和资料面板的宽度都取决于字体，所以不能硬写 900 ——
        大字体机器上侧栏能到 334 px，写死 900 会把画布压到几乎看不见。
        """
        return (12 + self.side.panel_width + 12
                + CANVAS_MIN_WIDTH + 12 + INFO_PANEL_MIN_WIDTH + 12)

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(True)

        self.lbl_tool = QtWidgets.QLabel("工具：鼠标")
        self.lbl_tool.setStyleSheet("color:#45e0d0; padding-left:6px;")
        self.lbl_hint = QtWidgets.QLabel("")
        self.lbl_hint.setStyleSheet("color:#8fa3b0;")
        self.lbl_count = QtWidgets.QLabel("原子 0   化学键 0")
        self.lbl_count.setStyleSheet("color:#5d7080; padding-right:6px;")
        self.lbl_fps = QtWidgets.QLabel("")
        self.lbl_fps.setStyleSheet("color:#3f5260; padding-right:10px;")

        bar.addWidget(self.lbl_tool)
        bar.addWidget(self.lbl_hint, 1)
        bar.addPermanentWidget(self.lbl_count)
        bar.addPermanentWidget(self.lbl_fps)

        self._fps_timer = QtCore.QTimer(self)
        self._fps_timer.setInterval(700)
        self._fps_timer.timeout.connect(self._update_fps)
        self._fps_timer.start()

    def _connect(self) -> None:
        self.side.element_picked.connect(self._on_element)
        self.side.tool_picked.connect(self._on_tool)
        self.side.stand_up.connect(self._on_stand_up)
        self.side.back_to_edit.connect(self._on_back)
        self.side.clear_all.connect(self._on_clear)
        self.side.reset_view.connect(self._on_reset_view)
        self.side.undo_requested.connect(self._on_undo)
        self.side.redo_requested.connect(self._on_redo)
        self.side.toggle_grid.connect(self._on_toggle_grid)
        self.side.toggle_hydrogen.connect(self._on_toggle_h)
        self.side.toggle_info_panel.connect(self._set_info_visible)
        self.side.show_help.connect(self._on_help)

        self.canvas.message_changed.connect(self.lbl_hint.setText)
        self.canvas.world_changed.connect(self._on_world_changed)
        self.canvas.context_menu_requested.connect(self._on_context_menu)
        self.info.molecule_selected.connect(self._on_molecule_selected)

        # Ctrl+Z 撤销 / Ctrl+Y、Ctrl+Shift+Z 重做
        undo = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        undo.activated.connect(self._on_undo)
        redo = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self)
        redo.activated.connect(self._on_redo)
        redo2 = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self)
        redo2.activated.connect(self._on_redo)
        # Tab 切换立起来 / 重新编辑
        tab = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Tab), self)
        tab.activated.connect(self._on_tab)

    # ------------------------------------------------------------------
    # 右侧面板的显隐与窄窗口自适应
    # ------------------------------------------------------------------

    def _set_info_visible(self, visible: bool, auto: bool = False) -> None:
        """显示/隐藏右侧资料面板。

        auto=True 表示这是「窄窗口自适应」触发的，不是玩家点的。
        玩家一旦手动勾选，就记住他的选择：``_user_choice`` 为真后
        自适应逻辑不再擅自改动，只在勾选框状态上跟随实际可见性
        （否则玩家取消勾选后，下一次 resize 又会被自动勾回来）。
        """
        self._info_visible = visible
        self.info_scroll.setVisible(visible)
        self.side.set_info_checked(visible)
        if auto:
            self._auto_collapsed = not visible
        else:
            self._user_choice = True
            self._auto_collapsed = False
        if not visible:
            self.lbl_hint.setText("资料面板已收起（左侧勾选「显示资料面板」可展开）")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:       # noqa: N802
        super().resizeEvent(event)
        if self._user_choice:
            # 玩家手动决定过显隐，不再自动干预
            return
        narrow = self.width() < NARROW_WIDTH
        if narrow and self._info_visible:
            # 窄窗口：自动收起资料面板给画布让位
            self._set_info_visible(False, auto=True)
        elif not narrow and not self._info_visible:
            # 宽度恢复：自动展开
            self._set_info_visible(True, auto=True)

    # ------------------------------------------------------------------

    def _on_element(self, symbol: str) -> None:
        self.canvas.selected_element = symbol
        self.canvas.setFocus()
        name = ELEMENTS.get(symbol, {}).get("name", symbol)
        self.canvas.say(f"已选中{name}：按住它拖到画布上即可放一个原子"
                        f"（也可以右键画布选「在此放置」）", 3600)

    def _on_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self.lbl_tool.setText(f"工具：{tool}")
        self.canvas.setFocus()

    # ------------------------------------------------------------------
    # 右键上下文菜单
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos: QtCore.QPoint, bond_index: int) -> None:
        """画布右键：按当前选择与落点动态生成菜单。"""
        menu = QtWidgets.QMenu(self)
        selected = list(self.canvas.renderer.selected_uids)
        world = self.world

        if bond_index >= 0:
            # 点在键上：给出**明确**的升/降键级项，而不是含义模糊的"切换"
            bond = world.bonds[bond_index]
            order = bond.order
            label = {1: "单键", 2: "双键", 3: "三键"}.get(order, "键")
            if order < 3:
                nxt = {1: "双键", 2: "三键"}[order]
                menu.addAction(f"当前是{label} → 改为{nxt}",
                               lambda: self._set_bond_order(bond_index, order + 1))
            if order > 1:
                prv = {2: "单键", 3: "双键"}[order]
                menu.addAction(f"当前是{label} → 改为{prv}",
                               lambda: self._set_bond_order(bond_index, order - 1))
            menu.addAction("删除这根键", lambda: self.canvas._delete_bond(bond_index))
        elif len(selected) == 2:
            a, b = selected
            has_bond = world.bond_index(a, b) >= 0
            if has_bond:
                cur = world.bonds[has_bond].order
                if cur < 3:
                    nxt = {1: "双键", 2: "三键"}[cur]
                    menu.addAction(f"把键升为{nxt}",
                                   lambda: self._set_bond_order(has_bond, cur + 1))
                if cur > 1:
                    prv = {2: "单键", 3: "双键"}[cur]
                    menu.addAction(f"把键降为{prv}",
                                   lambda: self._set_bond_order(has_bond, cur - 1))
                menu.addAction("删除这根键", self._delete_selected_bond)
            else:
                menu.addAction("成键（连单键）", self.canvas.bond_selection)
            menu.addSeparator()
            menu.addAction("删除选中的 2 个原子", self.canvas.delete_selection)
        elif len(selected) >= 1:
            menu.addAction(f"删除选中的 {len(selected)} 个原子", self.canvas.delete_selection)
            if len(selected) == 1:
                uid = selected[0]
                menu.addAction("以它为起点连键",
                               lambda: self._start_bond_from(uid))
            menu.addSeparator()
            menu.addAction("取消选择", self._clear_selection)
        else:
            # 空白处：在此放置
            sub = menu.addMenu("在此放置")
            for sym in ("C", "H", "O", "N"):
                name = ELEMENTS[sym]["name"]
                sub.addAction(f"{sym} {name}",
                              lambda s=sym, p=pos: self._place_at(pos, s))
        menu.addSeparator()
        act_undo = menu.addAction("撤销  (Ctrl+Z)", self._on_undo)
        act_undo.setEnabled(world.can_undo())
        act_redo = menu.addAction("重做  (Ctrl+Y)", self._on_redo)
        act_redo.setEnabled(world.can_redo())

        menu.exec(self.canvas.mapToGlobal(pos))

    def _set_bond_order(self, bond_index: int, order: int) -> None:
        """右键菜单里明确指定键级（升键时 World 会自动腾插槽）。"""
        if not (0 <= bond_index < len(self.world.bonds)):
            return
        self.world.push_undo()
        ok, msg = self.world.set_bond_order(bond_index, order)
        if ok:
            now = self.world.bonds[bond_index].order
            name = {1: "单键", 2: "双键", 3: "三键"}[now]
            extra = ""
            if self.world.last_removed_hydrogens:
                extra = f"（自动移除了 {len(self.world.last_removed_hydrogens)} 个氢）"
            self.canvas.say(f"键级已改为{name}{extra}", 2800)
        else:
            self.canvas.say(msg, 3200)
        self._on_world_changed()
        self.canvas.update()

    def _place_at(self, pos: QtCore.QPoint, symbol: str) -> None:
        """右键菜单里「在此放置」：用画布坐标换算出世界坐标后放一个原子。"""
        wp = self.canvas.camera.screen_to_plane(pos.x(), pos.y())
        self.canvas.selected_element = symbol
        self.side.set_element(symbol)
        self.world.push_undo()
        ok, msg, _atom = self.world.create_atom(symbol, wp[0], wp[1])
        if not ok:
            self.canvas.say(msg, 2600)
        else:
            self.canvas.say(f"放了 1 个{ELEMENTS[symbol]['name']}原子", 2000)
        self.canvas.world_changed.emit()
        self.canvas.update()

    def _start_bond_from(self, uid: int) -> None:
        self.canvas.set_tool("连键")
        self.side.set_tool("连键")
        self.lbl_tool.setText("工具：连键")
        self.canvas._bond_from_uid = uid
        self.canvas.say("已设为起点，拖到另一个原子上松手即可连键", 3000)

    def _clear_selection(self) -> None:
        self.canvas.renderer.selected_uids = []
        self.canvas.update()

    def _delete_selected_bond(self) -> None:
        sel = list(self.canvas.renderer.selected_uids)
        if len(sel) != 2:
            return
        bi = self.world.bond_index(sel[0], sel[1])
        if bi >= 0:
            self.canvas._delete_bond(bi)

    def _on_molecule_selected(self, key: str) -> None:
        self.info.select_molecule(key)
        self.info.update_from_world(self.world)

    def _on_stand_up(self) -> None:
        self.canvas.stand_up()
        self.side.set_mode(self.world.mode)

    def _on_back(self) -> None:
        self.canvas.back_to_edit()
        self.side.set_mode(self.world.mode)

    def _on_tab(self) -> None:
        if self.world.mode == "view3d":
            self._on_back()
        else:
            self._on_stand_up()

    def _on_clear(self) -> None:
        if self.world.is_empty():
            self.canvas.say("画布已经是空的", 1800)
            return
        self.world.push_undo()
        self.world.clear()
        self.canvas.renderer.preview = None
        self.canvas.renderer.selected_uids = []
        self.canvas.camera.reset(top_down=True)
        self.canvas.say("已清空画布（Ctrl+Z 可撤销）", 2400)
        self.side.set_mode("build")
        self._on_world_changed()
        self.canvas.update()

    def _on_reset_view(self) -> None:
        self.canvas.camera.reset(top_down=self.world.mode != "view3d")
        if self.world.mode == "view3d":
            self.canvas.camera.fit(SceneRenderer.content_radius(self.world), 1.22)
        self.canvas.say("视角已重置", 1600)
        self.canvas.update()

    def _on_toggle_grid(self, on: bool) -> None:
        self.canvas.renderer.show_grid = on
        self.canvas.update()

    def _on_toggle_h(self, on: bool) -> None:
        self.canvas.renderer.show_hydrogens = on
        self.canvas.update()

    def build_help_dialog(self) -> QtWidgets.QDialog:
        """操作说明弹窗：可滚动、可缩放。

        早先这里用 QMessageBox 塞一大段 HTML —— 内容超过一定高度就
        直接裁掉、看不到下半截，也没有滚动条。改成 QDialog + QTextBrowser：
        QTextBrowser 原生支持富文本和垂直滚动。
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("操作说明")
        dlg.resize(600, 560)
        dlg.setMinimumSize(480, 400)

        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(12, 12, 12, 12)

        viewer = QtWidgets.QTextBrowser(dlg)
        viewer.setOpenExternalLinks(False)
        viewer.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        viewer.setHtml(HELP_HTML)
        viewer.setObjectName("helpViewer")
        root.addWidget(viewer, 1)

        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setDefault(True)
        close_btn.clicked.connect(dlg.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        root.addLayout(row)

        dlg.help_viewer = viewer        # 供测试与后续调整使用
        return dlg

    def _on_help(self) -> None:
        self.build_help_dialog().exec()

    def _on_undo(self) -> None:
        if self.world.undo():
            self.canvas.renderer.preview = None
            self.canvas.renderer.breaking_uid = None
            self.canvas.say("已撤销（Ctrl+Y 可重做）", 1800)
            self.side.set_mode(self.world.mode)
            self._sync_tool_buttons()
            self._on_world_changed()
            self.canvas.update()
        else:
            self.canvas.say("没有可撤销的操作", 1600)

    def _on_redo(self) -> None:
        if self.world.redo():
            self.canvas.renderer.preview = None
            self.canvas.renderer.breaking_uid = None
            self.canvas.say("已重做", 1600)
            self.side.set_mode(self.world.mode)
            self._sync_tool_buttons()
            self._on_world_changed()
            self.canvas.update()
        else:
            self.canvas.say("没有可重做的操作", 1600)

    def _sync_tool_buttons(self) -> None:
        """撤销/重做可能改回 build 模式，按钮显示要跟着走。"""
        self.lbl_tool.setText(f"工具：{self.canvas.tool}")

    def _on_world_changed(self) -> None:
        self.lbl_count.setText(f"原子 {self.world.atom_count}   化学键 {self.world.bond_count}")
        self.side.set_history(self.world.undo_depth(), self.world.redo_depth())
        self.info.update_from_world(self.world)
        # 资料面板收起时，用状态栏给一行简版分子信息
        if not self._info_visible:
            parts, _total = self.world.identify()
            parts = [p for p in parts if not p.loose_atom]
            if parts:
                main = parts[0]
                rec = main.matched
                name = rec["names"][0] if rec else "未收录"
                mass = f"{main.mass:.2f}" if main.mass else "—"
                extra = f"（共 {len(parts)} 个分子）" if len(parts) > 1 else ""
                self.lbl_count.setText(
                    f"{name} {main.formula} · 相对分子质量 {mass}{extra}"
                    f"　原子 {self.world.atom_count}　化学键 {self.world.bond_count}")

    def _update_fps(self) -> None:
        ms = self.canvas.renderer.stats.draw_ms
        if ms > 0.01:
            self.lbl_fps.setText(f"{1000.0 / ms:.0f} fps · 绘制 {ms:.1f} ms")
        self.lbl_tool.setText(f"工具：{self.canvas.tool}")
        # 3D 模式下的提示
        if self.world.mode == "view3d":
            self.lbl_hint.setText("拖动旋转视角 · 滚轮缩放 · 中键/右键平移 · Tab 回到平面")
