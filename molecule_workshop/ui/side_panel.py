"""左侧工具栏：元素盘 + 工具选择 + 视图开关。

元素按钮带元素色的球体图示，工具按钮显示快捷键；
「立起来看 3D / 重新编辑」按当前模式切换显示。

**面板宽度由字体度量算出来，不写死**：
    早先这里写死 `setFixedWidth(226)`。在那台机器上没问题，但换个
    显示缩放（125%/150%）或换个字体，同样 13px 的字会占更多像素，
    写死的宽度就装不下 —— 表现是文字互相重叠、糊成一团。
    现在改成「量一遍所有控件真正需要多宽，再留出内边距」，
    字体变大面板就跟着变宽，永远不会挤坏。
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..chem.constants import ELEMENTS, PLACEABLE
from ..chem.vector import hex_to_rgb
from ..render.canvas import TOOLS

#: 侧栏设计宽度（在标准 DPI / 微软雅黑 13px 下就是它）
SIDE_PANEL_WIDTH = 226
#: 侧栏左右内边距（见 _build 里的 setContentsMargins）
SIDE_PANEL_MARGIN = 14
#: 宽度下限，保证再小的字体也不至于挤成一条
SIDE_PANEL_MIN_WIDTH = 200


def sphere_icon(hex_color: str, size: int = 18) -> QtGui.QIcon:
    """画一个带光照的小球作为图标（和 3D 视图里的球同一套观感）。"""
    pm = QtGui.QPixmap(size * 2, size * 2)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    r = size
    base = QtGui.QColor(hex_color)
    grad = QtGui.QRadialGradient(r * 0.62, r * 0.58, max(0.5, r * 0.06), r, r, r * 1.02)
    grad.setColorAt(0.0, base.lighter(205))
    grad.setColorAt(0.45, base)
    grad.setColorAt(1.0, base.darker(265))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QBrush(grad))
    p.drawEllipse(QtCore.QPointF(r, r), r - 1, r - 1)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.setPen(QtGui.QPen(base.darker(400), 1.2))
    p.drawEllipse(QtCore.QPointF(r, r), r - 1.6, r - 1.6)
    p.end()
    return QtGui.QIcon(pm)


#: 原子盘拖放用的 MIME 类型（与 canvas.ELEMENT_MIME 必须一致）
ELEMENT_MIME = "application/x-mol-element"


class ElementButton(QtWidgets.QPushButton):
    """元素按钮：单击选中，**按住拖出**则可以拖到画布上放一个原子。

    两种操作共用同一个按钮：
        单击  → 只切换"当前元素"（右键菜单里的「在此放置」会用）
        拖出  → 启动 Qt 拖放，画布收到 drop 后在落点创建原子
    """

    def __init__(self, symbol: str, on_pick) -> None:
        info = ELEMENTS[symbol]
        super().__init__(f"  {symbol}  {info['name']}        {info['valence']} 个价键")
        self.symbol = symbol
        self.setObjectName("elemButton")
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setIcon(sphere_icon(info["color"], 10))
        self.setIconSize(QtCore.QSize(20, 20))
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setToolTip(f"{info['name']}（{info['name_en']}）— 最多能连 {info['valence']} 根键\n"
                        f"按住拖到画布上即可放一个{info['name']}原子")
        self.clicked.connect(lambda: on_pick(symbol))
        self._press_pos: QtCore.QPoint | None = None
        self._dragging = False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:      # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:       # noqa: N802
        if self._press_pos is None or self._dragging:
            super().mouseMoveEvent(event)
            return
        moved = (event.position().toPoint() - self._press_pos).manhattanLength()
        if moved < QtWidgets.QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        # 超过拖动阈值 → 启动拖放
        self._dragging = True
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData(ELEMENT_MIME, self.symbol.encode("utf-8"))
        drag.setMimeData(mime)
        # 用球体图标当拖拽缩略图，拖的时候看得见抓的是什么
        pm = sphere_icon(ELEMENTS[self.symbol]["color"], 26).pixmap(52, 52)
        drag.setPixmap(pm)
        drag.setHotSpot(QtCore.QPoint(26, 26))
        drag.exec(QtCore.Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:    # noqa: N802
        self._press_pos = None
        self._dragging = False
        super().mouseReleaseEvent(event)


class ToolButton(QtWidgets.QPushButton):
    def __init__(self, name: str, shortcut: str, on_pick) -> None:
        super().__init__(f"  {name}            {shortcut}")
        self.tool_name = name
        self.setObjectName("toolButton")
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(lambda: on_pick(name))


class SidePanel(QtWidgets.QWidget):
    """左侧面板。"""

    element_picked = QtCore.Signal(str)
    tool_picked = QtCore.Signal(str)
    stand_up = QtCore.Signal()
    back_to_edit = QtCore.Signal()
    clear_all = QtCore.Signal()
    reset_view = QtCore.Signal()
    undo_requested = QtCore.Signal()
    redo_requested = QtCore.Signal()
    toggle_grid = QtCore.Signal(bool)
    toggle_hydrogen = QtCore.Signal(bool)
    toggle_info_panel = QtCore.Signal(bool)
    show_help = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._panel_width = 0
        self._build()
        self._apply_width()

    @property
    def panel_width(self) -> int:
        """面板的设计宽度。

        给布局计算用 —— 不要拿 `width()`：构造期间控件可能还没被布局，
        `width()` 会返回 100 之类的默认值，算出来的窗口下限就偏小。
        """
        return self._panel_width or SIDE_PANEL_WIDTH

    def _section(self, root: QtWidgets.QVBoxLayout, title: str) -> None:
        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName("sectionLabel")
        root.addWidget(lbl)

    # ------------------------------------------------------------------
    # 宽度：量出来，不写死
    # ------------------------------------------------------------------

    #: 内容区与最宽按钮之间要留的余量。
    #: 实测发现原来的写法余量正好是 0：内容区 198 px、最宽按钮 198 px，
    #: 一个像素的富余都没有。于是只要字体稍微大一点（125% 显示缩放、
    #: 或系统换了字体），文字就会顶出按钮、和相邻控件叠在一起，
    #: 看起来就是"糊成一团"。留 12 px 之后就有真正的容错空间。
    WIDTH_SLACK = 12

    def content_width(self) -> int:
        """面板里的控件真正需要多宽（不含左右内边距）。

        组成 = 最宽按钮的文字宽度 + 该按钮的"附加宽度"（图标、图标间距、
        样式表 padding、边框）。

        附加宽度用 `QStyle` 的 cmHint 算，**不靠已经布局出来的宽度反推**：
        构造阶段控件还没被布局，宽度是默认的 100 px，反推出来的 chrome
        会是 0，于是面板永远停在 226 不动 —— 实测字体放大到 2 倍时
        内容区还是 198 px，文字早就顶出去了。
        """
        needed = 0
        style = self.style()
        for btn in self.findChildren(QtWidgets.QPushButton):
            if not btn.text():
                continue
            fm = QtGui.QFontMetrics(btn.font())
            icon_w = 0
            if not btn.icon().isNull():
                icon_w = btn.iconSize().width() + 8       # 图标 + 间距
            cm = style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ButtonMargin, None, btn)
            frame = style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_DefaultFrameWidth, None, btn)
            # 样式表 QPushButton#elemButton/#toolButton 给的是 padding: 7px 10px
            needed = max(needed, fm.horizontalAdvance(btn.text())
                         + icon_w + 2 * max(cm, 10) + 2 * frame)
        for lbl in self.findChildren(QtWidgets.QLabel):
            if lbl.text():
                needed = max(needed,
                             QtGui.QFontMetrics(lbl.font()).horizontalAdvance(lbl.text()))
        return needed

    def _apply_width(self) -> None:
        """按当前字体算出并固定面板宽度。

        量之前必须先 `ensurePolished()`：否则子控件还没套上样式表里的
        `font-size: 13px`，量到的是默认字号，算出来的宽度永远偏小 ——
        实测把字体调到 20px 时仍算出 226，文字早就顶出面板了。
        """
        self.ensurePolished()
        for child in self.findChildren(QtWidgets.QWidget):
            child.ensurePolished()
        width = max(SIDE_PANEL_WIDTH,
                    self.content_width() + 2 * SIDE_PANEL_MARGIN + self.WIDTH_SLACK)
        self.setFixedWidth(width)
        self.setMinimumWidth(width)
        self._panel_width = width

    def showEvent(self, event: QtGui.QShowEvent) -> None:      # noqa: N802
        """显示时再核对一次宽度（此时字体/样式一定已经生效）。"""
        super().showEvent(event)
        self._apply_width()

    def _on_style_or_font_change(self) -> None:
        """字体或样式变化后重新算宽度（换 DPI、换主题、换字体会走到这里）。"""
        self._apply_width()

    def changeEvent(self, event: QtCore.QEvent) -> None:      # noqa: N802
        """字体/样式变化时重量一次宽度（换 DPI、换主题都会走到这里）。"""
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.FontChange,
                            QtCore.QEvent.Type.StyleChange):
            self._on_style_or_font_change()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(9)

        # ---- 元素盘 ----
        self._section(root, "原 子 盘")
        self.elem_buttons: dict[str, ElementButton] = {}
        for symbol in PLACEABLE:
            btn = ElementButton(symbol, self.element_picked.emit)
            self.elem_buttons[symbol] = btn
            root.addWidget(btn)
        self.elem_buttons["C"].setChecked(True)

        # ---- 工具 ----
        root.addSpacing(6)
        self._section(root, "工 具")
        self.tool_buttons: dict[str, ToolButton] = {}
        for name, key in (("鼠标", "V"), ("连键", "B"),
                          ("删除", "D"), ("平移", "Space")):
            btn = ToolButton(name, key, self.tool_picked.emit)
            self.tool_buttons[name] = btn
            root.addWidget(btn)
        self.tool_buttons["鼠标"].setChecked(True)

        # ---- 撤销 / 重做 ----
        root.addSpacing(6)
        self._section(root, "编 辑")
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.btn_undo = QtWidgets.QPushButton("撤销")
        self.btn_undo.setToolTip("撤销上一步（Ctrl+Z）")
        self.btn_undo.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_redo = QtWidgets.QPushButton("重做")
        self.btn_redo.setToolTip("重做刚撤销的操作（Ctrl+Y / Ctrl+Shift+Z）")
        self.btn_redo.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_redo.clicked.connect(self.redo_requested.emit)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_redo)
        root.addLayout(row)

        self.lbl_history = QtWidgets.QLabel("无可撤销的操作")
        self.lbl_history.setObjectName("caption")
        root.addWidget(self.lbl_history)

        # ---- 视图 ----
        root.addSpacing(6)
        self._section(root, "视 图")

        self.btn_standup = QtWidgets.QPushButton("立起来看 3D   (Tab)")
        self.btn_standup.setObjectName("primaryButton")
        self.btn_standup.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_standup.setToolTip("自动补满氢原子，再按真实键角算出立体构型")
        self.btn_standup.clicked.connect(self.stand_up.emit)
        root.addWidget(self.btn_standup)

        self.btn_edit = QtWidgets.QPushButton("重新编辑   (Tab)")
        self.btn_edit.setObjectName("primaryButton")
        self.btn_edit.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_edit.setToolTip("回到平面拼装，原来的布局不会被打乱")
        self.btn_edit.clicked.connect(self.back_to_edit.emit)
        self.btn_edit.setVisible(False)
        root.addWidget(self.btn_edit)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.btn_reset = QtWidgets.QPushButton("重置视角")
        self.btn_reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset_view.emit)
        self.btn_clear = QtWidgets.QPushButton("清空画布")
        self.btn_clear.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_all.emit)
        row.addWidget(self.btn_reset)
        row.addWidget(self.btn_clear)
        root.addLayout(row)

        self.chk_grid = QtWidgets.QCheckBox("显示参考网格")
        self.chk_grid.setChecked(True)
        self.chk_grid.setToolTip("网格只是参考线，原子不会被吸附到格子上")
        self.chk_grid.toggled.connect(self.toggle_grid.emit)
        root.addWidget(self.chk_grid)

        self.chk_h = QtWidgets.QCheckBox("显示氢原子  (H)")
        self.chk_h.setChecked(True)
        self.chk_h.toggled.connect(self.toggle_hydrogen.emit)
        root.addWidget(self.chk_h)

        # 窄窗口下可以把资料面板收起来给画布让位
        self.chk_info = QtWidgets.QCheckBox("显示资料面板")
        self.chk_info.setChecked(True)
        self.chk_info.setToolTip("窗口较窄时可以收起，把空间让给画布")
        self.chk_info.toggled.connect(self.toggle_info_panel.emit)
        root.addWidget(self.chk_info)

        root.addStretch(1)

        self.btn_help = QtWidgets.QPushButton("操作说明")
        self.btn_help.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_help.clicked.connect(self.show_help.emit)
        root.addWidget(self.btn_help)

    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        is3d = mode == "view3d"
        self.btn_standup.setVisible(not is3d)
        self.btn_edit.setVisible(is3d)

    def set_tool(self, tool: str) -> None:
        btn = self.tool_buttons.get(tool)
        if btn is not None:
            btn.setChecked(True)

    def set_element(self, symbol: str) -> None:
        btn = self.elem_buttons.get(symbol)
        if btn is not None:
            btn.setChecked(True)

    def set_history(self, undo_depth: int, redo_depth: int) -> None:
        """更新撤销/重做按钮的可用状态与文字提示。"""
        self.btn_undo.setEnabled(undo_depth > 0)
        self.btn_redo.setEnabled(redo_depth > 0)
        if undo_depth == 0 and redo_depth == 0:
            text = "无可撤销的操作"
        elif redo_depth == 0:
            text = f"可撤销 {undo_depth} 步"
        elif undo_depth == 0:
            text = f"可重做 {redo_depth} 步"
        else:
            text = f"可撤销 {undo_depth} 步 · 可重做 {redo_depth} 步"
        self.lbl_history.setText(text)

    def set_info_checked(self, checked: bool) -> None:
        """同步「显示资料面板」勾选状态（不带信号，避免回环）。"""
        self.chk_info.blockSignals(True)
        self.chk_info.setChecked(checked)
        self.chk_info.blockSignals(False)
