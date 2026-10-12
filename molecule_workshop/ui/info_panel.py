"""右侧物质资料面板。

显示内容分两种情形：
    已收录的分子  中文名 / 英文名 / 分子式 / 相对分子质量 / 结构特征 / 常见用途 / 备注 / 元素组成
    库外的分子    分子式 / 相对分子质量 / 元素组成（这些由原子直接算出来，一定准确），
                  并明确说明「未收录」

画布上有多个分子时，面板显示最大的那个，并提示共有几个分子。
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..chem.constants import ELEMENTS
from ..chem.vector import hex_to_rgb
from ..world import World


class CompositionBar(QtWidgets.QWidget):
    """一根元素质量百分比条。"""

    def __init__(self, symbol: str, name: str, percent: float, parent=None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.name = name
        self.percent = percent
        self.setFixedHeight(22)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:          # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        font = p.font()
        font.setPixelSize(11)
        p.setFont(font)
        p.setPen(QtGui.QColor("#8fa3b0"))
        p.drawText(QtCore.QRectF(0, 0, 78, h),
                   QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                   f"{self.name} {self.symbol}")

        # 百分比数字要留够宽度。
        # 曾经的疑问：面板里这个数字看起来像 "160.0%"，怀疑被进度条盖住。
        # 实际量过：13px 字号下 "100.0%" 宽 47 px，这里留 60 px，
        # 进度条右边缘 = 302-60-8 = 234，文字从 302-60=242 开始，中间有 8 px 间隙，
        # 实测：微软雅黑 11px 下 "100.0%" 需要 59 px，
        # 这里留 74 px（多 15 px 余量），进度条不会顶到数字上。
        # 说明：缩略图上数字看着像被吞掉或串位，是缩放预览的假象 ——
        # 1:1 渲染出来是完整的 "100.0%"，已用单条渲染核实过。
        pct_w = 74.0
        gap = 8.0
        track_x = 82.0
        track_w = max(10.0, w - track_x - pct_w - gap)

        track = QtCore.QRectF(track_x, h / 2 - 3.5, track_w, 7)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 20)))
        p.drawRoundedRect(track, 3.5, 3.5)

        color = QtGui.QColor(ELEMENTS.get(self.symbol, {}).get("color", "#888888"))
        fill_w = max(2.0, track.width() * min(1.0, self.percent / 100.0))
        p.setBrush(QtGui.QBrush(color))
        p.drawRoundedRect(QtCore.QRectF(track.x(), track.y(), fill_w, track.height()), 3.5, 3.5)

        p.setPen(QtGui.QColor("#a9bcc7"))
        p.drawText(QtCore.QRectF(w - pct_w, 0, pct_w, h),
                   QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight,
                   f"{self.percent:.1f}%")
        p.end()


class InfoPanel(QtWidgets.QWidget):
    """资料面板。调用 update_from_world(world) 刷新。

    画布上有多个分子时，顶部会出现一个**分子选择器**，
    可以逐个点开查看（不再只显示最大的那个）。
    选择用 selection_key 记住：结构变化后若该分子还在就保持选中，
    不在则回落到最大的那个。
    """

    #: 用户切换了要查看的分子（参数是 selection_key）
    molecule_selected = QtCore.Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._comp_bars: list[QtWidgets.QWidget] = []
        self._selector_buttons: list[QtWidgets.QPushButton] = []
        self._selection_key: str | None = None
        self._build()
        self.update_from_world(None)

    # ------------------------------------------------------------------

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ---- 分子选择器（多个分子时才显示） ----
        self.selector_box = QtWidgets.QWidget()
        sel_lay = QtWidgets.QVBoxLayout(self.selector_box)
        sel_lay.setContentsMargins(0, 0, 0, 0)
        sel_lay.setSpacing(5)
        self.lbl_selector_title = QtWidgets.QLabel("选 择 分 子")
        self.lbl_selector_title.setObjectName("sectionLabel")
        sel_lay.addWidget(self.lbl_selector_title)
        self.selector_host = QtWidgets.QWidget()
        self.selector_layout = QtWidgets.QVBoxLayout(self.selector_host)
        self.selector_layout.setContentsMargins(0, 0, 0, 0)
        self.selector_layout.setSpacing(4)
        sel_lay.addWidget(self.selector_host)
        root.addWidget(self.selector_box)
        self.selector_box.setVisible(False)

        # ---- 标题 ----
        self.lbl_name = QtWidgets.QLabel("—")
        self.lbl_name.setObjectName("titleLabel")
        self.lbl_name.setWordWrap(True)
        self.lbl_en = QtWidgets.QLabel("")
        self.lbl_en.setObjectName("subtitleLabel")
        root.addWidget(self.lbl_name)
        root.addWidget(self.lbl_en)

        line = QtWidgets.QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: rgba(120,170,190,40);")
        root.addWidget(line)

        # ---- 四格基本信息 ----
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        self.cells: dict[str, QtWidgets.QLabel] = {}
        specs = [("分子式", "formula"), ("相对分子质量", "mass"),
                 ("原子数", "count"), ("状态", "status")]
        for i, (caption, key) in enumerate(specs):
            box = QtWidgets.QFrame()
            box.setObjectName("panelSoft")
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(10, 7, 10, 7)
            lay.setSpacing(1)
            cap = QtWidgets.QLabel(caption)
            cap.setObjectName("caption")
            val = QtWidgets.QLabel("—")
            val.setObjectName("bigValue")
            lay.addWidget(cap)
            lay.addWidget(val)
            self.cells[key] = val
            grid.addWidget(box, i // 2, i % 2)
        root.addLayout(grid)

        # ---- 结构特征 / 用途 / 备注 ----
        self.sec_desc = self._add_section(root, "结 构 特 征", "bodyText", wrap=True)
        self.sec_use = self._add_section(root, "常 见 用 途", "bodyText", wrap=True)
        self.sec_note = self._add_section(root, "备 注", "dimText", wrap=True)

        # ---- 元素组成 ----
        self.lbl_comp_title = QtWidgets.QLabel("元 素 组 成")
        self.lbl_comp_title.setObjectName("sectionLabel")
        root.addWidget(self.lbl_comp_title)
        self.comp_host = QtWidgets.QWidget()
        self.comp_layout = QtWidgets.QVBoxLayout(self.comp_host)
        self.comp_layout.setContentsMargins(0, 0, 0, 0)
        self.comp_layout.setSpacing(2)
        root.addWidget(self.comp_host)

        # ---- 多分子提示 ----
        self.lbl_multi = QtWidgets.QLabel("")
        self.lbl_multi.setObjectName("dimText")
        self.lbl_multi.setWordWrap(True)
        self.lbl_multi.setStyleSheet(
            "color: #ffb340; background: rgba(255,179,64,26);"
            "border: 1px solid rgba(255,179,64,70); border-radius: 8px; padding: 8px;")
        root.addWidget(self.lbl_multi)

        root.addStretch(1)

    def _add_section(self, root: QtWidgets.QVBoxLayout, title: str,
                     object_name: str, wrap: bool) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QtWidgets.QLabel(title)
        head.setObjectName("sectionLabel")
        body = QtWidgets.QLabel("")
        body.setObjectName(object_name)
        body.setWordWrap(wrap)
        lay.addWidget(head)
        lay.addWidget(body)
        root.addWidget(holder)
        holder.body = body          # type: ignore[attr-defined]
        return holder

    # ------------------------------------------------------------------

    @staticmethod
    def _part_key(part) -> str:
        """给一个组分算稳定标识，用于记住"玩家在看哪个分子"。"""
        return part.key + "|" + part.formula

    def select_molecule(self, key: str | None) -> None:
        """外部指定要查看的分子（None = 回到最大的那个）。"""
        self._selection_key = key

    def _rebuild_selector(self, parts: list) -> None:
        """重建分子选择器按钮。只有一个分子时整块隐藏。"""
        for btn in self._selector_buttons:
            self.selector_layout.removeWidget(btn)
            btn.deleteLater()
        self._selector_buttons.clear()

        if len(parts) <= 1:
            self.selector_box.setVisible(False)
            return

        keys = [self._part_key(p) for p in parts]
        if self._selection_key not in keys:
            self._selection_key = keys[0]

        for part, key in zip(parts, keys):
            rec = part.matched
            label = rec["names"][0] if rec else part.formula
            btn = QtWidgets.QPushButton(f"{label}   {part.formula}")
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setChecked(key == self._selection_key)
            btn.setObjectName("toolButton")
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self._on_selector_clicked(k))
            self.selector_layout.addWidget(btn)
            self._selector_buttons.append(btn)
        self.selector_box.setVisible(True)

    def _on_selector_clicked(self, key: str) -> None:
        self._selection_key = key
        self.molecule_selected.emit(key)

    def update_from_world(self, world: World | None) -> None:
        parts, total = world.identify() if world is not None else ([], 0)
        parts = [p for p in parts if not p.loose_atom]

        for bar in self._comp_bars:
            self.comp_layout.removeWidget(bar)
            bar.deleteLater()
        self._comp_bars.clear()

        if not parts:
            self._rebuild_selector([])
            self._set_placeholder()
            return

        self._rebuild_selector(parts)
        keys = [self._part_key(p) for p in parts]
        # 被选中的分子消失了 → 回落到最大的那个（不报错）
        if self._selection_key not in keys:
            self._selection_key = keys[0]
        main = parts[keys.index(self._selection_key)]
        rec = main.matched

        self.lbl_name.setText(rec["names"][0] if rec else main.formula)
        self.lbl_en.setText(rec.get("en", "") if rec else "（库外分子）")
        self.cells["formula"].setText(main.formula)
        self.cells["mass"].setText(f"{main.mass:.2f}" if main.mass else "—")
        self.cells["count"].setText(f"{main.count} 个")
        self.cells["status"].setText("已收录" if rec else "未收录")
        self.cells["status"].setStyleSheet(
            f"color: {'#45e0d0' if rec else '#ffb340'};")

        if rec:
            self.sec_desc.body.setText(rec.get("desc", ""))
            self.sec_use.body.setText(rec.get("use", ""))
            self.sec_note.body.setText(rec.get("note", ""))
        else:
            # 库外分子：能算的照算，并如实说明
            self.sec_desc.body.setText(
                "这个结构不在内置分子库里。不过分子式、相对分子质量与元素质量百分比"
                "都是由原子直接算出来的，可以放心参考。"
            )
            self.sec_use.body.setText("")
            self.sec_note.body.setText("")

        self.sec_use.setVisible(bool(rec and rec.get("use")))
        self.sec_note.setVisible(bool(rec and rec.get("note")))

        for comp in main.composition:
            bar = CompositionBar(comp["symbol"], comp["name"], comp["percent"])
            self.comp_layout.addWidget(bar)
            self._comp_bars.append(bar)

        if len(parts) > 1:
            self.lbl_multi.setText(
                f"画布上共有 {len(parts)} 个独立分子。用上方的选择器可以逐个查看，"
                f"当前显示的是「{self.lbl_name.text()}」。")
            self.lbl_multi.setVisible(True)
        else:
            self.lbl_multi.setVisible(False)

    def _set_placeholder(self) -> None:
        self.lbl_name.setText("还没有分子")
        self.lbl_en.setText("")
        self.cells["formula"].setText("—")
        self.cells["mass"].setText("—")
        self.cells["count"].setText("—")
        self.cells["status"].setText("—")
        self.cells["status"].setStyleSheet("color: #e6f0f5;")
        self.sec_desc.body.setText(
            "在画布上放几个原子、把它们连成键，这里就会显示这个物质是什么、有什么特征和用途。"
        )
        self.sec_use.setVisible(False)
        self.sec_note.setVisible(False)
        self.lbl_multi.setVisible(False)
