"""编辑状态：原子、键、化合价校验、自动补氢、视图模式、撤销重做、插槽吸附。

这里是唯一被修改的可变状态。渲染与界面都只读它。

坐标约定：
    编辑平面用原子的 (x, y)；
    立起来后的三维坐标放在 atom.pos3 上。
    因此反复「立起来 / 重新编辑」不会破坏玩家摆好的平面布局 ——
    平面布局永远不会被 3D 求解结果覆盖。

与网页版的区别：
    1. 编辑时**没有网格吸附**。格子与插槽方向在几何上根本不兼容：
       键长=1 时只有 0°/180° 能落在方格上。拖动完全跟手，
       靠「插槽吸附」在松手时精确就位。
    2. 空插槽方向按**可成键数**自适应角间距（见 free_slot_directions）——
       这是 2D 平面能摆得开、氢原子不互相重叠的关键。
    3. 支持撤销 + 重做，并支持拖动超阈值自动断键。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .chem.constants import (
    DRAG_MIN_ATOM_DIST,
    ELEMENTS,
    MAX_ATOMS,
    MAX_BONDS,
    REDO_STACK_SIZE,
    RELAX_MAX_ATOMS,
    SLOT_ANGLE_BY_VALENCE,
    SLOT_DIRS_2D,
    SLOT_MIN_SEPARATION,
    SLOT_SNAP_DIR_TOLERANCE,
    SLOT_SNAP_RADIUS,
    UNDO_STACK_SIZE,
)
from .chem.graph import Molecule
from .chem.relax import solve as chem_solve
from .chem.identify import identify_molecule
from .chem.vector import clamp, dist2, normalize2

# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------


@dataclass
class WorldAtom:
    """一个原子。

    uid     稳定唯一标识（删除/查找都用它，不用下标，避免下标漂移）
    el      元素符号
    x, y    编辑平面坐标
    pos3    三维坐标（立起来后有效）
    auto_h  是否由「自动补氢」创建（清理时只动这些）
    """

    uid: int
    el: str
    x: float = 0.0
    y: float = 0.0
    pos3: tuple[float, float, float] | None = None
    auto_h: bool = False


@dataclass
class WorldBond:
    a: int          # 原子 uid
    b: int
    order: int = 1


@dataclass
class SnapCandidate:
    """拖动时的吸附候选。

    拖动中的原子 X 靠近某个已有原子 A 的空插槽时，会产生一个候选：
    落点 = A 的位置 + 方向 × 键长，玩家松手就把 X 精确放到落点上并成键。

    upgrade_order > 0 表示这是一种特殊候选：A 与 X **已经相连**，
    玩家把两者拖到一起的意图是「升键」（单→双→三），而不是再连一根新键。
    这种情况下 point 就是 A 自己所在的位置附近，界面不画虚线、只提示键级。
    """

    host_uid: int                       # 宿主原子
    host_el: str
    point: tuple[float, float]          # 理想落点（世界坐标）
    direction: tuple[float, float]      # 插槽方向
    distance: float                     # X 当前位置到落点的距离
    angle_deviation: float              # X 相对方向的偏差（度）
    feasible: bool                      # 宿主是否还有空插槽
    reason: str = ""                    # 不可行的原因（提示用）
    slot_index: int = 0                 # 这是宿主的第几个空插槽（界面靠它精确高亮）
    upgrade_order: int = 0              # >0 表示升键候选，值是目标键级


# ----------------------------------------------------------------------------


def _angle_gaps(dirs: list[tuple[float, float]]
                ) -> tuple[float, float, float]:
    """给定一组单位方向，返回 (最宽空档的起始角, 最宽空档的宽度, 方向数)。

    角度一律归一化到 [0, 2π) 再求差 —— `atan2` 会返回负数（-π/2 等），
    不归一化的话最后一个到第一个的 gap 会算出负数，最宽空档会被误判。
    """
    if not dirs:
        return 0.0, 0.0, 0
    angles = sorted(a % (2.0 * math.pi) for a in
                    (math.atan2(d[1], d[0]) for d in dirs))
    best_gap = -1.0
    best_start = 0.0
    for i, a0 in enumerate(angles):
        a1 = angles[(i + 1) % len(angles)]
        if i == len(angles) - 1:
            a1 += 2.0 * math.pi
        gap = a1 - a0
        if gap > best_gap:
            best_gap = gap
            best_start = a0
    return best_start, best_gap, len(angles)


class World:
    """编辑器状态。"""

    def __init__(self) -> None:
        self.atoms: list[WorldAtom] = []
        self.bonds: list[WorldBond] = []
        self.mode: str = "build"            # 'build' | 'view3d'
        self._next_uid = 1
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self.last_message: str = ""         # 最近一次操作的提示（供界面显示）
        self.last_removed_hydrogens: list[int] = []   # 最近一次升键自动脱掉的氢
        # 附加状态编解码器（画布注册相机用）
        self._state_get = None
        self._state_set = None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def atom(self, uid: int) -> WorldAtom | None:
        for a in self.atoms:
            if a.uid == uid:
                return a
        return None

    def bond_index(self, a_uid: int, b_uid: int) -> int:
        for i, bond in enumerate(self.bonds):
            if (bond.a == a_uid and bond.b == b_uid) or (bond.a == b_uid and bond.b == a_uid):
                return i
        return -1

    def used_valence(self, uid: int) -> int:
        total = 0
        for bond in self.bonds:
            if bond.a == uid or bond.b == uid:
                total += bond.order
        return total

    def free_valence(self, uid: int) -> int:
        atom = self.atom(uid)
        if atom is None:
            return 0
        info = ELEMENTS.get(atom.el)
        if info is None:
            return 0
        return info["valence"] - self.used_valence(uid)

    def neighbors(self, uid: int) -> list[int]:
        out = []
        for bond in self.bonds:
            if bond.a == uid:
                out.append(bond.b)
            elif bond.b == uid:
                out.append(bond.a)
        return out

    def heavy_neighbors(self, uid: int) -> list[int]:
        out = []
        for n in self.neighbors(uid):
            other = self.atom(n)
            if other is not None and other.el != "H":
                out.append(n)
        return out

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def bond_count(self) -> int:
        return len(self.bonds)

    def is_empty(self) -> bool:
        return not self.atoms

    # ------------------------------------------------------------------
    # 修改
    # ------------------------------------------------------------------

    def create_atom(self, el: str, x: float, y: float) -> tuple[bool, str, WorldAtom | None]:
        """新建原子（不做网格吸附，位置就是鼠标位置）。"""
        if el not in ELEMENTS:
            return False, f"未知元素：{el}", None
        if len(self.atoms) >= MAX_ATOMS:
            return False, f"原子数量已达上限（{MAX_ATOMS}）", None
        atom = WorldAtom(uid=self._next_uid, el=el, x=x, y=y)
        self._next_uid += 1
        self.atoms.append(atom)
        self._invalidate_geometry()
        return True, "", atom

    def remove_atom(self, uid: int) -> bool:
        """删除原子及其所有键。"""
        idx = next((i for i, a in enumerate(self.atoms) if a.uid == uid), -1)
        if idx < 0:
            return False
        self.atoms.pop(idx)
        self.bonds = [b for b in self.bonds if b.a != uid and b.b != uid]
        self._invalidate_geometry()
        return True

    def clear(self) -> None:
        self.atoms = []
        self.bonds = []
        self.mode = "build"
        self.last_message = ""

    def move_atom(self, uid: int, x: float, y: float, avoid_overlap: bool = False) -> None:
        """移动原子。

        两条原则：
          1. **不做网格吸附** —— 那正是旧版「拖动生硬」的原因，
             而且六方向插槽与方格在几何上并不兼容（只有 0°/180° 能落在格点上）。
          2. **默认不强行推开别的原子** —— 那会让拖动不跟手、还破坏玩家刚摆好的布局。
             重叠只影响观感，渲染时用深度排序处理即可；真需要间距时由补氢逻辑自己避让。
        avoid_overlap=True 时才做最小间距分离（留作它用）。
        """
        atom = self.atom(uid)
        if atom is None:
            return
        atom.x, atom.y = x, y
        if avoid_overlap:
            self._separate_overlaps(uid)
        self._invalidate_geometry()

    def _separate_overlaps(self, moving_uid: int) -> None:
        """把与移动原子太近的其它原子轻轻推开（保持最小间距）。"""
        moving = self.atom(moving_uid)
        if moving is None:
            return
        for other in self.atoms:
            if other.uid == moving_uid:
                continue
            d = dist2(moving.x, moving.y, other.x, other.y)
            if d >= DRAG_MIN_ATOM_DIST or d < 1e-6:
                continue
            nx, ny = normalize2(moving.x - other.x, moving.y - other.y)
            push = (DRAG_MIN_ATOM_DIST - d) * 0.5
            other.x -= nx * push
            other.y -= ny * push

    def add_bond(self, a_uid: int, b_uid: int, order: int = 1) -> tuple[bool, str]:
        """尝试连键。双方都必须有足够的剩余插槽。"""
        order = int(clamp(order, 1, 3))
        if a_uid == b_uid:
            return False, "不能把原子连到自己"
        a, b = self.atom(a_uid), self.atom(b_uid)
        if a is None or b is None:
            return False, "原子不存在"

        existing = self.bond_index(a_uid, b_uid)
        if existing >= 0:
            return self.set_bond_order(existing, order)

        if len(self.bonds) >= MAX_BONDS:
            return False, "化学键数量已达上限"

        free_a = self.free_valence(a_uid)
        free_b = self.free_valence(b_uid)
        if free_a <= 0:
            name = ELEMENTS[a.el]["name"]
            return False, f"{name}的 {ELEMENTS[a.el]['valence']} 个价键已用满"
        if free_b <= 0:
            name = ELEMENTS[b.el]["name"]
            return False, f"{name}的 {ELEMENTS[b.el]['valence']} 个价键已用满"
        if free_a < order or free_b < order:
            return False, "剩余价键不足，无法形成该键级的键"

        self.bonds.append(WorldBond(a_uid, b_uid, order))
        self._invalidate_geometry()
        return True, ""

    def set_bond_order(self, bond_idx: int, order: int) -> tuple[bool, str]:
        """改键级。升键时若插槽不够，会自动脱掉自动补的氢（见 free_slots_for_order）。

        这条自动脱氢是「乙烷 → 乙烯」能被走通的关键：
        补满氢的乙烷两个碳都没有空插槽，不脱氢就永远升不了键，
        玩家看到的就会是「双键根本做不出来」。
        """
        if not (0 <= bond_idx < len(self.bonds)):
            return False, "键不存在"
        bond = self.bonds[bond_idx]
        order = int(clamp(order, 1, 3))
        if order == bond.order:
            return True, ""
        if order > bond.order:
            ok, msg, removed = self.free_slots_for_order(bond_idx, order)
            if not ok:
                return False, msg
            bond.order = order
            self._invalidate_geometry()
            # 键级变了 → 插槽方向也变了，把还留着的氢重新摆到新插槽上
            # （否则乙炔会画成折线，看不出"四原子共线"）
            self.tidy_hydrogens(bond.a)
            self.tidy_hydrogens(bond.b)
            self.last_removed_hydrogens = removed
            return True, ""
        # 降键：不自动补氢（避免玩家没意识到结构被改了），只提示
        bond.order = order
        self._invalidate_geometry()
        self.last_removed_hydrogens = []
        return True, ""

    def cycle_bond_order(self, bond_idx: int) -> tuple[bool, str]:
        """键级循环：单键 → 双键 → 三键 → 单键。"""
        if not (0 <= bond_idx < len(self.bonds)):
            return False, "键不存在"
        cur = self.bonds[bond_idx].order
        return self.set_bond_order(bond_idx, 1 if cur >= 3 else cur + 1)

    # ------------------------------------------------------------------
    # 自动补氢
    # ------------------------------------------------------------------

    def predict_hydrogen_additions(self) -> int:
        """试算「补满氢」会新增几个原子（**不改变任何东西**）。

        测试报告 P1：立起来时先补氢、再发现超限失败，氢已悄悄改坏画布
        且不可撤销。正确做法是先算好规模，超限就直接拒绝 —— 一个氢都不补。
        公式与 auto_fill_hydrogens 的 need_to_add 完全一致：
            化合价 - 与重原子的键用量 - 已挂着的氢数。
        """
        total = 0
        for atom in self.atoms:
            if atom.el == "H":
                continue
            info = ELEMENTS.get(atom.el)
            if info is None:
                continue
            heavy_used = 0
            h_attached = 0
            for bond in self.bonds:
                if bond.a != atom.uid and bond.b != atom.uid:
                    continue
                other_uid = bond.b if bond.a == atom.uid else bond.a
                other = self.atom(other_uid)
                if other is None:
                    continue
                if other.el == "H":
                    h_attached += 1
                else:
                    heavy_used += bond.order
            total += max(0, info["valence"] - heavy_used - h_attached)
        return total

    def auto_fill_hydrogens(self, max_total: int = MAX_ATOMS) -> dict:
        """按每个重原子的剩余价键数补满氢原子。

        已有的氢（无论自动补的还是玩家手动放的）都计入，只补差额。

        两个必须遵守的规则（都是踩过坑的）：
          1. 只删「自己补的、且现在不该保留的」氢。
             网页版曾因为把 existing 里的全部元素都当候选，
             结果把**相邻的碳原子**当成多余氢删掉，分子直接消失。
          2. 已有的自动氢不能计入「已补数量」——
             否则已经补满的原子会算出 target==autoCount，
             看起来持平，实际永远走不到补齐分支，反而把氢一遍遍删掉再重补。
        """
        added = removed = 0

        # 1. 统计每个重原子挂着哪些氢
        existing: dict[int, list[WorldAtom]] = {}
        manual_h: dict[int, int] = {}
        to_drop: list[int] = []

        for atom in self.atoms:
            if atom.el != "H":
                continue
            heavy = self.heavy_neighbors(atom.uid)
            if len(heavy) != 1:
                to_drop.append(atom.uid)      # 游离氢或多重连接，清理掉
                continue
            host = heavy[0]
            existing.setdefault(host, []).append(atom)
            if not atom.auto_h:
                manual_h[host] = manual_h.get(host, 0) + 1

        for uid in to_drop:
            if self.remove_atom(uid):
                removed += 1

        # 2. 逐个重原子补齐 / 裁减
        heavy_atoms = [a for a in self.atoms if a.el != "H"]
        for host in heavy_atoms:
            info = ELEMENTS.get(host.el)
            if info is None:
                continue

            # 「还能补几个氢」必须只看**与重原子的键**，不能把已有的 C—H 键也算进去。
            #
            # 这是最要命的一处坑，网页版和 Python 版都错了一轮，记下来：
            #   used_valence 会把已有的 C—H 键计进去，于是「已补满的碳」算出 need=0；
            #   此时若 target 也跟着变成 0，就会去执行「裁减到 0 个」，
            #   把刚补好的氢全删掉 —— 表现是连搭两个碳时第一个碳的氢全被删光。
            # 正确做法：
            #   need_to_add = 化合价 - 已有的重原子键用量 - 已有的氢数
            #   而已有的氢（无论自动补的还是手动放的）都要保留。
            heavy_used = 0
            h_attached = 0
            for bond in self.bonds:
                if bond.a != host.uid and bond.b != host.uid:
                    continue
                other_uid = bond.b if bond.a == host.uid else bond.a
                other = self.atom(other_uid)
                if other is None:
                    continue
                if other.el == "H":
                    h_attached += 1
                else:
                    heavy_used += bond.order

            need_to_add = max(0, info["valence"] - heavy_used - h_attached)
            have_manual = manual_h.get(host.uid, 0)
            valence_minus_heavy = max(0, info["valence"] - heavy_used)

            # 目标氢数 = 这个重原子「应该有几个氢」+ 玩家手动放的那些（手动的不删）。
            #
            # 这是最要命的一处坑，网页版和 Python 版都错了一轮，把正确公式记下来：
            #   v = 化合价 - 与重原子的键用量      ← 这个原子应有的氢总数
            #   target_h = max(v, 手动氢数)        ← 手动放的氢必须保留
            #   keep_auto = target_h - 手动氢数    ← 需要保留的自动氢
            #
            # 错法（两次都错在这里）：把「已有的氢数」从 v 里减掉再当成目标，
            # 于是已补满的碳算出目标 0，接着执行「裁减到 0 个」，
            # 把刚补好的氢全删掉 —— 表现是连搭两个碳时第一个碳的氢全被删光、
            # 分子式算错（C 而不是 CH4）。
            target_h = max(valence_minus_heavy, have_manual)
            keep_auto = max(0, target_h - have_manual)

            attached = existing.get(host.uid, [])
            auto_list = [a for a in attached if a.auto_h]

            if len(auto_list) > keep_auto:
                for atom in auto_list[keep_auto:]:
                    if self.remove_atom(atom.uid):
                        removed += 1
            kept_auto = min(len(auto_list), keep_auto)

            # 补齐差额
            for slot in range(max(0, keep_auto - kept_auto)):
                if len(self.atoms) >= max_total:
                    self._invalidate_geometry()
                    return {"added": added, "removed": removed,
                            "total": len(self.atoms), "truncated": True}
                px, py = self.pick_hydrogen_spot(host, kept_auto + slot, keep_auto)
                ok, _msg, new_atom = self.create_atom("H", px, py)
                if not ok or new_atom is None:
                    break
                new_atom.auto_h = True
                self.add_bond(host.uid, new_atom.uid, 1)
                added += 1

        self._invalidate_geometry()
        return {"added": added, "removed": removed, "total": len(self.atoms), "truncated": False}

    def pick_hydrogen_spot(self, host: WorldAtom, slot: int, total: int | None = None
                           ) -> tuple[float, float]:
        """给重原子挑一个放氢的位置，返回世界坐标。

        做法：把「该原子最终要放的氢数」在它周围的空白扇形里均匀铺开，
        每个氢再单独避让 —— 先找出一个不与已有原子/已放的氢重叠的角度。

        为什么必须传 total：早先只按「当前是第几个」算角度，
        甲烷的 4 个氢里有 2 个会算到同一个角度上、直接叠在一起，
        结果补氢只补出 3 个原子（分子式因此算错）。
        """
        neighbors = self.neighbors(host.uid)
        angles = []
        for n in neighbors:
            other = self.atom(n)
            if other is not None:
                angles.append(math.atan2(other.y - host.y, other.x - host.x))
        angles.sort()

        # 最宽的空白扇形
        if not angles:
            base, span = math.pi / 4.0, 2.0 * math.pi
        else:
            base, span, best_gap = angles[0], 0.0, -1.0
            for k, a0 in enumerate(angles):
                a1 = angles[(k + 1) % len(angles)]
                if k == len(angles) - 1:
                    a1 += 2.0 * math.pi
                gap = a1 - a0
                if gap > best_gap:
                    best_gap = gap
                    base = (a0 + a1) / 2.0
                    span = gap
            if span <= 0:
                span = 2.0 * math.pi

        count = max(1, total if total else slot + 1)
        if not angles and count >= 3:
            # 中心原子四周没有邻居：直接按总数均匀铺满一圈
            start = 0.0
        else:
            usable = span * 0.8
            start = base - usable / 2.0
            usable = usable
        step = (2.0 * math.pi / count) if (not angles and count >= 3) else (span * 0.8 / count)

        ang = start + step * (slot + 0.5)

        # 逐槽位避让：不允许与任何已有原子（含刚放下的氢）重叠
        for _ in range(48):
            x = host.x + math.cos(ang)
            y = host.y + math.sin(ang)
            if not self._too_close(x, y, skip_uid=host.uid):
                return x, y
            ang += math.pi / 24.0

        # 实在找不到就沿这个方向放远一点，至少不重叠
        x = host.x + math.cos(ang) * 1.2
        y = host.y + math.sin(ang) * 1.2
        return x, y

    def _too_close(self, x: float, y: float, skip_uid: int) -> bool:
        for atom in self.atoms:
            if atom.uid == skip_uid:
                continue
            if dist2(x, y, atom.x, atom.y) < 0.62:
                return True
        return False

    # ------------------------------------------------------------------
    # 空插槽方向（唯一真相来源）
    # ------------------------------------------------------------------

    def occupied_directions(self, uid: int) -> list[tuple[float, float]]:
        """该原子上「已被邻居占掉」的方向（单位向量，指向邻居）。

        返回顺序按**方位角从小到大**排列。
        不能按 (x, y) 数值排序 —— 那样 atan2 出来的角度会乱序
        （(-1,0) 的角度是 π 却排在 0 后面、(-0,-1) 是 -π/2 却排在最前），
        后续用它找"最宽角度空档"时算出的 gap 会变成负数，
        结果所有候选都落空、拖到已满原子上连提示都不给。
        """
        host = self.atom(uid)
        if host is None:
            return []
        out: list[tuple[float, float]] = []
        for n_uid in self.neighbors(uid):
            other = self.atom(n_uid)
            if other is None:
                continue
            out.append(normalize2(other.x - host.x, other.y - host.y))
        out.sort(key=lambda d: math.atan2(d[1], d[0]))
        return out

    def _base_slot_directions(self, uid: int) -> list[tuple[float, float]]:
        """空插槽方向的基础计算，**不做跨原子避让**。

        必须和 free_slot_directions 分开，否则会无限递归：
        避让逻辑要查询邻居的空位，而邻居查询又会触发避让……
        """
        host = self.atom(uid)
        if host is None:
            return []
        info = ELEMENTS.get(host.el)
        if info is None:
            return []

        free = self.free_valence(uid)
        if free <= 0:
            return []

        step_deg = SLOT_ANGLE_BY_VALENCE.get(free, 60.0)
        step = math.radians(step_deg)
        occupied = self.occupied_directions(uid)

        if not occupied:
            # 一根键都没连：按 free 等分整圈，45° 起（比水平起更自然）
            base = math.pi / 4.0
            return [
                (math.cos(base + step * (i - (free - 1) / 2.0)),
                 math.sin(base + step * (i - (free - 1) / 2.0)))
                for i in range(free)
            ]

        # 找最宽的角度空档
        best_start, best_gap, _n = _angle_gaps(occupied)

        # 只有一根键时最宽空档几乎就是整圈 —— 应当整圈等分，
        # 而不是把多个插槽挤在 180° 附近
        if best_gap >= math.radians(350.0):
            base = math.atan2(occupied[0][1], occupied[0][0]) + math.pi
            return [
                (math.cos(base + 2.0 * math.pi * i / free),
                 math.sin(base + 2.0 * math.pi * i / free))
                for i in range(free)
            ]

        mid = best_start + best_gap / 2.0
        if best_gap < step * free:
            step = best_gap / free        # 空档放不下就改为在空档内均分
        return [
            (math.cos(mid + step * (i - (free - 1) / 2.0)),
             math.sin(mid + step * (i - (free - 1) / 2.0)))
            for i in range(free)
        ]

    def free_slot_directions(self, uid: int) -> list[tuple[float, float]]:
        """该原子**空着的**插槽方向（单位向量）。

        这是修「氢原子互相重叠」的核心。规则：

          1. 还能连几根键 → 决定角间距
               4 根 → 90°（十字形）
               3 根 → 60°
               2 根 → 180°（对开）
             一律 60° 时 4 个空位刚好占满一圈，相邻碳上的空位会落到同一处，
             拖动时就会出现「氢叠在氢上」和「吸到错误的碳上」。
          2. 有已占方向时，把空位放进**最宽的角度空档**并居中排布 ——
             插槽永远指在"没有键"的那一侧。
          3. 若空位绕满整圈，还要整体旋转一个相位，**避开相邻原子的空位**，
             否则乙烷两个碳的空位落点会完全重合（实测距离 0.000 键长）。

        与旧的「固定取 SLOT_DIRS_2D 前 valence 个」相比：
        旧做法给出的方向和实际空闲方向**不一致**（碳连了一根键后，
        画出来的仍是 0/60/120/180），玩家照着亮点拖却连不上。
        """
        dirs = self._base_slot_directions(uid)
        host = self.atom(uid)
        if host is None or len(dirs) < 2:
            return dirs

        # 判断是否"绕满整圈"：相邻方向夹角都在 360/count 附近
        count = len(dirs)
        step = 2.0 * math.pi / count
        angles = sorted(math.atan2(dy, dx) for dx, dy in dirs)
        full_circle = True
        for i, a0 in enumerate(angles):
            a1 = angles[(i + 1) % len(angles)]
            if i == len(angles) - 1:
                a1 += 2.0 * math.pi
            if abs((a1 - a0) - step) > 1e-6:
                full_circle = False
                break
        if not full_circle:
            return dirs

        # 绕满整圈 → 可以整体旋转，找一个不与邻居空位撞车的相位
        others: list[tuple[float, float]] = []
        for other in self.atoms:
            if other.uid == uid:
                continue
            for d, _pt in self._neighbour_slots(other.uid):
                others.append((other.x + d[0], other.y + d[1]))
        if not others:
            return dirs

        base = angles[0]
        best_phase = 0.0
        best_score = -1.0
        samples = 24
        for s in range(samples):
            phase = 2.0 * math.pi * s / (samples * count)
            score = math.inf
            for i in range(count):
                ang = base + phase + step * i
                px = host.x + math.cos(ang)
                py = host.y + math.sin(ang)
                nearest = min((dist2(px, py, ox, oy) for ox, oy in others),
                              default=math.inf)
                score = min(score, nearest)
            if score > best_score:
                best_score = score
                best_phase = phase

        total = best_phase
        return [
            (math.cos(a + total), math.sin(a + total)) for a in angles
        ]

    def _neighbour_slots(self, uid: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """只算基础方向、不做避让的插槽（供避让逻辑查询邻居用，避免递归）。"""
        host = self.atom(uid)
        if host is None:
            return []
        return [
            (d, (host.x + d[0], host.y + d[1]))
            for d in self._base_slot_directions(uid)
        ]


    def free_slot_points(self, uid: int, length: float = 1.0
                         ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """空插槽的 (方向, 落点) 列表。

        落点 = 宿主位置 + 方向 × 键长；若两个插槽夹角过小，
        把落点沿方向外推，保证两个落点之间有足够间距（otherwise 点不中）。
        """
        host = self.atom(uid)
        if host is None:
            return []
        dirs = self.free_slot_directions(uid)
        out: list[tuple[tuple[float, float], tuple[float, float]]] = []
        min_sep = math.radians(SLOT_MIN_SEPARATION)
        n = len(dirs)
        for i, (dx, dy) in enumerate(dirs):
            reach = length
            if n > 1:
                # 找与它相邻的那个插槽，算夹角
                worst = math.pi
                for j in range(n):
                    if i == j:
                        continue
                    cosang = clamp(dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1], -1.0, 1.0)
                    worst = min(worst, math.acos(cosang))
                if worst < min_sep and worst > 1e-6:
                    # 落点间距 = 2·reach·sin(夹角/2)，要 ≥ 0.55 键长
                    need = 0.55 / (2.0 * math.sin(worst / 2.0))
                    reach = max(reach, min(need, 2.2))
            out.append(((dx, dy), (host.x + dx * reach, host.y + dy * reach)))
        return out

    # ------------------------------------------------------------------
    # 插槽吸附（拖动时的新交互核心）
    # ------------------------------------------------------------------

    def find_snap_candidate(self, moving_uid: int,
                            radius: float = SLOT_SNAP_RADIUS,
                            dir_tolerance: float = SLOT_SNAP_DIR_TOLERANCE,
                            allow_bonded_host: bool = False,
                            pull_from: tuple[float, float] | None = None
                            ) -> SnapCandidate | None:
        """拖动原子时，找出最近的「空插槽落点」候选。

        对每个其它原子 A 的每个空插槽方向 d，理想落点是 A + d×键长。
        方向来自 free_slot_directions（按可成键数自适应角间距）。

        **排序规则里 feasible 排第一位**，这是修「吸附到已满碳」的关键：
            排序键 = (不可行, 距离, 方向偏差)
        早先只按距离排序，于是「一个已满的碳」只要离得近一点，
        就会抢走本该属于旁边那个还没满的碳的吸附 ——
        玩家看到的现象就是「拖到指定位置放不下去，被识别成连到了已满的碳」。
        现在只要宿主还有空插槽，就一定优先于已满的宿主，哪怕它稍远。

        另外用 SLOT_MIN_SEPARATION 过滤掉彼此挨得太近的插槽对，
        避免出现两个落点几乎重合、点哪个都不确定的候选。

        allow_bonded_host=True（仅连键工具与拖放建键使用）：
        对**已经相连**的宿主也返回一个"升键候选"（upgrade_order 非 0），
        这样把两个已经连着的原子拖到一起，意图就是「升键」而不是再连一根。
        鼠标平移路径必须保持 False，否则挪动原子会意外升键。
        """
        moving = self.atom(moving_uid)
        if moving is None:
            return None

        best: SnapCandidate | None = None
        best_key: tuple | None = None
        min_sep = math.radians(SLOT_MIN_SEPARATION)

        for host in self.atoms:
            if host.uid == moving_uid:
                continue

            info = ELEMENTS.get(host.el)
            if info is None:
                continue

            existing_bond = self.bond_index(moving_uid, host.uid)
            if existing_bond >= 0:
                # 已经相连：只有在允许升键时，才给一个"升键候选"
                if not allow_bonded_host:
                    continue
                cur_order = self.bonds[existing_bond].order
                if cur_order >= 3:
                    continue                    # 三键到顶
                order_dist = dist2(moving.x, moving.y, host.x, host.y)
                # 升键的触发距离：要比一个键长明显更近才算"把两个原子拉到一起"。
                # 不能用 radius（0.5 键长）—— 那是"靠近某个空插槽"的判定，
                # 而升键时两者本来就相连，插槽点在 1.0 键长外永远达不到。
                # 取 0.9 键长：既能通过"拖到对方球上"触发，又不会被普通平移误触。
                if order_dist > SLOT_SNAP_RADIUS + 0.4:
                    continue
                # 只有"确实比开始时更靠近"才算升键意图。
                # 不加这一条的话，任何带键原子被挪动一下都会冒出升键候选，
                # 玩家只是想调个方向却把键级改了。
                if pull_from is not None:
                    before = dist2(pull_from[0], pull_from[1], host.x, host.y)
                    if order_dist >= before - 1e-6:
                        continue
                ok, _msg, _rm = self.free_slots_for_order(existing_bond, cur_order + 1)
                cand = SnapCandidate(
                    host_uid=host.uid, host_el=host.el,
                    point=(host.x, host.y), direction=(0.0, 0.0),
                    distance=order_dist, angle_deviation=0.0,
                    feasible=ok,
                    reason=("" if ok else "插槽不够，无法升键"),
                    slot_index=-1,
                    upgrade_order=cur_order + 1,
                )
                key = (0 if ok else 1, order_dist, 0.0)
                if best_key is None or key < best_key:
                    best_key = key
                    best = cand
                continue

            free = self.free_valence(host.uid)
            # P4：feasible 不能只看宿主 —— 被拖的原子自己满价时，
            # 吸过去也连不上（实测：满价碳被吸到插槽点、键却没加上）。
            moving_free = self.free_valence(moving_uid)
            feasible = free > 0 and moving_free > 0
            reason = ""
            if not feasible:
                if moving_free <= 0:
                    reason = f"{moving.el}自己的价键已用满，不能再连"
                else:
                    reason = f"{info['name']}的 {info['valence']} 个价键已用满"

            # 落点来源要分「宿主满」和「被拖的原子满」两种情况：
            #   宿主有空位 → 用真正的空插槽点（哪怕 moving 自己满价，
            #                   也要给出候选，好让界面提示原因）；
            #   宿主满了   → 用「最宽空档」估一个方向，仅用于提示。
            # 早先这里按 feasible 分支 —— moving 满价时 feasible 也是 False，
            # 结果走成了「宿主已满」的提示路径，拿到的方向对孤立原子只有
            # 一个 (1,0)，落点离插槽 1.414 键长，候选根本产生不出来。
            if free > 0:
                slots = self.free_slot_points(host.uid)
            else:
                dirs = self._relaxed_directions(host.uid)
                slots = [((dx, dy), (host.x + dx, host.y + dy)) for dx, dy in dirs]

            for slot_index, ((dx, dy), point) in enumerate(slots):
                dist = dist2(moving.x, moving.y, point[0], point[1])
                if dist > radius:
                    continue

                # 方向偏差：原子相对宿主的方向，与插槽方向的夹角
                rel = normalize2(moving.x - host.x, moving.y - host.y)
                cosang = clamp(rel[0] * dx + rel[1] * dy, -1.0, 1.0)
                dev = math.degrees(math.acos(cosang))
                if dev > dir_tolerance:
                    continue

                # 与同宿主其它插槽夹角过小的，跳过（两个落点会挤在一起）。
                # 这里比较的是**插槽之间**的夹角，不是插槽与已占键的夹角 ——
                # 插槽与已占键的夹角本来就可能是 180°（对开），那是正常的。
                # 只在有空位、且空位多于两个时才需要判重叠。
                if feasible and len(slots) > 2:
                    too_close = False
                    for j, ((ox, oy), _op) in enumerate(slots):
                        if j == slot_index:
                            continue
                        cos_o = clamp(dx * ox + dy * oy, -1.0, 1.0)
                        if math.acos(cos_o) < min_sep:
                            too_close = True
                            break
                    if too_close:
                        continue

                key = (0 if feasible else 1, dist, dev)
                if best_key is None or key < best_key:
                    best_key = key
                    best = SnapCandidate(
                        host_uid=host.uid,
                        host_el=host.el,
                        point=point,
                        direction=(dx, dy),
                        distance=dist,
                        angle_deviation=dev,
                        feasible=feasible,
                        reason=reason,
                        slot_index=slot_index,
                    )

        return best

    def _relaxed_directions(self, uid: int) -> list[tuple[float, float]]:
        """给「已满」的宿主算一个"离空插槽最近"的方向，仅用于提示。

        已满的原子没有空插槽，但玩家把原子拖过去时仍需要一条虚线告诉他
        「这里连不上」。做法：找最宽空档的中点方向（不管还剩几根键）。
        """
        host = self.atom(uid)
        if host is None:
            return []
        occupied = self.occupied_directions(uid)
        if not occupied:
            return [(1.0, 0.0)]
        best_start, best_gap, _n = _angle_gaps(occupied)
        mid = best_start + best_gap / 2.0
        return [(math.cos(mid), math.sin(mid))]


    def apply_snap(self, moving_uid: int, candidate: SnapCandidate) -> tuple[bool, str]:
        """把原子精确吸附到候选落点并成键。

        P4 修复：add_bond 失败时要把位置退回 —— 早先先挪位置、后成键，
        失败不回滚，原子停在插槽点上却没连上（实测满价碳就是这样）。
        """
        moving = self.atom(moving_uid)
        if moving is None:
            return False, "原子不存在"
        if not candidate.feasible:
            return False, candidate.reason or "插槽不足"
        prev = (moving.x, moving.y)
        moving.x, moving.y = candidate.point
        self._invalidate_geometry()
        ok, msg = self.add_bond(moving_uid, candidate.host_uid, 1)
        if not ok:
            moving.x, moving.y = prev
            self._invalidate_geometry()
            return False, msg
        # 新键改变了双方的插槽方向 → 把已有的氢重新摆正
        self.tidy_hydrogens(moving_uid)
        self.tidy_hydrogens(candidate.host_uid)
        return True, ""

    # ------------------------------------------------------------------
    # 识别
    # ------------------------------------------------------------------

    def to_molecule(self) -> Molecule:
        """导出成化学引擎用的分子图（下标重新编号）。"""
        from .chem.graph import Bond

        index = {a.uid: i for i, a in enumerate(self.atoms)}
        mol = Molecule()
        for atom in self.atoms:
            mol.atoms.append(_chem_atom(atom))
        for bond in self.bonds:
            if bond.a in index and bond.b in index:
                mol.bonds.append(Bond(index[bond.a], index[bond.b], bond.order))
        return mol

    def identify(self) -> tuple[list, int]:
        """识别画布上的所有组分。

        返回 (各组分识别结果列表, 总原子数)。大组分排在前面。
        """
        from .chem.graph import connected_components, subgraph
        from .chem.identify import identify_molecule

        mol = self.to_molecule()
        if not mol.atoms:
            return [], 0
        parts = []
        for comp in connected_components(mol):
            sub = subgraph(mol, comp)
            parts.append(identify_molecule(sub))
        parts.sort(key=lambda r: -r.count)
        return parts, len(mol.atoms)

    # ------------------------------------------------------------------
    # 3D 几何
    # ------------------------------------------------------------------

    def _invalidate_geometry(self) -> None:
        for atom in self.atoms:
            atom.pos3 = None
        if self.mode == "view3d":
            self.mode = "build"

    def invalidate_geometry(self) -> None:
        """公开版本：界面层（画布）在直接改动坐标后需要调用。"""
        self._invalidate_geometry()

    def has_geometry(self) -> bool:
        return bool(self.atoms) and all(a.pos3 is not None for a in self.atoms)

    def solve_geometry(self) -> tuple[bool, str, float]:
        """求解 3D 构型，结果写入每个原子的 pos3。

        必须在 auto_fill_hydrogens 之后调用。
        返回 (是否成功, 失败原因, 最大半径)。
        """
        mol = self.to_molecule()
        if not mol.atoms:
            return False, "还没有原子", 1.0
        if len(mol.atoms) > RELAX_MAX_ATOMS:
            return False, (f"原子太多（{len(mol.atoms)} 个），"
                           f"超过 {RELAX_MAX_ATOMS} 个后不再自动计算构型"), 1.0
        result = chem_solve(mol)
        if result is None:
            return False, "构型求解失败", 1.0
        for atom, pos in zip(self.atoms, result.positions):
            atom.pos3 = pos
        self.mode = "view3d"
        return True, "", result.radius

    # ------------------------------------------------------------------
    # 带键拖动：绕锚点整体旋转（保持键长）
    # ------------------------------------------------------------------

    def mobile_group(self, uid: int, anchor_uid: int) -> list[int]:
        """拖动 uid 时要一起动的原子集合（含 uid 自己）。

        做法：从 uid 出发 BFS，**不穿过 anchor_uid**。
        得到的就是「挂在 uid 这一侧的整棵支链」——
        把它绕 anchor 刚体旋转，组内所有键长、键角都天然保持不变。

        为什么这样做：玩家拖一个带键的原子，想要的是「改方向」，
        不是把键拉长。刚性旋转是唯一能同时保证「方向跟手」与「键长不变」的做法。
        """
        seen = {anchor_uid, uid}
        group = [uid]
        queue = [uid]
        # 邻居查询用一次 BFS 覆盖整个支链
        adjacency: dict[int, list[int]] = {a.uid: self.neighbors(a.uid) for a in self.atoms}
        while queue:
            cur = queue.pop(0)
            for nxt in adjacency.get(cur, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                group.append(nxt)
                queue.append(nxt)
        return group

    def rotate_group(self, uid: int, anchor_uid: int, angle: float) -> int:
        """把 uid 所在支链绕 anchor 旋转 angle 弧度（锚点不动）。返回移动的原子数。

        键长由几何保证不变：刚体旋转不改变任何两点距离。
        """
        anchor = self.atom(anchor_uid)
        moving = self.atom(uid)
        if anchor is None or moving is None:
            return 0
        group = self.mobile_group(uid, anchor_uid)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for gid in group:
            atom = self.atom(gid)
            if atom is None:
                continue
            rx, ry = atom.x - anchor.x, atom.y - anchor.y
            atom.x = anchor.x + rx * cos_a - ry * sin_a
            atom.y = anchor.y + rx * sin_a + ry * cos_a
        self._invalidate_geometry()
        return len(group)

    def snapshot_positions(self) -> dict[int, tuple[float, float]]:
        """只记平面坐标的轻量快照（旋转拖动用，不涉及键）。"""
        return {a.uid: (a.x, a.y) for a in self.atoms}

    def restore_geometry_positions(self, snap: dict[int, tuple[float, float]] | None) -> None:
        """把平面坐标恢复成给定快照。

        拖动带键原子时每次 move 都从「起始姿态」重新算旋转量，
        而不是在上一帧结果上累加 —— 累加会让浮点误差越滚越大，
        玩家来回拖几次之后键长就悄悄变了。
        """
        if not snap:
            return
        for atom in self.atoms:
            pos = snap.get(atom.uid)
            if pos is not None:
                atom.x, atom.y = pos

    def bond_length_of(self, a_uid: int, b_uid: int) -> float:
        """两个原子当前的实际距离（用来断言"键长有没有变"）。"""
        a, b = self.atom(a_uid), self.atom(b_uid)
        if a is None or b is None:
            return 0.0
        return dist2(a.x, a.y, b.x, b.y)

    # ------------------------------------------------------------------
    # 连键并摆正
    # ------------------------------------------------------------------

    def connect_atoms(self, src_uid: int, dst_uid: int, order: int = 1
                      ) -> tuple[bool, str, int]:
        """把 src 连到 dst 上，并把 src 摆到 dst 的插槽精确位置上。

        为什么要"摆正"：两个原子相距多远都能被连上（玩家可能拖得比较随意），
        但连完之后必须落在**标准键长**上，否则 2D 结构会越来越乱、
        立起来时的求解器也要花力气纠回来。
        落点 = dst 位置 + 插槽方向 × 键长，天然就是标准键长。

        返回 (是否成功, 消息, 被移动的原子 uid 或 -1)。
        如果两者已经相连，走升键路径（单→双→三）。
        """
        src, dst = self.atom(src_uid), self.atom(dst_uid)
        if src is None or dst is None:
            return False, "原子不存在", -1
        if src_uid == dst_uid:
            return False, "不能把原子连到自己", -1

        if self.bond_index(src_uid, dst_uid) >= 0:
            ok, msg = self.add_bond(src_uid, dst_uid, order)
            return ok, msg, -1

        # 找一个朝向 src 的空插槽（不要求 src 已经在附近）
        best_dir = None
        best_score = math.inf
        rel = normalize2(src.x - dst.x, src.y - dst.y)
        for dx, dy in self.free_slot_directions(dst_uid):
            cosang = clamp(rel[0] * dx + rel[1] * dy, -1.0, 1.0)
            dev = math.acos(cosang)
            if dev < best_score:
                best_score = dev
                best_dir = (dx, dy)
        if best_dir is None:
            info = ELEMENTS.get(dst.el)
            name = info["name"] if info else dst.el
            return False, f"{name}的价键已用满，连不上了", -1

        # 把 src 整条支链平移到落点（平移不改变 src 支链内部的键长键角）
        target = (dst.x + best_dir[0], dst.y + best_dir[1])
        group = set(self.mobile_group(src_uid, dst_uid))
        dx_move, dy_move = target[0] - src.x, target[1] - src.y
        for gid in group:
            atom = self.atom(gid)
            if atom is not None:
                atom.x += dx_move
                atom.y += dy_move

        ok, msg = self.add_bond(src_uid, dst_uid, order)
        if not ok:
            # 成键失败就把位置退回去，别留下一个被挪走的原子
            for gid in group:
                atom = self.atom(gid)
                if atom is not None:
                    atom.x -= dx_move
                    atom.y -= dy_move
            self._invalidate_geometry()
            return False, msg, -1
        self._invalidate_geometry()
        # 新键改变了双方的插槽方向 → 把已有的氢重新摆正
        self.tidy_hydrogens(src_uid)
        self.tidy_hydrogens(dst_uid)
        return True, "", src_uid

    # ------------------------------------------------------------------
    # 键级：升键时自动腾出插槽
    # ------------------------------------------------------------------

    def free_slots_for_order(self, bond_idx: int, order: int
                             ) -> tuple[bool, str, list[int]]:
        """升键前不够插槽时，移除该键两端**自动补的氢**来腾出位置。

        这是「乙烷 → 乙烯」这条最自然的路径能被走通的关键：
            乙烷补满氢后两个碳各有 4 根 C—H 键、剩余插槽为 0，
            想把 C—C 升成双键会被化合价校验拒绝
            （实测报错：碳上没有足够的空插槽（还差 1 个））。
        化学上正确的做法就是：升键的同时脱掉相应数目的氢。

        只动 auto_h=True 的氢 —— 玩家手动放的氢与其它重原子一律不碰。

        **先试算、再执行**：只有确认两个原子都腾得出足够插槽时才真的删原子。
        不这样做的话，中途发现腾不出来就得回滚已删的原子 ——
        而删掉的原子没法可靠还原（uid 会变），会留下半个分子。
        所以这里分两趟：第一趟只数，第二趟才删。

        返回 (是否足够, 消息, 被移除的 uid 列表)。
        """
        if not (0 <= bond_idx < len(self.bonds)):
            return False, "键不存在", []
        bond = self.bonds[bond_idx]
        order = int(clamp(order, 1, 3))
        delta = order - bond.order
        if delta <= 0:
            return True, "", []

        # ---- 第一趟：试算，只数不删 ----
        plan: list[tuple[int, list[int]]] = []      # (宿主 uid, 要删的氢 uid 列表)
        for uid in (bond.a, bond.b):
            atom = self.atom(uid)
            if atom is None:
                continue
            need = delta - self.free_valence(uid)
            if need <= 0:
                plan.append((uid, []))
                continue
            auto_h = [n for n in self.neighbors(uid)
                      if (lambda o: o is not None and o.el == "H" and o.auto_h)(self.atom(n))]
            if len(auto_h) < need:
                info = ELEMENTS.get(atom.el)
                name = info["name"] if info else atom.el
                missing = need - len(auto_h)
                return False, (f"{name}上还差 {missing} 个插槽："
                               f"请先删掉手动放置的氢，或改用别的原子"), []
            plan.append((uid, auto_h[:need]))

        # ---- 第二趟：确认可行后才真的删 ----
        removed: list[int] = []
        for _uid, victims in plan:
            for v_uid in victims:
                if self.remove_atom(v_uid):
                    removed.append(v_uid)
        self._invalidate_geometry()
        return True, "", removed

    def tidy_hydrogens(self, uid: int) -> int:
        """把某个原子上的氢摆到"除重原子邻居之外"的空插槽方向上。返回动了几个。

        为什么需要这一步（实测问题）：
            补氢时氢的位置是"当时"按插槽算出来的，但之后键级一变
            （乙烷 → 乙炔），插槽方向就变了，氢却还留在原地 ——
            乙炔于是画成两条折线，"四个原子排成一条直线"这个最重要的
            特征在 2D 里完全看不出来（实测残留方向 -32° 和 86°）。

        实现要点：**不能直接用 free_slot_points** —— 已经挂着的氢本身
        就是一个"被占用的方向"，算出来只剩 0 个空插槽，什么也摆不动。
        所以这里只把**重原子邻居**当作已占方向，剩下留给氢。
        """
        host = self.atom(uid)
        if host is None:
            return 0
        hydrogens = [n for n in self.neighbors(uid)
                     if (lambda o: o is not None and o.el == "H")(self.atom(n))]
        if not hydrogens:
            return 0

        heavy_dirs: list[tuple[float, float]] = []
        heavy_used = 0
        for n_uid in self.neighbors(uid):
            other = self.atom(n_uid)
            if other is None or other.el == "H":
                continue
            heavy_dirs.append(normalize2(other.x - host.x, other.y - host.y))
            bond = self.bonds[self.bond_index(uid, n_uid)]
            heavy_used += bond.order

        info = ELEMENTS.get(host.el)
        if info is None:
            return 0
        # 这个碳一共能带几个氢 = 化合价 - 与**重原子**之间的键用量。
        # 注意不能用 free_valence()：它把 C—H 键也算进"已用"，
        # 于是乙炔那个碳算出来是 0 个空位，一个氢都摆不动（实测正是如此）。
        capacity = max(0, info["valence"] - heavy_used)
        if capacity <= 0:
            return 0
        free = capacity

        # 角间距由**杂化态**决定，而不是"空档里塞得下几个"：
        #   sp  （有氢时才可能，这里是三键）→ 与重键共线 180°
        #   sp² （有双键）→ 与重键成 120°（平面三角形，乙烯要画成"<"形）
        #   sp³         → 四面体角投影 109.47°
        # 同一条「与重键的夹角」规则也适用于唯一的重键方向。
        has_triple = any(
            self.bonds[self.bond_index(uid, n)].order >= 3
            for n in self.neighbors(uid)
            if (lambda o: o is not None and o.el != "H")(self.atom(n)))
        has_double = any(
            self.bonds[self.bond_index(uid, n)].order == 2
            for n in self.neighbors(uid)
            if (lambda o: o is not None and o.el != "H")(self.atom(n)))
        if has_triple:
            bond_angle = 180.0
        elif has_double:
            bond_angle = 120.0
        else:
            bond_angle = 109.47
        step = math.radians(bond_angle)

        if not heavy_dirs:
            base = math.pi / 4.0
            dirs = [(math.cos(base + step * (i - (free - 1) / 2.0)),
                     math.sin(base + step * (i - (free - 1) / 2.0)))
                    for i in range(free)]
        elif len(heavy_dirs) == 1:
            # 只有一根重键：氢就按杂化角分布在这根键的两侧。
            # 这才是"乙炔画成直线、乙烯画成＜形"的关键 ——
            # 用"最宽空档"去分（60° 间距）会把乙烯的两个氢摆成共线。
            axis = math.atan2(heavy_dirs[0][1], heavy_dirs[0][0])
            dirs = []
            for i in range(free):
                # free=1 → 正好反向；free=2 → ±120°；free=3 → ±109.47° + 反向
                if free == 1:
                    offset = math.pi
                else:
                    offset = step * (i - (free - 1) / 2.0) + math.pi
                if free == 3:
                    offset = math.pi + step * (i - 1)
                dirs.append((math.cos(axis + offset), math.sin(axis + offset)))
        else:
            best_start, best_gap, _n = _angle_gaps(heavy_dirs)
            if best_gap >= math.radians(350.0):
                base = math.atan2(heavy_dirs[0][1], heavy_dirs[0][0]) + math.pi
                dirs = [(math.cos(base + 2.0 * math.pi * i / free),
                         math.sin(base + 2.0 * math.pi * i / free))
                        for i in range(free)]
            else:
                mid = best_start + best_gap / 2.0
                use = step if best_gap >= step * free else best_gap / free
                dirs = [(math.cos(mid + use * (i - (free - 1) / 2.0)),
                         math.sin(mid + use * (i - (free - 1) / 2.0)))
                        for i in range(free)]

        moved = 0
        for h_uid, (dx, dy) in zip(hydrogens, dirs):
            h = self.atom(h_uid)
            if h is None:
                continue
            h.x, h.y = host.x + dx, host.y + dy
            moved += 1
        return moved

    def tidy_all_hydrogens(self) -> int:
        """对所有重原子做一遍 tidy_hydrogens。结构变化后调用。"""
        moved = 0
        for atom in list(self.atoms):
            if atom.el != "H":
                moved += self.tidy_hydrogens(atom.uid)
        return moved

    def break_all_bonds(self, uid: int) -> int:
        """把该原子身上的键全部断开，返回断掉的键数。

        用于「拖动超过阈值自动断键」：玩家把原子拖远，就是想把这块「拔下来」
        换到别处连，而不是把整个分子拖变形。
        只断键、不动其它原子 —— 分子的其余部分留在原位。
        """
        before = len(self.bonds)
        self.bonds = [b for b in self.bonds if b.a != uid and b.b != uid]
        removed = before - len(self.bonds)
        if removed:
            self._invalidate_geometry()
        return removed

    # ------------------------------------------------------------------
    # 撤销 / 重做
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        snap = {
            "atoms": [(a.uid, a.el, a.x, a.y, a.pos3, a.auto_h) for a in self.atoms],
            "bonds": [(b.a, b.b, b.order) for b in self.bonds],
            "mode": self.mode,
            "next_uid": self._next_uid,
        }
        # 相机状态（由画布注册的编解码器提供，化学层不关心内容）。
        # 测试报告 P5：3D 下撤销后相机残留透视角度，平面被投影变形 ——
        # 把相机纳入快照后，撤销/重做连视角一起恢复。
        if self._state_get is not None:
            try:
                snap["camera"] = self._state_get()
            except Exception:            # noqa: BLE001 相机问题不该毁掉快照
                pass
        return snap

    def restore(self, snap: dict) -> None:
        self.atoms = [
            WorldAtom(uid=u, el=el, x=x, y=y, pos3=p, auto_h=h)
            for (u, el, x, y, p, h) in snap["atoms"]
        ]
        self.bonds = [WorldBond(a, b, o) for (a, b, o) in snap["bonds"]]
        self.mode = snap["mode"]
        self._next_uid = snap["next_uid"]
        if "camera" in snap and self._state_set is not None:
            try:
                self._state_set(snap["camera"])
            except Exception:            # noqa: BLE001
                pass
        # 注意：这里**不能**调 _invalidate_geometry() —— 它会把手动把
        # mode 拨回 build、清掉 pos3，3D 状态的撤销/重做就被毁掉了。
        # 快照里的 pos3 / mode 就是权威值，原样恢复即可。

    def set_state_codec(self, getter, setter) -> None:
        """注册「世界之外的附加状态」编解码器（相机用它）。

        化学层不知道相机是什么 —— 只负责在快照里带上/恢复一段
        由调用方解释的状态。画布在构造时注册相机编解码器。
        """
        self._state_get = getter
        self._state_set = setter

    # 兼容旧的私有名字（canvas 里在用）
    _snapshot = snapshot
    _restore = restore

    def push_undo(self) -> None:
        """记录一个撤销点。任何新操作都会清空重做栈（标准语义）。"""
        self.commit_undo(self.snapshot())

    def commit_undo(self, before: dict) -> None:
        """统一入口：把「操作前快照」提交成撤销点。

        stand_up 这类「先快照、再执行、后提交」的操作用它 ——
        早先它直接 `_undo.append(before)`，绕过了清重做栈和 40 条上限
        （测试报告 P3：立起来后 Ctrl+Y 会回放旧操作、撤销栈无限增长）。
        """
        self._undo.append(before)
        if len(self._undo) > UNDO_STACK_SIZE:
            self._undo.pop(0)
        self._redo.clear()

    def replace_last_undo(self, snap: dict) -> None:
        """把撤销栈顶替换成给定快照。

        拖动断键时用：撤销点必须记成「断键之后」的状态，
        这样 Ctrl+Z 恢复的是断掉的键，而不是把原子搬回原位
        （后者会让玩家以为撤销坏了）。
        """
        if self._undo:
            self._undo[-1] = snap
        else:
            self._undo.append(snap)

    def pop_last_undo_if_unchanged(self, expected: dict) -> bool:
        """若撤销栈顶就是 expected 这条快照，把它弹掉。

        拖动最终没有改变任何东西时（位置被恢复、也没断键）用：
        不该在撤销栈里留一条空记录，否则玩家按 Ctrl+Z 会觉得"没反应"。
        比较的是内容而不是身份 —— 栈顶可能是被 replace_last_undo 换过的。
        """
        if not self._undo:
            return False
        if self._undo[-1] == expected:
            self._undo.pop()
            return True
        return False

    def undo(self) -> bool:
        if not self._undo:
            return False
        snap = self._undo.pop()
        self._redo.append(self.snapshot())
        if len(self._redo) > REDO_STACK_SIZE:
            self._redo.pop(0)
        self.restore(snap)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        snap = self._redo.pop()
        self._undo.append(self.snapshot())
        if len(self._undo) > UNDO_STACK_SIZE:
            self._undo.pop(0)
        self.restore(snap)
        return True

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_depth(self) -> int:
        return len(self._undo)

    def redo_depth(self) -> int:
        return len(self._redo)

    def clear_history(self) -> None:
        self._undo.clear()
        self._redo.clear()


def _chem_atom(atom: WorldAtom):
    from .chem.graph import Atom
    return Atom(el=atom.el, x=atom.x, y=atom.y, pos3=atom.pos3, auto_h=atom.auto_h)
