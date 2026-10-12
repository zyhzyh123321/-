"""分子识别：粗筛哈希 + 精确图同构 + 分子式/质量/组成。

为什么不追求「完美的规范串」：
    让规范串对同构不变是件很难的事（对称环、等价原子的先后顺序、标记编码都会踩坑）。
    而识别根本不需要它：
        1. 用迭代 WL 哈希 + 分子式做粗筛，把候选缩到几条
        2. 对候选做真正的图同构判定，命中才算成功
    这样「正确性」来自同构判定本身，而不是来自某个精巧的字符串。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .constants import ATOMIC_MASS, ELEMENTS
from .graph import Molecule, build_adjacency
from .parse_spec import ParseError, parse_spec

# ----------------------------------------------------------------------------
# 重原子骨架
# ----------------------------------------------------------------------------


@dataclass
class HeavyGraph:
    """重原子骨架 + 每个重原子的氢数。

    所有比较都用这个统一表示：把「含显式氢的分子图」压成
    「重原子图 + 氢数」，于是显式氢和隐式氢两种写法可以互通。
    """

    els: list[str]
    h_count: list[int]
    adj: list[list[tuple[int, int]]]     # [(邻居, 键级)]
    lone_h: int = 0                      # 游离氢（没连到任何重原子）

    @property
    def n(self) -> int:
        return len(self.els)

    @property
    def edge_count(self) -> int:
        return sum(len(a) for a in self.adj)


def heavy_graph(mol: Molecule) -> HeavyGraph:
    """把分子图压成重原子骨架 + 氢数。"""
    heavy = [i for i, a in enumerate(mol.atoms) if a.el != "H"]
    pos_of = {orig: new for new, orig in enumerate(heavy)}
    n = len(heavy)
    h_count = [0] * n
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]

    for bond in mol.bonds:
        order = bond.order or 1
        ea, eb = mol.atoms[bond.a].el, mol.atoms[bond.b].el
        if ea == "H" and eb == "H":
            continue
        if ea == "H":
            h_count[pos_of[bond.b]] += 1
            continue
        if eb == "H":
            h_count[pos_of[bond.a]] += 1
            continue
        pa, pb = pos_of[bond.a], pos_of[bond.b]
        adj[pa].append((pb, order))
        adj[pb].append((pa, order))

    for lst in adj:
        lst.sort()

    lone_h = 0
    for i, atom in enumerate(mol.atoms):
        if atom.el != "H":
            continue
        attached = any(b.a == i or b.b == i for b in mol.bonds)
        if not attached:
            lone_h += 1

    return HeavyGraph(els=[mol.atoms[i].el for i in heavy], h_count=h_count, adj=adj, lone_h=lone_h)


# ----------------------------------------------------------------------------
# 粗筛键
# ----------------------------------------------------------------------------


def _hash32(text: str) -> str:
    h = 5381
    for ch in text:
        h = ((h * 33) ^ ord(ch)) & 0xFFFFFFFF
    return base36(h)


def base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


def coarse_key(g: HeavyGraph) -> str:
    """粗筛键：固定轮数迭代 WL 哈希的多重集。

    每轮都把字符串压成 32 位哈希值 —— 否则签名长度每轮翻倍，
    几轮之后就会膨胀到不可接受（网页版实测整库跑不完）。
    """
    n = g.n
    if n == 0:
        return f"H-only({g.lone_h})"

    cur: list[str] = []
    for v in range(n):
        nb = sorted(f"{order}:{g.els[to]}" for to, order in g.adj[v])
        cur.append(_hash32(f"{g.els[v]}({g.h_count[v]}):{','.join(nb)}"))

    acc = ".".join(sorted(cur))
    for _ in range(3):
        nxt: list[str] = []
        for v in range(n):
            nb = sorted(f"{order}:{cur[to]}" for to, order in g.adj[v])
            nxt.append(_hash32(f"{cur[v]}|{','.join(nb)}"))
        cur = nxt
        acc += ";" + ".".join(sorted(cur))

    return acc + (f"|loneH({g.lone_h})" if g.lone_h else "")


# ----------------------------------------------------------------------------
# 精确同构判定
# ----------------------------------------------------------------------------


def isomorphic(g1: HeavyGraph, g2: HeavyGraph, node_budget: int = 200_000) -> bool:
    """两个重原子骨架是否表示同一个结构。

    这是识别的正确性来源。带回溯的一一映射，配合
    「(元素, 氢数, 度数, 已用键级)」标签分组来缩小候选。

    必须双侧检查：只查一侧会漏掉「同一对原子有多重连接」的差异。
    """
    n = g1.n
    if n != g2.n or g1.lone_h != g2.lone_h or g1.edge_count != g2.edge_count:
        return False
    if n == 0:
        return True

    def label(g: HeavyGraph, v: int) -> tuple:
        used = sum(order for _, order in g.adj[v])
        return (g.els[v], g.h_count[v], len(g.adj[v]), used)

    lab1 = [label(g1, v) for v in range(n)]
    lab2 = [label(g2, v) for v in range(n)]
    if sorted(lab1) != sorted(lab2):
        return False

    candidates: list[list[int]] = []
    for v in range(n):
        lst = [w for w in range(n) if lab2[w] == lab1[v]]
        if not lst:
            return False
        candidates.append(lst)

    mapping = [-1] * n
    used = [False] * n
    # 候选最少的原子最先处理（约束最强，剪枝最快）
    visit_order = sorted(range(n), key=lambda v: len(candidates[v]))

    # 预先把邻接表转成字典，便于 O(1) 查键级
    adj1 = [{to: o for to, o in g1.adj[v]} for v in range(n)]
    adj2 = [{to: o for to, o in g2.adj[v]} for v in range(n)]
    # 反向映射：g2 的下标 → g1 的下标，避免每次都线性扫描 mapping
    reverse = [-1] * n

    nodes = 0

    def compatible(u: int, v: int) -> bool:
        if g1.els[u] != g2.els[v] or g1.h_count[u] != g2.h_count[v]:
            return False
        if len(g1.adj[u]) != len(g2.adj[v]):
            return False
        # g1 侧：u 的每个已映射邻居，必须在 g2[v] 里有同键级的对应
        for to, order_ in g1.adj[u]:
            w = mapping[to]
            if w < 0:
                continue
            if adj2[v].get(w) != order_:
                return False
        # g2 侧：v 的每个已映射邻居，必须在 g1[u] 里有同键级的对应
        for to, order_ in g2.adj[v]:
            w = reverse[to] if to < n else -1
            if w < 0:
                continue
            if adj1[w].get(u) != order_:
                return False
        return True

    def verify_all() -> bool:
        edges = 0
        for u in range(n):
            v = mapping[u]
            for to, order_ in g1.adj[u]:
                if adj2[v].get(mapping[to]) != order_:
                    return False
                edges += 1
        return edges > 0 or n == 1

    def search(k: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return False
        if k == n:
            return verify_all()
        u = visit_order[k]
        for v in candidates[u]:
            if used[v] or not compatible(u, v):
                continue
            mapping[u] = v
            used[v] = True
            reverse[v] = u
            if search(k + 1):
                return True
            mapping[u] = -1
            used[v] = False
            reverse[v] = -1
        return False

    return search(0)


# ----------------------------------------------------------------------------
# 分子式 / 相对分子质量 / 元素组成
# ----------------------------------------------------------------------------


def formula_of(mol: Molecule) -> str:
    """Hill 顺序分子式：C 第一、H 第二、其余按字母序。"""
    counts: dict[str, int] = {}
    for atom in mol.atoms:
        counts[atom.el] = counts.get(atom.el, 0) + 1

    def sort_key(sym: str):
        if sym == "C":
            return (0, "")
        if sym == "H":
            return (1, "")
        return (2, sym)

    parts = []
    for sym in sorted(counts, key=sort_key):
        n = counts[sym]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)


def molar_mass(mol: Molecule) -> float | None:
    """相对分子质量（保留两位小数）。"""
    total = 0.0
    for atom in mol.atoms:
        m = ATOMIC_MASS.get(atom.el)
        if m is None:
            return None
        total += m
    return round(total, 2)


def composition(mol: Molecule) -> list[dict]:
    """元素质量百分比，按质量降序。"""
    mass: dict[str, float] = {}
    total = 0.0
    for atom in mol.atoms:
        m = ATOMIC_MASS.get(atom.el, 0.0)
        mass[atom.el] = mass.get(atom.el, 0.0) + m
        total += m

    out = []
    for sym, m in mass.items():
        info = ELEMENTS.get(sym, {})
        out.append({
            "symbol": sym,
            "name": info.get("name", sym),
            "mass": round(m, 2),
            "percent": (m / total * 100.0) if total > 0 else 0.0,
        })
    out.sort(key=lambda d: -d["percent"])
    return out


def composition_summary(mol: Molecule) -> str:
    return " · ".join(f"{c['symbol']} {c['percent']:.1f}%" for c in composition(mol))


def element_counts(mol: Molecule) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in mol.atoms:
        counts[atom.el] = counts.get(atom.el, 0) + 1
    return counts


# ----------------------------------------------------------------------------
# 库索引
# ----------------------------------------------------------------------------

_INDEX: dict[str, list[tuple[dict, HeavyGraph]]] | None = None


class _Record:
    """库里的记录 + 它的重原子骨架（解析一次后缓存）。"""

    __slots__ = ("data", "graph")

    def __init__(self, data: dict, graph: HeavyGraph) -> None:
        self.data = data
        self.graph = graph


def build_index(force: bool = False) -> dict[str, list[_Record]]:
    """建立库索引：粗筛键 → 候选记录列表。

    惰性解析结构式：库里的 spec 字符串在首次识别时才解析成分子图。
    网页版这里踩过坑 —— 索引只处理「已解析过」的记录，
    而游戏运行过程中没人解析过它们，结果索引为空、所有分子都识别不出来。
    """
    global _INDEX
    if _INDEX is not None and not force:
        return _INDEX

    from ..data.molecules import MOLECULES

    index: dict[str, list[_Record]] = {}
    for rec in MOLECULES:
        try:
            mol = parse_spec(rec["spec"])
        except ParseError:
            continue
        g = heavy_graph(mol)
        index.setdefault(coarse_key(g), []).append(_Record(rec, g))
    _INDEX = index
    return index


def reset_index() -> None:
    global _INDEX
    _INDEX = None


# ----------------------------------------------------------------------------
# 识别
# ----------------------------------------------------------------------------


@dataclass
class IdentifyResult:
    key: str
    formula: str
    mass: float | None
    count: int
    composition: list[dict]
    matched: dict | None = None
    loose_atom: bool = False


def identify_molecule(mol: Molecule) -> IdentifyResult:
    """识别一个分子图（单个连通分量）。

    流程：粗筛键命中候选 → 逐条精确同构匹配 → 命中才算识别成功。
    库只有几十条，候选通常 1~3 条，开销可以忽略。
    """
    g = heavy_graph(mol)
    res = IdentifyResult(
        key=coarse_key(g),
        formula=formula_of(mol),
        mass=molar_mass(mol),
        count=len(mol.atoms),
        composition=composition(mol),
        matched=None,
    )
    res.loose_atom = len(mol.atoms) == 1 and mol.atoms[0].el == "H" and not mol.bonds

    bucket = build_index().get(res.key)
    if bucket:
        for rec in bucket:
            if isomorphic(g, rec.graph):
                res.matched = rec.data
                break
    return res
