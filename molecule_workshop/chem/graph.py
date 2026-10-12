"""分子图数据结构与通用图算法。

本文件不含任何化学规则，只处理「原子 + 键」的图结构。

坐标约定：编辑平面用二维坐标 (x, y)；立起来后三维坐标存在原子的 pos3 里，
因此反复「立起来 / 重新编辑」不会破坏玩家摆好的平面布局。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .vector import Vec3

# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------


@dataclass
class Atom:
    """一个原子。

    el      元素符号
    x, y    编辑平面坐标
    pos3    三维坐标（立起来后才有效）
    auto_h  是否由「自动补氢」创建（删除时会优先清理它）
    h_count 结构式里显式写出的氢数（仅解析库中结构时使用，None 表示未写）
    """

    el: str
    x: float = 0.0
    y: float = 0.0
    pos3: Vec3 | None = None
    auto_h: bool = False
    h_count: int | None = None
    uid: int = 0


@dataclass
class Bond:
    """一根键。a / b 是原子在 Molecule.atoms 里的下标。"""

    a: int
    b: int
    order: int = 1


@dataclass
class Molecule:
    """分子图（索引式：键用下标指向原子）。"""

    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.atoms)

    def copy(self) -> "Molecule":
        return Molecule(
            atoms=[Atom(a.el, a.x, a.y, a.pos3, a.auto_h, a.h_count, a.uid) for a in self.atoms],
            bonds=[Bond(b.a, b.b, b.order) for b in self.bonds],
        )


# ----------------------------------------------------------------------------
# 邻接结构
# ----------------------------------------------------------------------------


@dataclass
class Adjacency:
    """邻接表：adj[v] = [(邻居下标, 键级), ...]（已按下标排序，保证确定性）。"""

    adj: list[list[tuple[int, int]]]

    def __len__(self) -> int:
        return len(self.adj)

    def neighbors(self, v: int) -> list[int]:
        return [to for to, _ in self.adj[v]]

    def degree(self, v: int) -> int:
        return len(self.adj[v])


def build_adjacency(mol: Molecule) -> Adjacency:
    adj: list[list[tuple[int, int]]] = [[] for _ in mol.atoms]
    for bond in mol.bonds:
        order = bond.order or 1
        adj[bond.a].append((bond.b, order))
        adj[bond.b].append((bond.a, order))
    for lst in adj:
        lst.sort()
    return Adjacency(adj)


def used_valence(mol: Molecule, adj: Adjacency) -> list[int]:
    """每个原子已使用的化合价（单键 1 / 双键 2 / 三键 3）。"""
    return [sum(order for _, order in adj.adj[v]) for v in range(len(mol.atoms))]


def degrees(mol: Molecule, adj: Adjacency) -> list[int]:
    return [len(lst) for lst in adj.adj]


def find_bond_index(mol: Molecule, a: int, b: int) -> int:
    for i, bond in enumerate(mol.bonds):
        if (bond.a == a and bond.b == b) or (bond.a == b and bond.b == a):
            return i
    return -1


def find_bond(mol: Molecule, a: int, b: int) -> Bond | None:
    i = find_bond_index(mol, a, b)
    return None if i < 0 else mol.bonds[i]


# ----------------------------------------------------------------------------
# 连通分量与子图
# ----------------------------------------------------------------------------


def connected_components(mol: Molecule) -> list[list[int]]:
    """连通分量。返回原子下标列表的列表。"""
    n = len(mol.atoms)
    adj = build_adjacency(mol)
    seen = [False] * n
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for to, _ in adj.adj[v]:
                if not seen[to]:
                    seen[to] = True
                    stack.append(to)
        comp.sort()
        comps.append(comp)
    return comps


def subgraph(mol: Molecule, index_list: Sequence[int]) -> Molecule:
    """从完整分子里抽出若干个原子，得到独立的子图（下标重新编号）。"""
    mapping = {orig: new for new, orig in enumerate(index_list)}
    atoms = [mol.atoms[i] for i in index_list]
    bonds = [
        Bond(mapping[b.a], mapping[b.b], b.order or 1)
        for b in mol.bonds
        if b.a in mapping and b.b in mapping
    ]
    return Molecule(atoms=atoms, bonds=bonds)


# ----------------------------------------------------------------------------
# 生成树与找环
# ----------------------------------------------------------------------------


@dataclass
class SpanningTree:
    """骨架树。

    为什么必须用 DFS 而不是 BFS：
      DFS 会让整条环落在同一条搜索路径上，于是环闭合边恰好连接「路径首尾」，
      补位时只需移动一个原子就能闭合环。用 BFS 的话环上的原子会被分到不同分支，
      闭合边跨越多个未定位的原子，环几何直接崩掉（表现为键长算到 5 Å）。
    """

    root: int
    parent: list[int]
    depth: list[int]
    children: list[list[int]]
    ring_edges: list[tuple[int, int, int]]   # (a, b, 键级)
    dfs_order: list[int]


def spanning_tree(mol: Molecule, adj: Adjacency, seed: int | None = None) -> SpanningTree:
    n = len(mol.atoms)
    starts = list(range(n))
    if seed is not None and 0 <= seed < n:
        starts.remove(seed)
        starts.insert(0, seed)

    parent = [-1] * n
    depth = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    visited = [False] * n
    dfs_order: list[int] = []

    for start in starts:
        if visited[start]:
            continue
        visited[start] = True
        depth[start] = 0
        dfs_order.append(start)
        # 显式栈，避免大分子递归过深
        stack: list[tuple[int, list[tuple[int, int]], int]] = [(start, list(adj.adj[start]), 0)]
        while stack:
            v, neighbors, next_i = stack[-1]
            if next_i >= len(neighbors):
                stack.pop()
                continue
            stack[-1] = (v, neighbors, next_i + 1)
            child = neighbors[next_i][0]
            if visited[child]:
                continue
            visited[child] = True
            parent[child] = v
            depth[child] = depth[v] + 1
            children[v].append(child)
            dfs_order.append(child)
            stack.append((child, list(adj.adj[child]), 0))

    # 收集闭合边：不属于树的边
    ring_edges: list[tuple[int, int, int]] = []
    seen_pair: set[tuple[int, int]] = set()
    for a in range(n):
        for b, order in adj.adj[a]:
            lo, hi = (a, b) if a < b else (b, a)
            if (lo, hi) in seen_pair:
                continue
            if parent[b] == a or parent[a] == b:
                continue        # 树边
            seen_pair.add((lo, hi))
            ring_edges.append((a, b, order))

    return SpanningTree(starts[0], parent, depth, children, ring_edges, dfs_order)


def path_to_root(tree: SpanningTree, v: int) -> list[int]:
    path = [v]
    guard = 0
    while tree.parent[v] >= 0 and guard < 4096:
        v = tree.parent[v]
        path.append(v)
        guard += 1
    return path


def find_rings(mol: Molecule, adj: Adjacency, tree: SpanningTree) -> list[list[int]]:
    """从闭合边还原环：闭合边 + 两端到共同祖先的树路径。

    返回每个环的原子下标序列（有序，首尾相邻）。
    """
    rings: list[list[int]] = []
    for a, b, _order in tree.ring_edges:
        path_a = path_to_root(tree, a)
        path_b = path_to_root(tree, b)
        pos_in_a = {v: i for i, v in enumerate(path_a)}
        meet = None
        meet_in_b = -1
        for i, v in enumerate(path_b):
            if v in pos_in_a:
                meet = v
                meet_in_b = i
                break
        if meet is None:
            continue
        ring = path_a[: pos_in_a[meet] + 1] + list(reversed(path_b[:meet_in_b]))
        if len(ring) >= 3:
            rings.append(ring)
    return rings


# ----------------------------------------------------------------------------
# 几何辅助
# ----------------------------------------------------------------------------


def centroid(mol: Molecule, index_list: Sequence[int]) -> Vec3:
    n = len(index_list)
    if n == 0:
        return (0.0, 0.0, 0.0)
    sx = sy = sz = 0.0
    for i in index_list:
        p = mol.atoms[i].pos3
        if p is None:
            return (0.0, 0.0, 0.0)
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / n, sy / n, sz / n)


def ring_normal(mol: Molecule, index_list: Sequence[int]) -> Vec3:
    """环所在平面的法向量（用前三个不共线的原子求）。"""
    from .vector import v3_cross, v3_len, v3_normalize, v3_sub

    for i in range(1, len(index_list) - 1):
        p0 = mol.atoms[index_list[0]].pos3
        p1 = mol.atoms[index_list[i]].pos3
        p2 = mol.atoms[index_list[i + 1]].pos3
        if p0 is None or p1 is None or p2 is None:
            continue
        n = v3_cross(v3_sub(p1, p0), v3_sub(p2, p0))
        if v3_len(n) > 1e-6:
            return v3_normalize(n)
    return (0.0, 0.0, 1.0)
