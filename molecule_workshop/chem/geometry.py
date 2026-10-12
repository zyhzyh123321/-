"""3D 构型生成：杂化判定 + 键方向分配。

分两步：
    1. 本文件：按杂化态决定每根键的方向，搭出「理想骨架」（键长键角都对）
    2. relax.py：几何约束投影 + 力场松弛，把残余张力消掉（环己烷的椅式在这里出现）

化学依据：
    有 1 个三键 → sp，直线形，180°
    有 1 个双键 → sp²，平面三角形，120°
    其余       → sp³，正四面体，约 109.47°
杂化态决定键方向，键方向决定立体构型 —— 所以原子的 3D 位置由结构算出来，
不能由玩家随手拖动决定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from .constants import BOND_ANGLE, ELEMENTS, SLOT_DIRS_3D, bond_length
from .graph import (
    Molecule,
    SpanningTree,
    build_adjacency,
    find_bond,
    find_rings,
    spanning_tree,
)
from .vector import (
    Vec3,
    deg2rad,
    rad2deg,
    v3_add,
    v3_angle_between,
    v3_basis_from,
    v3_cross,
    v3_dot,
    v3_len,
    v3_mul,
    v3_normalize,
    v3_project_perp,
    v3_rotate_axis,
    v3_sub,
)

# ----------------------------------------------------------------------------
# 杂化判定
# ----------------------------------------------------------------------------


def hybridization(bonds: list[tuple[int, int]]) -> tuple[str, int]:
    """依据原子的成键情况判断杂化态。

    bonds 是该原子的 [(邻居下标, 键级), ...]
    返回 (杂化态, 能容纳的最大方向数)
    """
    has_triple = any(order >= 3 for _, order in bonds)
    has_double = any(order == 2 for _, order in bonds)
    if has_triple:
        return "sp", 2
    if has_double:
        return "sp2", 3
    # 多元环上的饱和碳仍按 sp3 处理，靠松弛把环收缩成实际构象
    return "sp3", max(4, len(bonds))


def ring_internal_angle_deg(ring_size: int) -> float:
    """k 元环每个内角的几何必然值：正 k 边形的内角 (k-2)·180/k。

    为什么需要这个函数：
        投影阶段把环摆成了正多边形（内角正是这个值），
        但松弛阶段如果仍按 sp³ 的 109.47° 去拉，两套目标会互相打架，
        环会被拉歪成不等边三角形（实测环丙烷环边 1.497/1.567/1.497，
        最大内角 63.13°、能量 4.04）。
        让两个阶段用同一个目标，环就稳定在正多边形附近。
    """
    k = max(3, ring_size)
    return (k - 2) * 180.0 / k


def ideal_angle_deg(center_el: str, bond_orders: list[int]) -> float:
    """理想键角：由中心原子的成键情况决定。"""
    if any(o >= 3 for o in bond_orders):
        return BOND_ANGLE["sp"]
    if any(o == 2 for o in bond_orders):
        return BOND_ANGLE["sp2"]
    return BOND_ANGLE["sp3"]


def template_pool(h: str) -> list[Vec3]:
    """模板方向池（去重后的单位向量）。"""
    seen: set[Vec3] = set()
    out: list[Vec3] = []
    for raw in SLOT_DIRS_3D.get(h, SLOT_DIRS_3D["sp3"]):
        d = v3_normalize(raw)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ----------------------------------------------------------------------------
# 方向分配
# ----------------------------------------------------------------------------


def deviation_score(dirs: list[Vec3], ideal_deg: float, placed: list[Vec3]) -> float:
    """给一组方向打分（越小越好）：夹角与理想值的偏差平方均值。

    另外加了共面偏好：对 sp² 中心，如果这组方向张成三维，分数会额外变大。
    原因：模板里「三个两两 120° 的方向」有两种可能 —— 一种共面（真正想要的），
    另一种是绕轴转开的立体排布。只按夹角偏差打分时两者都得 0 分，
    会随机选中立体那种，乙烯因此被摆成不共面（实测离平面 0.94 Å）。
    """
    target = deg2rad(ideal_deg)
    score = 0.0
    count = 0
    for a, b in combinations(dirs, 2):
        diff = v3_angle_between(a, b) - target
        score += diff * diff
        count += 1
    for d in dirs:
        for p in placed:
            diff = v3_angle_between(d, p) - target
            score += diff * diff
            count += 1
    score = score / count if count else 0.0

    # 共面偏好（仅对 3 个方向的 sp² 类型中心有意义）
    if len(dirs) >= 3 and abs(ideal_deg - 120.0) < 1.0:
        for a, b, c in combinations(dirs, 3):
            n = v3_cross(a, b)
            if v3_len(n) < 1e-9:
                continue
            vol = abs(v3_dot(v3_normalize(n), c))
            score += 0.05 * vol * vol
    return score


def assign_from_template(h: str, k: int, total_slots: int | None = None) -> list[Vec3]:
    """没有入边时（分子起点、或多个取代基的中心）：从模板里挑最优组合。"""
    pool = template_pool(h)
    ideal = BOND_ANGLE.get(h, BOND_ANGLE["sp3"])
    need = min(len(pool), max(1, min(k, len(pool))))

    best_dirs: list[Vec3] | None = None
    best_score = math.inf
    nodes = 0
    chosen: list[Vec3] = [None] * need  # type: ignore[list-item]

    def search(pos: int, start: int) -> None:
        nonlocal best_dirs, best_score, nodes
        nodes += 1
        if nodes > 40000:
            return
        if pos == need:
            s = deviation_score(chosen, ideal, [])
            if s < best_score:
                best_score = s
                best_dirs = list(chosen)
            return
        for i in range(start, len(pool)):
            chosen[pos] = pool[i]
            search(pos + 1, i + 1)

    search(0, 0)
    if best_dirs:
        return best_dirs[:k]

    # 模板不够：在球面上按黄金角均匀铺开
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out: list[Vec3] = []
    for j in range(k):
        y = 1.0 - (2.0 * j + 1.0) / max(1, k)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        az = golden * j
        out.append((math.cos(az) * r, y, math.sin(az) * r))
    return out


def assign_directions(
    h: str,
    placed: list[Vec3],
    k: int,
    mode: str | None = None,
    total_slots: int | None = None,
) -> list[Vec3]:
    """为一个中心原子分配 k 个新的出边方向。

    关键点：
      * 有入边时，所有出边都在以入边为轴、张角=理想键角的圆锥面上。
        方位角间隔取 2π/(n+1)（n = 除入边外还剩几根键）：
          - sp³ 圆锥张角 109.47°，两根键相差 120° 时夹角正好 ≈109.47°
          - sp² 圆锥张角 120°，两根键相差 120° 时夹角正好 120°
        早先 sp² 错用了 2π/(n+1)=60°，两根键被摆到同一侧，乙烯被折成立体。
      * 没有入边时退回模板搜索。
    """
    if k <= 0:
        return []
    ideal = BOND_ANGLE.get(h, BOND_ANGLE["sp3"])
    ideal_rad = deg2rad(ideal)
    m = mode or ("cone" if placed else "free")

    if m == "free" or not placed:
        return assign_from_template(h, k, total_slots)

    in_dir = placed[-1]

    # 直线形：只能取反方向
    if h == "sp":
        anti = v3_mul(in_dir, -1.0)
        return [anti] * k

    slots = total_slots or (3 if h == "sp2" else 4)

    if h == "sp2":
        # 三根键共面且两两 120°：绕入边圆锥的方位角间隔就是 120°
        e1, e2 = v3_basis_from(in_dir)
        n_side = max(1, min(k, slots - 1))
        out: list[Vec3] = []
        for j in range(k):
            phi = 2.0 * math.pi * j / n_side
            out.append(v3_normalize(v3_add(
                v3_mul(in_dir, math.cos(ideal_rad)),
                v3_add(v3_mul(e1, math.sin(ideal_rad) * math.cos(phi)),
                       v3_mul(e2, math.sin(ideal_rad) * math.sin(phi))),
            )))
        return out

    # 四面体：入边占一根，其余按 2π/(n+1) 在圆锥面上展开
    e1, e2 = v3_basis_from(in_dir)
    n_besides = max(1, slots - 1)
    out = []
    for j in range(k):
        az = 2.0 * math.pi * j / (n_besides + 1)
        out.append(v3_normalize(v3_add(
            v3_mul(in_dir, math.cos(ideal_rad)),
            v3_add(v3_mul(e1, math.sin(ideal_rad) * math.cos(az)),
                   v3_mul(e2, math.sin(ideal_rad) * math.sin(az))),
        )))
    return out


def next_best_direction(h: str, used_dirs: list[Vec3], total_slots: int | None = None,
                        remaining: int = 1) -> Vec3:
    """求下一个方向的兜底方法：以已占方向的质心为轴，搜索最优方位角。"""
    ideal_deg = BOND_ANGLE.get(h, BOND_ANGLE["sp3"])
    ideal_rad = deg2rad(ideal_deg)
    slots = total_slots or (2 if h == "sp" else 3 if h == "sp2" else 4)

    if not used_dirs:
        return assign_from_template(h, 1, slots)[0]
    if h == "sp":
        return v3_mul(used_dirs[0], -1.0)

    # 优先用圆锥展开（对 sp²/sp³ 都能给出精确理想夹角）
    cone = assign_directions(h, [used_dirs[0]], 1, "cone", slots)
    if cone:
        cand = cone[0]
        ok = all(
            abs(rad2deg(v3_angle_between(cand, u)) - ideal_deg) <= 25.0
            for u in used_dirs
        )
        if ok:
            return cand

    sx = sy = sz = 0.0
    for u in used_dirs:
        sx += u[0]
        sy += u[1]
        sz += u[2]
    total = (sx, sy, sz)
    axis = v3_normalize(total) if v3_len(total) > 1e-6 else (0.0, 1.0, 0.0)
    e1, e2 = v3_basis_from(axis)

    best_dir = axis
    best_score = math.inf
    steps = 72
    for p in range(steps):
        phi = 2.0 * math.pi * p / steps
        cand = v3_normalize(v3_add(
            v3_mul(axis, math.cos(ideal_rad)),
            v3_add(v3_mul(e1, math.sin(ideal_rad) * math.cos(phi)),
                   v3_mul(e2, math.sin(ideal_rad) * math.sin(phi))),
        ))
        score = sum((v3_angle_between(cand, u) - ideal_rad) ** 2 for u in used_dirs)
        if score < best_score:
            best_score = score
            best_dir = cand
    return best_dir


# ----------------------------------------------------------------------------
# 骨架生成
# ----------------------------------------------------------------------------


@dataclass
class GeometryResult:
    positions: list[Vec3]
    tree: SpanningTree
    hybridization: list[str]
    used_valence: list[int]
    root: int
    rings: list[list[int]] = field(default_factory=list)
    on_ring: list[bool] = field(default_factory=list)


def layout_ring(mol: Molecule, ring: list[int], positions: list[Vec3 | None],
                on_ring: list[bool], ring_atoms: list[int]) -> None:
    """把一个环摆成多边形。

    为什么必须专门处理环：
      环的最后一个原子由「闭合边」确定，而不是由父节点的键方向确定。
      若按普通树的方式摆放，最后那个原子会落在圆锥方位 0 上，与闭合要求冲突。

    **关键：每条边用它自己的真实目标键长**，不能用平均键长。
      甲苯这类环上单双键交替的分子（1.54/1.34），用平均值 1.44 会让
      所有环边都差 ±0.1 Å，闭环误差累积到 0.6 Å 以上，
      后面的约束投影怎么迭代都收不回来，分子直接报废。
      正确做法是按每条边的目标键长逐边推进角度，再把各顶点放到切线上。
    """
    k = len(ring)
    if k < 3:
        return

    for idx in ring:
        on_ring[idx] = True
        if idx not in ring_atoms:
            ring_atoms.append(idx)

    # 每条环边的目标键长
    edge_len: list[float] = []
    for i in range(k):
        a1, a2 = ring[i], ring[(i + 1) % k]
        bond = find_bond(mol, a1, a2)
        edge_len.append(bond_length(mol.atoms[a1].el, mol.atoms[a2].el,
                                    bond.order if bond else 1))
    perimeter = sum(edge_len)
    if perimeter <= 1e-9:
        return

    # 顶点角度按「相邻两条边长」分配，再归一化到 2π 总和。
    # 等边情形下每个角都等于 2π/k（正多边形）；单双键交替的六元环则长短边交替。
    raw = [(edge_len[(i - 1) % k] + edge_len[i]) for i in range(k)]
    total_raw = sum(raw)
    if total_raw <= 1e-9:
        angles = [2.0 * math.pi / k] * k
    else:
        angles = [a * (2.0 * math.pi / total_raw) for a in raw]

    known = [positions[i] for i in ring if positions[i] is not None]

    center: Vec3 = (0.0, 0.0, 0.0)
    normal: Vec3 = (0.0, 0.0, 1.0)
    if len(known) >= 2:
        cx = sum(p[0] for p in known) / len(known)  # type: ignore[index]
        cy = sum(p[1] for p in known) / len(known)  # type: ignore[index]
        cz = sum(p[2] for p in known) / len(known)  # type: ignore[index]
        center = (cx, cy, cz)
        v1 = v3_sub(known[1], known[0])  # type: ignore[arg-type]
        normal = v3_project_perp((0.0, 0.0, 1.0), v1)

    e1 = v3_project_perp((1.0, 0.0, 0.0), normal)
    e2 = v3_cross(normal, e1)

    placed: dict[int, Vec3] = {i: positions[i] for i in ring if positions[i] is not None}  # type: ignore[misc]

    if not placed:
        # 整个环都还没摆放：从原点出发，第一条边沿 e1，逐边按目标键长走
        cur = center
        direction = e1
        for j in range(k):
            positions[ring[j]] = cur
            # 在 ring[j] 处转过 (π - 该顶点内角)，再前进下一条边
            turn = math.pi - angles[(j + 1) % k]
            direction = v3_rotate_axis(direction, normal, turn)
            cur = v3_add(cur, v3_mul(direction, edge_len[j]))
        return

    # 有已知顶点：以已知顶点为起点，按环序双向铺开
    start_j = next(j for j in range(k) if ring[j] in placed)
    cur = placed[ring[start_j]]
    positions[ring[start_j]] = cur
    # 正向：从 start_j 出发沿 e1 方向，按顶点角与边长推进
    direction = e1
    ref = placed.get(ring[(start_j - 1) % k])
    if ref is not None:
        d = v3_sub(cur, ref)
        if v3_len(d) > 1e-9:
            direction = v3_normalize(d)
    for step in range(1, k):
        j = (start_j + step) % k
        # 顶点 ring[j-1] 处需要转过的角
        turn = math.pi - angles[(j - 1) % k]
        direction = v3_rotate_axis(direction, normal, turn)
        cur = v3_add(cur, v3_mul(direction, edge_len[(j - 1) % k]))
        positions[ring[j]] = cur


def build_geometry(mol: Molecule) -> GeometryResult:
    """生成一个连通分量的 3D 初始构型。"""
    n = len(mol.atoms)
    adj = build_adjacency(mol)

    hyb: list[str] = []
    used: list[int] = []
    for v in range(n):
        h, _cap = hybridization(list(adj.adj[v]))
        hyb.append(h)
        used.append(sum(order for _, order in adj.adj[v]))

    # 选根：度数最大的原子（通常就是骨架中心）
    root = 0
    for r in range(1, n):
        if adj.degree(r) > adj.degree(root):
            root = r

    tree = spanning_tree(mol, adj, root)
    rings = find_rings(mol, adj, tree)
    on_ring = [False] * n

    positions: list[Vec3 | None] = [None] * n
    in_dir: list[Vec3 | None] = [None] * n
    used_dirs: list[list[Vec3]] = [[] for _ in range(n)]

    # ---- 第一步：环先摆成规则多边形 ----
    ring_atoms: list[int] = []
    for ring in rings:
        layout_ring(mol, ring, positions, on_ring, ring_atoms)

    # ---- 一点五步：环上原子的取代基单独摆（关键） ----
    #
    # 为什么必须单独处理：
    #   通用的 BFS 摆放只保证「与入边成理想键角」，不区分朝里朝外。
    #   环碳的两个环键方向是往两侧去的，锥面有两个对称解 ——
    #   一个朝环外、一个朝环内。BFS 会挑到朝内的那个，
    #   于是氢原子全被塞进环里、和对面碳原子重叠
    #   （实测甲苯的 H1 落在 (0.52, 1.34)，正好压在碳 0 上），
    #   分子式算对了但构型完全没法看，后面的松弛也救不回来。
    #
    # 做法：环碳的环内两条键的角平分线「朝外」，反方向朝内；
    #   把「朝内方向」当作 cone 的参考轴，取代基自然被摆到环外侧
    #   （与参考轴成 109.47° 时，与两条环键也正好都是 109.47°）。
    placed_ring_substituents: list[int] = []
    for ring in rings:
        k = len(ring)
        if k < 3:
            continue
        cx = sum(positions[i][0] for i in ring) / k      # type: ignore[index]
        cy = sum(positions[i][1] for i in ring) / k      # type: ignore[index]
        cz = sum(positions[i][2] for i in ring) / k      # type: ignore[index]
        centroid: Vec3 = (cx, cy, cz)
        nrm = v3_cross(v3_sub(positions[ring[1]], positions[ring[0]]),   # type: ignore[arg-type]
                       v3_sub(positions[ring[2]], positions[ring[0]]))   # type: ignore[arg-type]
        nrm = v3_normalize(nrm) if v3_len(nrm) > 1e-9 else (0.0, 0.0, 1.0)

        for pos_i, idx in enumerate(ring):
            subs = [w for w in adj.neighbors(idx) if w not in ring]
            if not subs or positions[idx] is None:
                continue
            a1 = ring[(pos_i - 1) % k]
            a2 = ring[(pos_i + 1) % k]
            radial = v3_sub(positions[idx], centroid)        # type: ignore[arg-type]
            outward = (v3_normalize(radial) if v3_len(radial) > 1e-9
                       else v3_normalize(v3_add(
                           v3_normalize(v3_sub(positions[a1], positions[idx])),   # type: ignore[arg-type]
                           v3_normalize(v3_sub(positions[a2], positions[idx])))))  # type: ignore[arg-type]

            # 需要几个方向：所有取代基一起算，保证彼此之间也有正确夹角
            subs.sort(key=lambda w: (mol.atoms[w].el == "H", w))
            slots = ELEMENTS.get(mol.atoms[idx].el, {}).get("valence", 4)

            if hyb[idx] == "sp2":
                # sp² 环碳（如苯环）：取代基必须**与环共面**，
                # 与两条环键各成 120°。它们就在环平面内朝外。
                # 早先对 sp² 也用了「绕朝内轴张锥面」的画法，
                # 结果苯环上的氢被摆到环平面外 60°，
                # 排斥能飙到 92、H—C—H 变成 109.47°（应为 120°）。
                dirs = [outward]
                if len(subs) > 1:
                    tang = v3_cross(nrm, outward)
                    dirs = [outward]
                    for j in range(1, len(subs)):
                        ang = 2.0 * math.pi * j / max(2, len(subs))
                        dirs.append(v3_normalize(v3_add(
                            v3_mul(outward, math.cos(ang)),
                            v3_mul(tang, math.sin(ang)),
                        )))
            elif hyb[idx] == "sp":
                dirs = [outward if j % 2 == 0 else v3_mul(outward, -1.0)
                        for j in range(len(subs))]
            else:
                # sp³ 环碳：取代基朝环外并离开环平面。
                # 用「朝内方向」作参考轴的锥面解，再取环外那一支；
                # 与两条环键的夹角都正好是 109.47°。
                dirs = assign_directions(hyb[idx], [v3_mul(outward, -1.0)], len(subs),
                                         "cone", slots)

            for j, w in enumerate(subs):
                if w in placed_ring_substituents:
                    continue
                d = dirs[j] if j < len(dirs) else outward
                if hyb[idx] != "sp2" and v3_dot(d, nrm) == 0.0 and v3_dot(d, outward) < 0:
                    d = v3_mul(d, -1.0)
                elif hyb[idx] != "sp2" and v3_dot(d, outward) < -1e-6:
                    d = v3_mul(d, -1.0)
                bond = find_bond(mol, idx, w)
                length = bond_length(mol.atoms[idx].el, mol.atoms[w].el,
                                     bond.order if bond else 1)
                positions[w] = v3_add(positions[idx], v3_mul(d, length))   # type: ignore[arg-type]
                in_dir[w] = d
                used_dirs[idx].append(d)
                placed_ring_substituents.append(w)

    # ---- 第二步：以环为锚点，用 BFS 摆放其余原子 ----
    queue: list[int] = []
    seeded: set[int] = set()
    for a in ring_atoms:
        if a not in seeded:
            seeded.add(a)
            queue.append(a)
    for a in placed_ring_substituents:
        if a not in seeded:
            seeded.add(a)
            queue.append(a)

    if not queue:
        positions[root] = (0.0, 0.0, 0.0)
        queue.append(root)

    guard = 0
    while queue and guard < n * 6 + 64:
        guard += 1
        v = queue.pop(0)

        # 尚未摆放的子节点
        pending = [w for w in tree.children[v] if positions[w] is None]

        # 已经摆放好的邻居都占一个方向槽位 —— 用真实位置算，而不是只看树边。
        # 环闭合边不在生成树上，光看树边会漏掉它，导致同一方向被重复占用。
        seen_dirs = list(used_dirs[v])
        for nb in adj.neighbors(v):
            if positions[nb] is None or nb in pending:
                continue
            d = v3_sub(positions[nb], positions[v])   # type: ignore[arg-type]
            if v3_len(d) > 1e-6:
                u = v3_normalize(d)
                if not any(abs(v3_dot(u, e) - 1.0) < 1e-6 for e in seen_dirs):
                    seen_dirs.append(u)
        used_dirs[v] = seen_dirs

        if not pending:
            continue

        # 重原子优先、氢最后
        pending.sort(key=lambda w: (mol.atoms[w].el == "H", w))

        slots = ELEMENTS.get(mol.atoms[v].el, {}).get("valence", 4)

        # 一次性算出该中心所有待放子节点的方向。
        # 不能逐个调用 next_best_direction：那样同一中心上的子方向之间没有相互约束，
        # 会出现「每根键与入边都成理想角、但子方向彼此挤在一起或拧成立体」的坏构型
        # （实测乙烯的第三个方向被摆到锥面外，分子直接不共面）。
        dirs_all = assign_directions(
            hyb[v], used_dirs[v], len(pending),
            "cone" if used_dirs[v] else "free", slots,
        )

        for i, w in enumerate(pending):
            d = dirs_all[i] if i < len(dirs_all) else next_best_direction(
                hyb[v], used_dirs[v], slots, len(pending) - i)
            bond = find_bond(mol, v, w)
            length = bond_length(mol.atoms[v].el, mol.atoms[w].el, bond.order if bond else 1)
            positions[w] = v3_add(positions[v], v3_mul(d, length))  # type: ignore[arg-type]
            in_dir[w] = d
            used_dirs[v].append(d)
            queue.append(w)

    # 兜底：不连通或有遗漏时给个位置
    for i in range(n):
        if positions[i] is None:
            positions[i] = (i * 0.6, 0.0, 0.0)

    return GeometryResult(
        positions=positions,  # type: ignore[arg-type]
        tree=tree,
        hybridization=hyb,
        used_valence=used,
        root=root,
        rings=rings,
        on_ring=on_ring,
    )


def normalize_positions(positions: list[Vec3]) -> tuple[list[Vec3], float]:
    """把坐标平移到质心原点，返回 (新坐标, 最大半径)。"""
    n = len(positions)
    if n == 0:
        return positions, 1.0
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n
    out = [(p[0] - cx, p[1] - cy, p[2] - cz) for p in positions]
    max_r = max((v3_len(p) for p in out), default=1.0)
    return out, (max_r if max_r > 1e-6 else 1.0)
