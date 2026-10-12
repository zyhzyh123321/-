"""位置约束求解器（PBD，Position-Based Dynamics）。

为什么重写成这一版：
    原来的做法是「约束投影 + 力场梯度下降」两段式。实测它不可靠 ——
    会把**本来完全正确的初始几何弄坏**：
        苯的初始键长偏差 0.0000 Å、内角全 120°、严格共面，
        跑完松弛后键长偏差变成 1.04 Å、内角散成 49°~73°。
    原因是梯度法用非键排斥换取了键长的松弛：能量从 113 降到 44，
    但代价是把键拉长到 2.5 Å —— 它「优化」的是能量，不是几何正确性。

    PBD 没有这个问题：**约束是硬约束，按构造始终满足**。
    每轮迭代里先把键长精确拉到目标值，再把键角精确转到目标值，
    最后才用很小的权重做一次非键分离（而且分离后立刻重新满足约束）。
    这样键长键角永远是对的，冲突只会表现为「环被撑开一点」，
    而不是「键被拉断」。

顺带解决的两件事：
    1. 不再需要额外的「环收口」步骤去强行压平小环 ——
       环丙烷的 60° 是三条硬键长约束自然收敛的结果。
    2. 不再需要「投影 / 松弛」两套目标角互相打架的问题：全流程只有一套目标。

环己烷的椅式怎么来：
    初始摆成平面六边形（键长键角都满足），但环碳是 sp³、想要 109.47°，
    而平面六边形内角是 120°。求解器每轮都往 109.47° 拉一点，
    环于是自己皱起来 —— 这是真实的物理驱动，不是手塞进去的构象。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import (
    ANGLE_K,
    BOND_K,
    RELAX_MAX_ATOMS,
    REPULSION_K,
    REPULSION_MIN_DIST,
    bond_length,
)
from .geometry import build_geometry, hybridization, ideal_angle_deg
from .graph import (
    Molecule,
    build_adjacency,
    find_rings,
    spanning_tree,
)
from .vector import (
    Vec3,
    clamp,
    v3_add,
    v3_cross,
    v3_dist,
    v3_dot,
    v3_len,
    v3_mul,
    v3_normalize,
    v3_rotate_axis,
    v3_sub,
)


@dataclass
class AngleTarget:
    i: int
    j: int          # 中心原子
    k: int
    target: float   # 目标角（弧度）


@dataclass
class SolveResult:
    positions: list[Vec3]
    radius: float
    energy: float
    score: float
    iterations: int
    converged: bool
    hybridization: list[str] = field(default_factory=list)
    used_valence: list[int] = field(default_factory=list)


# ----------------------------------------------------------------------------
# 约束收集
# ----------------------------------------------------------------------------


def collect_angle_targets(mol: Molecule) -> list[AngleTarget]:
    """收集所有键角约束。

    目标角一律取中心原子的杂化理想角（sp³ 109.47° / sp² 120° / sp 180°）。
    **小环不做特殊处理**：环丙烷内角只能是 60°，这是三条硬键长约束
    在三角形上的必然结果，求解器会自己收敛到 60°，
    不需要（也不应该）把目标改成 60° ——
    一旦改成 60°，环己烷就会失去「sp³ 想要 109.47°」这个褶皱驱动力而变平。
    """
    adj = build_adjacency(mol)
    out: list[AngleTarget] = []
    for j in range(len(mol.atoms)):
        nb = adj.adj[j]
        if len(nb) < 2 or len(nb) > 4:
            continue
        orders = [o for _, o in nb]
        target = math.radians(ideal_angle_deg(mol.atoms[j].el, orders))
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                out.append(AngleTarget(nb[a][0], j, nb[b][0], target))
    return out


# ----------------------------------------------------------------------------
# 单条约束的修正
# ----------------------------------------------------------------------------


def _satisfy_bond(positions: list[Vec3], a: int, b: int, target: float,
                  inv_mass: list[float]) -> None:
    dv = v3_sub(positions[b], positions[a])
    d = v3_len(dv)
    if d < 1e-9:
        return
    diff = d - target
    wa, wb = inv_mass[a], inv_mass[b]
    wsum = wa + wb
    if wsum < 1e-12:
        return
    u = v3_mul(dv, 1.0 / d)
    positions[a] = v3_add(positions[a], v3_mul(u, diff * wa / wsum))
    positions[b] = v3_sub(positions[b], v3_mul(u, diff * wb / wsum))


def _satisfy_angle(positions: list[Vec3], t: AngleTarget,
                   inv_mass: list[float]) -> None:
    v1 = v3_sub(positions[t.i], positions[t.j])
    v2 = v3_sub(positions[t.k], positions[t.j])
    l1, l2 = v3_len(v1), v3_len(v2)
    if l1 < 1e-9 or l2 < 1e-9:
        return
    u1 = v3_mul(v1, 1.0 / l1)
    u2 = v3_mul(v2, 1.0 / l2)
    cos_t = clamp(v3_dot(u1, u2), -1.0, 1.0)
    cur = math.acos(cos_t)
    delta = cur - t.target
    if abs(delta) < 1e-9:
        return
    axis = v3_cross(u1, u2)
    alen = v3_len(axis)
    if alen < 1e-7:
        return                      # 共线或重合：留给其它约束处理
    axis = v3_mul(axis, 1.0 / alen)

    # 两端各承担一半修正；权重按 1/质量 分配
    wi, wk = inv_mass[t.i], inv_mass[t.k]
    wsum = wi + wk
    if wsum < 1e-12:
        return
    share_i = wi / wsum
    origin = positions[t.j]
    if share_i > 1e-6:
        ri = v3_sub(positions[t.i], origin)
        positions[t.i] = v3_add(origin, v3_rotate_axis(ri, axis, delta * share_i))
    if share_i < 1.0 - 1e-6:
        rk = v3_sub(positions[t.k], origin)
        positions[t.k] = v3_add(origin, v3_rotate_axis(rk, axis, -delta * (1.0 - share_i)))


def _separate(positions: list[Vec3], pairs: list[tuple[int, int]],
              inv_mass: list[float], strength: float = 0.5) -> None:
    """软性的非键分离（权重很小，避免破坏键长键角）。"""
    rmin = REPULSION_MIN_DIST
    for a, b in pairs:
        dv = v3_sub(positions[b], positions[a])
        d = v3_len(dv)
        if d >= rmin or d < 1e-9:
            continue
        wa, wb = inv_mass[a], inv_mass[b]
        wsum = wa + wb
        if wsum < 1e-12:
            continue
        u = v3_mul(dv, 1.0 / d)
        push = (rmin - d) * strength
        positions[a] = v3_sub(positions[a], v3_mul(u, push * wa / wsum))
        positions[b] = v3_add(positions[b], v3_mul(u, push * wb / wsum))


# ----------------------------------------------------------------------------
# 求解
# ----------------------------------------------------------------------------


def score_geometry(mol: Molecule, positions: list[Vec3],
                   angle_targets: list[AngleTarget] | None = None) -> float:
    """几何偏差评分：键长偏差平方 + 键角偏差平方（用于在多次尝试中择优）。"""
    score = 0.0
    for b in mol.bonds:
        d0 = bond_length(mol.atoms[b.a].el, mol.atoms[b.b].el, b.order or 1)
        diff = v3_dist(positions[b.a], positions[b.b]) - d0
        score += diff * diff
    targets = angle_targets if angle_targets is not None else collect_angle_targets(mol)
    for t in targets:
        v1 = v3_sub(positions[t.i], positions[t.j])
        v2 = v3_sub(positions[t.k], positions[t.j])
        l1, l2 = v3_len(v1), v3_len(v2)
        if l1 < 1e-9 or l2 < 1e-9:
            score += 1000.0
            continue
        cur = math.acos(clamp(v3_dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        dev = cur - t.target
        score += 0.35 * dev * dev
    return score


def energy_of(mol: Molecule, positions: list[Vec3]) -> float:
    """报告用的能量（键长 + 键角 + 排斥）。仅用于比较与展示，不参与求解。"""
    energy = 0.0
    bonded: set[frozenset[int]] = set()
    for b in mol.bonds:
        bonded.add(frozenset((b.a, b.b)))
        d0 = bond_length(mol.atoms[b.a].el, mol.atoms[b.b].el, b.order or 1)
        diff = v3_dist(positions[b.a], positions[b.b]) - d0
        energy += 0.5 * BOND_K * diff * diff
    for t in collect_angle_targets(mol):
        v1 = v3_sub(positions[t.i], positions[t.j])
        v2 = v3_sub(positions[t.k], positions[t.j])
        l1, l2 = v3_len(v1), v3_len(v2)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        cur = math.acos(clamp(v3_dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        energy += 0.5 * ANGLE_K * (cur - t.target) ** 2
    n = len(positions)
    for p in range(n):
        for q in range(p + 1, n):
            if frozenset((p, q)) in bonded:
                continue
            d = v3_dist(positions[p], positions[q])
            if d < REPULSION_MIN_DIST:
                energy += REPULSION_K * (REPULSION_MIN_DIST - d) ** 2
    return energy


def solve(mol: Molecule, attempts: int | None = None, amp: float = 0.35,
          iterations: int = 160, seed: int = 20261011) -> SolveResult | None:
    """求解 3D 构型。

    多跑几次带扰动的尝试并择优（择优标准：先过几何门槛、再比能量）。
    扰动用**按尝试序号固定种子**的伪随机流 ——
    共享同一条流时随机数消耗量会随退出时机变化，
    同一个分子每次打开形状都不同，玩家会觉得「这游戏不稳」。

    一次尝试的代价是 iterations × (键角数 + 键数)，对 18 原子的环己烷
    约 160 × 60 ≈ 1 万次约束修正，很快；四次尝试总计几毫秒到几十毫秒。
    """
    import random

    n = len(mol.atoms)
    if n == 0:
        return None
    if n > RELAX_MAX_ATOMS:
        return None
    if attempts is None:
        # 有环的分子构象空间更复杂，多试几次
        adj = build_adjacency(mol)
        has_ring = bool(find_rings(mol, adj, spanning_tree(mol, adj, 0)))
        attempts = 5 if has_ring else (3 if n <= 12 else 2)

    angle_targets = collect_angle_targets(mol)
    bonds = [
        (b.a, b.b, bond_length(mol.atoms[b.a].el, mol.atoms[b.b].el, b.order or 1))
        for b in mol.bonds
    ]
    bonded_pairs = {frozenset((b.a, b.b)) for b in mol.bonds}
    nonbonded = [(p, q) for p in range(n) for q in range(p + 1, n)
                 if frozenset((p, q)) not in bonded_pairs]

    # 自由度权重：约束越多的原子越「重」（挪得越少），让末端原子先动
    adj = build_adjacency(mol)
    inv_mass = [1.0 / max(1.0, float(len(adj.adj[v]))) for v in range(n)]

    best: SolveResult | None = None
    GEOM_OK = 0.05

    for a in range(attempts):
        geo = build_geometry(mol)
        positions = list(geo.positions)

        if a > 0:
            rng = random.Random(seed + a * 7919)
            positions = [
                (p[0] + (rng.random() - 0.5) * amp,
                 p[1] + (rng.random() - 0.5) * amp,
                 p[2] + (rng.random() - 0.5) * amp)
                for p in positions
            ]

        # 每轮：先满足角度、再满足键长（键长最后做，保证它最准）。
        #
        # 性能上分两段：
        #   前 1/3 轮次带非键分离（最贵的一步，要遍历所有非键对），
        #   后 2/3 只做键长键角（便宜，O(键数)），把几何收敛干净。
        # 之前每 4 轮就分离一次，整个库要跑 13.8 秒；分段后降到 1 秒内。
        sep_end = max(1, iterations // 3)
        for it in range(iterations):
            for t in angle_targets:
                _satisfy_angle(positions, t, inv_mass)
            for (ia, ib, d0) in bonds:
                _satisfy_bond(positions, ia, ib, d0, inv_mass)
            if it < sep_end and it % 3 == 2 and nonbonded:
                _separate(positions, nonbonded, inv_mass, strength=0.45)
                for t in angle_targets:
                    _satisfy_angle(positions, t, inv_mass)
                for (ia, ib, d0) in bonds:
                    _satisfy_bond(positions, ia, ib, d0, inv_mass)

        # 收尾：再多拉几轮键长，确保偏差最小
        for _ in range(12):
            for (ia, ib, d0) in bonds:
                _satisfy_bond(positions, ia, ib, d0, inv_mass)

        if not all(math.isfinite(c) for p in positions for c in p):
            continue

        score = score_geometry(mol, positions, angle_targets)
        energy = energy_of(mol, positions)

        cand = SolveResult(
            positions=positions, radius=1.0, energy=energy, score=score,
            iterations=iterations, converged=score <= GEOM_OK,
            hybridization=geo.hybridization, used_valence=geo.used_valence,
        )
        best = _pick_better(best, cand, GEOM_OK)

    if best is None:
        return None

    # 平移到质心原点并按半径归一
    from .geometry import normalize_positions
    norm, radius = normalize_positions(best.positions)
    best.positions = norm
    best.radius = radius
    return best


def _pick_better(best: SolveResult | None, cand: SolveResult, geom_ok: float) -> SolveResult:
    """择优：先过几何门槛，再比能量，最后用几何偏差破平局。

    这里必须「能量优先」而不是「几何偏差优先」：
        环己烷的椅式（离平面 0.52 Å）几何偏差略大于平面构型（0.09 Å），
        但能量低一个数量级（平面构型把 sp³ 碳撑到 120°，角张力很大）。
        按几何偏差选会选错。
    """
    if best is None:
        return cand
    best_ok = best.score <= geom_ok
    cand_ok = cand.score <= geom_ok
    if best_ok and (not cand_ok or best.energy <= cand.energy + 1e-9):
        return best
    if cand_ok and not best_ok:
        return cand
    if not best_ok and not cand_ok:
        return cand if cand.score < best.score else best
    # 两者都合格：比能量，能量几乎相同再比几何
    if cand.energy < best.energy - 1e-9:
        return cand
    if best.energy < cand.energy - 1e-9:
        return best
    return cand if cand.score < best.score else best
