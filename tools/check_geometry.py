"""3D 构型质量检查：把我关心的化学正确性指标一次打出来。

用法：python tools/check_geometry.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from molecule_workshop.chem.constants import bond_length          # noqa: E402
from molecule_workshop.chem.geometry import build_geometry        # noqa: E402
from molecule_workshop.chem.parse_spec import parse_spec          # noqa: E402
from molecule_workshop.chem.relax import solve                    # noqa: E402
from molecule_workshop.chem.vector import (                       # noqa: E402
    v3_cross, v3_dist, v3_dot, v3_len, v3_normalize, v3_sub,
)


def angle_deg(pos, i, j, k) -> float:
    v1 = v3_normalize(v3_sub(pos[i], pos[j]))
    v2 = v3_normalize(v3_sub(pos[k], pos[j]))
    return math.degrees(math.acos(max(-1.0, min(1.0, v3_dot(v1, v2)))))


def ring_carbons(mol) -> list[int]:
    """环上的碳，按环序返回。

    判据是「真正在环上」，而不是「有两个碳邻居」——
    后者会把丙酮的中心碳之类的非环原子误判进来（检查脚本曾因此崩掉）。
    用生成树 + 环闭合边找出的环才可靠。
    """
    from molecule_workshop.chem.graph import build_adjacency, find_rings, spanning_tree

    adj = build_adjacency(mol)
    if not mol.atoms:
        return []
    root = max(range(len(mol.atoms)), key=lambda v: adj.degree(v))
    tree = spanning_tree(mol, adj, root)
    rings = find_rings(mol, adj, tree)
    if not rings:
        return []
    # 取第一个含碳的环
    for ring in rings:
        carbons = [i for i in ring if mol.atoms[i].el == "C"]
        if len(carbons) >= 3:
            return ring
    return []


def report(name: str, spec: str, ring_expect: int | None = None) -> None:
    mol = parse_spec(spec)
    t0 = time.time()
    res = solve(mol)
    ms = (time.time() - t0) * 1000
    if res is None:
        print(f"{name}: 求解失败")
        return
    pos = res.positions

    bond_err = []
    for b in mol.bonds:
        d0 = bond_length(mol.atoms[b.a].el, mol.atoms[b.b].el, b.order or 1)
        bond_err.append(abs(v3_dist(pos[b.a], pos[b.b]) - d0))
    max_bond_err = max(bond_err) if bond_err else 0.0

    parts = [f"{name:<8} 键长偏差 {max_bond_err:.3f} Å   E={res.energy:.4f}  {ms:.0f} ms"]

    ring = ring_carbons(mol)
    if ring and (ring_expect is None or len(ring) == ring_expect):
        p0, p1, p2 = pos[ring[0]], pos[ring[1]], pos[ring[2]]
        nrm = v3_normalize(v3_cross(v3_sub(p1, p0), v3_sub(p2, p0)))
        devs = [abs(v3_dot(v3_sub(pos[i], p0), nrm)) for i in ring]
        w = len(ring)
        inner = [angle_deg(pos, ring[(k + w - 1) % w], ring[k], ring[(k + 1) % w])
                 for k in range(w)]
        parts.append(
            f"环{w}元 内角 {min(inner):.1f}~{max(inner):.1f}° 离平面 {max(devs):.3f} Å")
    print("  ".join(parts))


def methane_check() -> None:
    mol = parse_spec("C(-H)(-H)(-H)(-H)")
    res = solve(mol)
    pos = res.positions
    hs = [i for i, a in enumerate(mol.atoms) if a.el == "H"]
    c = next(i for i, a in enumerate(mol.atoms) if a.el == "C")
    angs = [angle_deg(pos, hs[i], c, hs[j]) for i in range(4) for j in range(i + 1, 4)]
    dev = max(abs(a - 109.47) for a in angs)
    lens = [v3_dist(pos[h], pos[c]) for h in hs]
    print(f"  甲烷 6 个 H—C—H: {min(angs):.2f}° ~ {max(angs):.2f}°（最大偏离 {dev:.2f}°）"
          f"  C—H 键长 {min(lens):.3f}~{max(lens):.3f} Å")


def acetylene_check() -> None:
    mol = parse_spec("C(-H)#C(-H)")
    res = solve(mol)
    pos = res.positions
    c0, c1 = 0, 2
    h0, h1 = 1, 3
    a1 = angle_deg(pos, h0, c0, c1)
    a2 = angle_deg(pos, c0, c1, h1)
    a3 = angle_deg(pos, h0, c0, h1)
    dev = max(abs(a - 180.0) for a in (a1, a2, a3))
    print(f"  乙炔 H—C≡C {a1:.1f}°  C≡C—H {a2:.1f}°  H…H {a3:.1f}°（最大偏离 {dev:.1f}°）")


def ethylene_check() -> None:
    mol = parse_spec("C(-H)(-H)=C(-H)(-H)")
    res = solve(mol)
    pos = res.positions
    hch1 = angle_deg(pos, 1, 0, 2)
    hch2 = angle_deg(pos, 4, 3, 5)
    nrm = v3_normalize(v3_cross(v3_sub(pos[1], pos[0]), v3_sub(pos[3], pos[0])))
    planarity = max(abs(v3_dot(v3_sub(p, pos[0]), nrm)) for p in pos)
    print(f"  乙烯 H—C—H {hch1:.1f}° / {hch2:.1f}°  离平面最大 {planarity:.4f} Å")


def main() -> int:
    print("键角/共面性专项：")
    methane_check()
    acetylene_check()
    ethylene_check()

    print("\n各分子构型：")
    report("甲烷", "C(-H)(-H)(-H)(-H)")
    report("乙烷", "C(-H)(-H)(-H)-C(-H)(-H)(-H)")
    report("乙烯", "C(-H)(-H)=C(-H)(-H)")
    report("乙炔", "C(-H)#C(-H)")
    report("环丙烷", "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1", ring_expect=3)
    report("环丁烷", "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1", ring_expect=4)
    report("环戊烷", "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1", ring_expect=5)
    report("环己烷", "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1", ring_expect=6)
    report("苯", "C1(-H)=C(-H)-C(-H)=C(-H)-C(-H)=C(-H)-1", ring_expect=6)
    report("乙醇", "C(-H)(-H)(-H)-C(-H)(-H)-O(-H)")
    report("丙酮", "C(-H)(-H)(-H)-C(=O)-C(-H)(-H)(-H)")
    report("乙酸", "C(-H)(-H)(-H)-C(=O)-O(-H)")
    report("甘氨酸", "C(-H)(-H)(-N(-H)(-H))-C(=O)-O(-H)")

    print("\n整库求解耗时：")
    from molecule_workshop.data.molecules import MOLECULES
    t0 = time.time()
    slowest = ("", 0.0)
    for rec in MOLECULES:
        mol = parse_spec(rec["spec"])
        t1 = time.time()
        solve(mol, attempts=2)
        dt = (time.time() - t1) * 1000
        if dt > slowest[1]:
            slowest = (rec["id"], dt)
    total = (time.time() - t0) * 1000
    print(f"  {len(MOLECULES)} 条共 {total:.0f} ms，最慢 {slowest[0]} {slowest[1]:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
