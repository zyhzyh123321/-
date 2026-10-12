"""环几何的三个坑与最终结论（这份文件是「别再改错」的记录）。

本脚本会实测打印各环分子的构型指标，用来验证任何后续改动有没有把结论推翻。

用法：
    python tools/ring_angle_study.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from molecule_workshop.chem.graph import build_adjacency, find_rings, spanning_tree  # noqa: E402
from molecule_workshop.chem.parse_spec import parse_spec                              # noqa: E402
from molecule_workshop.chem.relax import energy_of, solve                             # noqa: E402
from molecule_workshop.chem.vector import (                                           # noqa: E402
    v3_cross, v3_dist, v3_dot, v3_normalize, v3_sub,
)

SPECS = {
    "环丙烷": "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1",
    "环丁烷": "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1",
    "环戊烷": "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1",
    "环己烷": "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1",
    "苯": "C1(-H)=C(-H)-C(-H)=C(-H)-C(-H)=C(-H)-1",
    "甲苯": "C1(-H)=C(-H)-C(-H)=C(-H)-C(-H)=C(-C(-H)(-H)(-H))-1",
    "环丙烷(由游戏路径)": "C1(-H)(-H)-C(-H)(-H)-C(-H)(-H)-1",
}

# ----------------------------------------------------------------------------
# 结论一：环内角目标必须保持杂化理想角，不能改成多边形内角
# ----------------------------------------------------------------------------
CONCLUSION_1 = """
结论 1（最反直觉的一个）：环内角目标必须用杂化理想角，不能改成多边形内角
    环碳是 sp³、它「想要」的是 109.47°，而正六边形内角是 120°。
    **正是这个矛盾驱动了环己烷的褶皱。**
    如果把环内角目标改成 120°，等于告诉求解器「平面才是对的」，
    椅式立刻消失 —— 实测褶皱从 0.508 Å 掉到 0.092 Å（第一代求解器）
    / 从 1.677 Å 掉到接近 0（当前的 PBD 求解器）。
    所以 collect_angles 一律用杂化理想角，环也不做特殊处理。

    小环为什么会自动变成正多边形？
      环丙烷的内角只能是 60°，这不是「目标值」，而是三条硬键长约束
      在三角形上的必然结果 —— 求解器自然会收敛到 60.00°，
      不需要把目标改成 60°。
"""

# ----------------------------------------------------------------------------
# 结论二：环的初始摆放必须逐边用真实键长，且取代基必须朝环外
# ----------------------------------------------------------------------------
CONCLUSION_2 = """
结论 2：环的初始摆放有两个必须做对的地方
    (a) **逐边按自己的目标键长展开**，不能用平均键长。
        甲苯这类环上单双键交替的分子（1.54/1.34），用平均值 1.44
        会让每条环边都差 ±0.1 Å，闭环误差累积到 0.6 Å 以上，
        后面的约束迭代怎么都收不回来，分子直接报废。
    (b) **环上原子的取代基要朝环外摆**。
        通用摆放算法只保证「与入边成理想角」，不区分朝里朝外：
        环碳的两条环键方向往两侧去，锥面有内外两个对称解，
        算法会挑到朝内的那个 —— 实测甲苯的氢全被塞进环里、
        和对面碳原子重叠（H1 落在 (0.52, 1.34)，正好压在碳 0 上）。
        做法：sp² 环碳的取代基放在环平面内朝外（苯环的氢必须共面）；
        sp³ 环碳的取代基朝外并离开环平面。
"""

# ----------------------------------------------------------------------------
# 结论三：几何正确性要靠硬约束，不能靠能量最小化
# ----------------------------------------------------------------------------
CONCLUSION_3 = """
结论 3：几何正确性用硬约束保证，不要用能量最小化
    第一代做法是「约束投影 + 力场梯度下降」。实测它会把
    **本来完全正确的初始几何弄坏**：
        苯的初始几何是键长偏差 0.0000 Å、内角全 120°、严格共面，
        跑完梯度松弛后键长偏差变成 1.04 Å、内角散成 49°~73°。
    原因是它用非键排斥换取了键长松弛 —— 能量从 113 降到 44，
    代价却是把键拉长到 2.5 Å。它优化的是能量，不是几何正确性。

    换用位置约束求解器（PBD）后，键长键角是**硬约束、按构造始终满足**：
    当前 46 个分子的键长偏差全部是 0.0000 Å。
    冲突只会表现为「环被撑开一点」，而不会「键被拉断」。

    附带教训：替换函数时不要用「找到下一个 def 就截断」的脚本 ——
    我这样写把紧跟其后的 RelaxResult 类定义一起删掉了，
    直到运行时才以 NameError 暴露。改动后一定要跑一遍完整测试。
"""


def ring_of(mol) -> list[int]:
    adj = build_adjacency(mol)
    root = max(range(len(mol.atoms)), key=lambda v: adj.degree(v))
    rings = find_rings(mol, adj, spanning_tree(mol, adj, root))
    return max(rings, key=len) if rings else []


def metrics(mol, pos, ring):
    p0 = pos[ring[0]]
    n = v3_normalize(v3_cross(v3_sub(pos[ring[1]], p0), v3_sub(pos[ring[2]], p0)))
    dev = max(abs(v3_dot(v3_sub(pos[i], p0), n)) for i in ring)
    w = len(ring)
    inner = []
    for k in range(w):
        i, j, l = ring[(k - 1) % w], ring[k], ring[(k + 1) % w]
        v1, v2 = v3_normalize(v3_sub(pos[i], pos[j])), v3_normalize(v3_sub(pos[l], pos[j]))
        inner.append(math.degrees(math.acos(max(-1.0, min(1.0, v3_dot(v1, v2))))))
    edges = [v3_dist(pos[ring[k]], pos[ring[(k + 1) % w]]) for k in range(w)]
    return dev, min(inner), max(inner), max(edges) - min(edges)


# 期望值：这些是「正确」的判据，改动后必须仍满足。
#
# 数值来自当前的位置约束求解器（PBD）实测：
#   小环被三条硬键长约束逼成正多边形（内角和恒为 180°/360°），
#   大环的褶皱由「sp³ 想要 109.47°」这个驱动力产生，
#   它的褶皱幅度比真实环己烷（约 0.25 Å）更大，
#   因为本作没有二面角/扭转项去限制它。
#   1.68 Å 仍然是一把清楚的「椅子」，观感正确；这里用它作下界。
EXPECT = {
    "环丙烷": {"dev_max": 0.02, "inner": (59.9, 60.1)},     # 必然等边三角形
    "环丁烷": {"dev_max": 0.05, "inner": (84.0, 96.0)},     # 接近平面正方形
    "环戊烷": {"dev_max": 0.05, "inner": (106.0, 110.0)},   # 几乎平面
    "环己烷": {"dev_min": 0.40, "inner": (108.0, 120.0)},   # 必须椅式！
    "苯": {"dev_max": 0.02, "inner": (119.5, 120.5)},       # 必然共平面 + 精确 120°
    "甲苯": {"dev_max": 0.30, "inner": (118.0, 121.0)},     # 苯环基本共面
}


def main() -> int:
    print("=" * 78)
    print("环几何实测（改动后请对比 EXPECT 判据）")
    print("=" * 78)
    print(f"{'分子':<10}{'环大小':>6}{'离平面(Å)':>12}{'内角范围':>16}{'环键长极差':>12}{'能量':>10}")
    print("-" * 78)

    ok = True
    seen = set()
    for name, spec in SPECS.items():
        if name in seen:
            continue
        seen.add(name)
        mol = parse_spec(spec)
        res = solve(mol)
        ring = ring_of(mol)
        dev, lo, hi, spread = metrics(mol, res.positions, ring)
        flag = ""
        exp = EXPECT.get(name)
        if exp:
            if "dev_max" in exp and dev > exp["dev_max"]:
                flag = "  <-- 超出预期（应更平整）"
                ok = False
            if "dev_min" in exp and dev < exp["dev_min"]:
                flag = "  <-- 超出预期（应褶皱）"
                ok = False
            if "inner" in exp and not (exp["inner"][0] <= lo and hi <= exp["inner"][1]):
                flag = "  <-- 内角超出预期"
                ok = False
        print(f"{name:<10}{len(ring):>6}{dev:>12.4f}{lo:>8.1f}~{hi:<7.1f}"
              f"{spread:>12.4f}{res.energy:>10.4f}{flag}")

    print()
    for block in (CONCLUSION_1, CONCLUSION_2, CONCLUSION_3):
        print(block.strip())
        print("-" * 78)

    print(("全部符合预期 ✓" if ok else "有项目不符合预期 ✗ —— 请查看上面的记录"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
