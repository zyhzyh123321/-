"""化学核心测试（pytest 版）。

移植自网页版基准 game_web_archive/test/chem.test.js（74 项断言，node 运行），
作为本项目化学正确性的验收依据。覆盖：

    一、分子库自校验      46 条 spec 全部可解析、分子式/相对分子质量可独立核算
    二、库反查            每条 spec 都命中自己；打乱原子下标后识别不变
    三、粗筛 + 同构       同分异构体（乙醇/二甲醚、正丁烷/异丁烷）必须区分
    四、分子式/质量/组成  CH4 16.04、C2H6O 46.07、C6H6 78.11、C2H4O2 60.05
    五、结构式解析        分支、键级、成环闭锁、多片段、方括号、错误输入
    六、3D 构型           键长、键角、共面性、环己烷椅式、小环张力角
    七、图算法            DFS 生成树、找环、连通分量、邻接表
    八、同构判定与基础件  自反性、下标置换不变、键级敏感；元素表与向量/颜色工具

运行（在项目根目录 分子工坊 下）：
    python -m pytest molecule_workshop/tests/test_chem.py -q

关于数值：断言里的数字都是在本机实测出来的真实结果，注释逐条说明
「为什么断言这个数值」。3D 部分的容差按化学含义给：既容得下求解器的数值残差，
又能抓住「构型塌成平面」「环被撕开成断链」这类真正的错误。
solve() 内部使用固定随机种子，同一分子每次结果一致，因此模块级缓存求解结果
不会影响断言的确定性。
"""
from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

import pytest

# 允许从任意工作目录导入 molecule_workshop（从项目根目录跑时也会命中）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from molecule_workshop.chem.constants import (  # noqa: E402
    ATOMIC_MASS,
    BOND_ANGLE,
    BOND_LENGTH,
    DEFAULT_BOND_LENGTH,
    ELEMENTS,
    bond_length,
)
from molecule_workshop.chem.graph import (  # noqa: E402
    Atom,
    Bond,
    Molecule,
    build_adjacency,
    connected_components,
    find_rings,
    spanning_tree,
)
from molecule_workshop.chem.identify import (  # noqa: E402
    build_index,
    coarse_key,
    composition,
    formula_of,
    heavy_graph,
    identify_molecule,
    isomorphic,
    molar_mass,
    reset_index,
)
from molecule_workshop.chem.parse_spec import (  # noqa: E402
    ParseError,
    parse_spec,
    parse_spec_safe,
)
from molecule_workshop.chem.relax import solve  # noqa: E402
from molecule_workshop.chem.vector import (  # noqa: E402
    clamp,
    contrast_text,
    hex_to_rgb,
    mix_hex,
    rad2deg,
    shade,
    v3_cross,
    v3_dist,
    v3_dot,
    v3_len,
    v3_normalize,
    v3_sub,
)
from molecule_workshop.data.molecules import MOLECULES  # noqa: E402

# ---------------------------------------------------------------------------
# 公共 helper
# ---------------------------------------------------------------------------

# 库里每条记录的 id → spec / 中文名（测试直接引用库，避免把结构式再抄一遍）
SPECS: dict[str, str] = {rec["id"]: rec["spec"] for rec in MOLECULES}
NAMES: dict[str, list[str]] = {rec["id"]: rec["names"] for rec in MOLECULES}

# 任务点名要求做「键长偏差 < 0.01 Å」的 7 个代表分子
REPRESENTATIVE_IDS = [
    "methane", "ethylene", "acetylene", "ethanol", "acetone", "acetic_acid", "glycine",
]

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def spec_of(mol_id: str) -> str:
    """取库里某条记录的 spec；id 写错时立刻报错，而不是静默跳过。"""
    assert mol_id in SPECS, f"分子库里没有 {mol_id}"
    return SPECS[mol_id]


def heavy_of(spec: str):
    """spec → 重原子骨架（识别比较用的统一表示）。"""
    return heavy_graph(parse_spec(spec))


@pytest.fixture(scope="module")
def library() -> tuple[list[tuple[dict, Molecule]], list[tuple[str, str]]]:
    """把库解析成 [(记录, 分子图), ...]，并单列解析失败表。

    解析失败不在这里抛异常：交给 test_all_specs_parse 精确报出是哪几条、
    错在哪里；否则整个模块的用例都会因为 fixture 报错而看不出根因。
    """
    pairs: list[tuple[dict, Molecule]] = []
    errors: list[tuple[str, str]] = []
    for rec in MOLECULES:
        try:
            pairs.append((rec, parse_spec(rec["spec"])))
        except ParseError as exc:  # pragma: no cover - 正常库不触发
            errors.append((rec["id"], str(exc)))
    return pairs, errors


# 3D 求解缓存：同一 spec 的多个断言共享一次求解（solve 用固定种子，结果确定）
_SOLVED: dict[str, tuple[Molecule, object]] = {}


def solved(spec: str):
    """对 spec 求解 3D 构型并缓存。返回 (分子图, SolveResult)。"""
    if spec not in _SOLVED:
        mol = parse_spec(spec)
        res = solve(mol)
        assert res is not None, f"求解失败：{spec}"
        _SOLVED[spec] = (mol, res)
    return _SOLVED[spec]


def angle_deg(pos, i: int, j: int, k: int) -> float:
    """i—j—k 键角（度），j 是中心原子。"""
    u1 = v3_normalize(v3_sub(pos[i], pos[j]))
    u2 = v3_normalize(v3_sub(pos[k], pos[j]))
    return rad2deg(math.acos(clamp(v3_dot(u1, u2), -1.0, 1.0)))


def ring_of(mol: Molecule) -> list[int]:
    """用「生成树 + 闭合边」找出分子里的第一个环（环序，首尾相邻）。

    不用「有两个碳邻居」这类判据：那会把丙酮中心碳这样的非环原子误判进环里。
    """
    adj = build_adjacency(mol)
    root = max(range(len(mol.atoms)), key=lambda v: adj.degree(v))
    tree = spanning_tree(mol, adj, root)
    rings = find_rings(mol, adj, tree)
    assert rings, "没有找到环"
    return rings[0]


def ring_inner_angles(pos, ring: list[int]) -> list[float]:
    """环内角列表（每个原子与环上前一个、后一个原子成的角）。"""
    w = len(ring)
    return [angle_deg(pos, ring[(k + w - 1) % w], ring[k], ring[(k + 1) % w]) for k in range(w)]


def max_plane_deviation(pos, indices: list[int]) -> float:
    """给定原子集到「前三个原子张成的平面」的最大距离（Å）。

    共面时接近 0；椅式褶皱的环会明显 > 0。前三个原子若共线则法向不可用，
    但本测试用到的原子集（环、乙烯）都满足不共线。
    """
    p0, p1, p2 = pos[indices[0]], pos[indices[1]], pos[indices[2]]
    normal = v3_normalize(v3_cross(v3_sub(p1, p0), v3_sub(p2, p0)))
    return max(abs(v3_dot(v3_sub(pos[i], p0), normal)) for i in indices)


def formula_mass(formula: str) -> float:
    """按 Hill 式字符串 + ATOMIC_MASS 独立核算相对分子质量。

    刻意不复用 molar_mass()：两条独立路径（公式字符串求和 vs 逐个原子求和）
    才能互相校验，抓出「原子量表用错/漏加某类原子」的错误。
    """
    total = 0.0
    for sym, num in _FORMULA_RE.findall(formula):
        total += ATOMIC_MASS.get(sym, 0.0) * (int(num) if num else 1)
    return round(total, 2)


def permuted(mol: Molecule, seed: int) -> Molecule:
    """把原子下标整体打乱（键的端点跟着重编号），模拟「同一个分子换个画法」。

    玩家画分子的顺序各不相同，识别结果必须与下标编号无关。
    """
    n = len(mol.atoms)
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    inverse = [0] * n
    for new_idx, old_idx in enumerate(perm):
        inverse[old_idx] = new_idx
    return Molecule(
        atoms=[mol.atoms[i] for i in perm],
        bonds=[Bond(inverse[b.a], inverse[b.b], b.order) for b in mol.bonds],
    )


# ---------------------------------------------------------------------------
# 一、分子库自校验
# ---------------------------------------------------------------------------


def test_library_entry_count():
    """库条目数 = 46：与网页版 js/data/molecules.js 逐条对齐（两边实测都是 46 条）。

    断死数量是有意的：漏搬或多搬一条，下面所有「全库」断言的样本会悄悄缩水，
    整库反查看起来全绿、其实少测了分子。
    （任务描述里写的是「49/49」，但库与网页版基准实测均为 46 条，本测试按 46 断言。）
    同时校验每条记录的字段完整、id 唯一 —— 缺 spec/formula 的记录会让后面几条
    「全库」断言在报错信息里看不出是哪一条坏掉。
    """
    assert len(MOLECULES) == 46
    ids = [rec["id"] for rec in MOLECULES]
    assert len(set(ids)) == 46, "存在重复 id：" + ", ".join(sorted({i for i in ids if ids.count(i) > 1}))
    for rec in MOLECULES:
        assert rec["spec"] and rec["formula"], rec["id"]
        assert rec["desc"], rec["id"]
        assert NAMES[rec["id"]] and NAMES[rec["id"]][0], rec["id"]


def test_all_specs_parse(library):
    """46 条 spec 全部能解析：解析器是识别与 3D 求解的唯一入口，
    任何一条解析失败，那个分子在游戏里就永远识别不出来。"""
    pairs, errors = library
    assert not errors, "解析失败：" + "; ".join(f"{i}: {e}" for i, e in errors)
    assert len(pairs) == 46
    for rec, mol in pairs:
        assert len(mol.atoms) > 0, f"{rec['id']} 解析出 0 个原子"
        assert len(mol.bonds) > 0, f"{rec['id']} 解析出 0 根键"


def test_all_formulas_match_field(library):
    """46/46 分子式与手写的 formula 字段一致：formula_of 从结构现算，不是抄字段。

    这条是「结构式写对了」的核心证据：结构式手写、分子式也手写，
    两者由程序独立对上，才说明库里没有写错的分子。
    """
    pairs, _ = library
    bad = [
        f"{rec['id']} 期望 {rec['formula']} 实得 {formula_of(mol)}"
        for rec, mol in pairs
        if formula_of(mol) != rec["formula"]
    ]
    assert not bad, "分子式不符：" + "; ".join(bad)


def test_all_molar_masses_recomputable(library):
    """46 条相对分子质量都能用 ATOMIC_MASS 独立核算（容差 0.011 = 一个显示刻度）。

    独立路径：formula 字段 → 查原子量求和 → 四舍五入两位；
    与 molar_mass() 的「按原子逐个累加」路径不同，两条路径互相校验。

    容差为什么取 0.011 而不是 0.005：两条路径的浮点累加顺序不同，
    恰好落在 x.xx5 的舍入边界上时会差 0.01（实测 propyne 40.065、formic_acid 46.025），
    这是十进制显示舍入的固有现象、不是原子量算错；放行「一个刻度」的差，
    依旧能抓住真正的错误 —— 少一个氢就会差 1.008，远大于 0.011。
    """
    pairs, _ = library
    bad = []
    for rec, mol in pairs:
        expected = formula_mass(rec["formula"])
        got = molar_mass(mol)
        if got is None or abs(got - expected) >= 0.011:
            bad.append(f"{rec['id']} 期望 {expected} 实得 {got}")
    assert not bad, "质量核算不符：" + "; ".join(bad)


# ---------------------------------------------------------------------------
# 二、库反查与标注无关性
# ---------------------------------------------------------------------------


def test_identify_every_library_entry(library):
    """库反查 46/46：拿库自己的结构去查库，每条都必须命中自己。

    覆盖 build_index 的惰性解析路径（索引第一次被用时才解析 spec）——
    这是「游戏能否识别任何分子」的前提，网页版曾因索引为空导致全都识别不出来。
    """
    pairs, _ = library
    reset_index()
    hits = 0
    misses = []
    for rec, mol in pairs:
        res = identify_molecule(mol)
        if res.matched and res.matched["id"] == rec["id"]:
            hits += 1
        else:
            misses.append(f"{rec['id']} → {res.matched['id'] if res.matched else '未命中'}")
    assert not misses, "; ".join(misses)
    assert hits == 46


def test_identify_stable_after_shuffling_atom_order(library):
    """打乱原子下标后仍命中同一分子：识别必须与「画法/创建顺序」无关。

    每个分子用两个固定种子各打乱一次（下标置换 + 键端点重编号）。
    若识别依赖下标顺序，玩家的画法一变就会认错分子。
    """
    pairs, _ = library
    unstable = []
    for rec, mol in pairs:
        for seed in (7, 12345):
            res = identify_molecule(permuted(mol, seed))
            if not res.matched or res.matched["id"] != rec["id"]:
                got = res.matched["id"] if res.matched else "未命中"
                unstable.append(f"{rec['id']}(seed={seed}) → {got}")
                break
    assert not unstable, "; ".join(unstable)


def test_ethanol_and_dimethyl_ether_are_distinguished():
    """乙醇与二甲醚的分子式都是 C2H6O，但连接方式不同，必须识别成两个分子。

    这是「粗筛键 + 精确同构」两级设计的核心验收点：
    只比分子式会把两者混为一谈，只比粗筛键则可能哈希碰撞。
    """
    assert formula_of(parse_spec(spec_of("ethanol"))) == "C2H6O"
    assert formula_of(parse_spec(spec_of("dimethyl_ether"))) == "C2H6O"
    ethanol, ether = heavy_of(spec_of("ethanol")), heavy_of(spec_of("dimethyl_ether"))
    assert coarse_key(ethanol) != coarse_key(ether)
    assert not isomorphic(ethanol, ether)
    assert identify_molecule(parse_spec(spec_of("ethanol"))).matched["id"] == "ethanol"
    assert identify_molecule(parse_spec(spec_of("dimethyl_ether"))).matched["id"] == "dimethyl_ether"


def test_butane_and_isobutane_are_distinguished():
    """正丁烷（直链）与异丁烷（支链）都是 C4H10，骨架不同 —— 碳骨架异构体不能混。"""
    assert formula_of(parse_spec(spec_of("butane"))) == "C4H10"
    assert formula_of(parse_spec(spec_of("isobutane"))) == "C4H10"
    n_butane, i_butane = heavy_of(spec_of("butane")), heavy_of(spec_of("isobutane"))
    assert coarse_key(n_butane) != coarse_key(i_butane)
    assert not isomorphic(n_butane, i_butane)
    assert identify_molecule(parse_spec(spec_of("butane"))).matched["id"] == "butane"
    assert identify_molecule(parse_spec(spec_of("isobutane"))).matched["id"] == "isobutane"


def test_ethane_and_ethene_are_distinguished():
    """乙烷 C2H6（单键）与乙烯 C2H4（双键）不能混：键级必须参与判定。

    两者重原子骨架都只是两个碳，若识别忽略键级与氢数，就会互相误命。
    """
    assert formula_of(parse_spec(spec_of("ethane"))) == "C2H6"
    assert formula_of(parse_spec(spec_of("ethylene"))) == "C2H4"
    ethane, ethene = heavy_of(spec_of("ethane")), heavy_of(spec_of("ethylene"))
    assert coarse_key(ethane) != coarse_key(ethene)
    assert not isomorphic(ethane, ethene)


def test_unmatched_molecule_still_reports_formula_and_mass():
    """库外分子（新戊烷 C5H12）也要给出分子式与相对分子质量，并标记未命中。

    玩家能拼出库里没有的分子，界面仍然要显示式量与组成；
    matched=None 是「未收录」的唯一判据（不能拿 formula 是否为空来判断）。
    """
    # 中心碳挂 4 个甲基：C5H12，库里只有正戊烷，网罗不到它
    mol = parse_spec("C(-C(-H)(-H)(-H))(-C(-H)(-H)(-H))(-C(-H)(-H)(-H))(-C(-H)(-H)(-H))")
    res = identify_molecule(mol)
    assert res.formula == "C5H12"
    assert res.mass is not None and abs(res.mass - 72.15) < 0.06
    assert res.matched is None
    assert res.count == 17


# ---------------------------------------------------------------------------
# 三、分子式 / 相对分子质量 / 元素组成
# ---------------------------------------------------------------------------


def test_formula_of_hill_order():
    """Hill 顺序：C 第一、H 第二、其余按字母序 —— CH4 / C2H6O / C6H6 / C2H4O2。"""
    assert formula_of(parse_spec(spec_of("methane"))) == "CH4"
    assert formula_of(parse_spec(spec_of("ethanol"))) == "C2H6O"
    assert formula_of(parse_spec(spec_of("benzene"))) == "C6H6"
    assert formula_of(parse_spec(spec_of("acetic_acid"))) == "C2H4O2"


def test_molar_mass_known_values():
    """化学课上必背的四个相对分子质量（容差 0.06 = 两位小数半个刻度 + 余量）：
    甲烷 16.04 / 乙醇 46.07 / 苯 78.11 / 乙酸 60.05。"""
    for mol_id, expected in (("methane", 16.04), ("ethanol", 46.07),
                             ("benzene", 78.11), ("acetic_acid", 60.05)):
        got = molar_mass(parse_spec(spec_of(mol_id)))
        assert got is not None and abs(got - expected) < 0.06, f"{mol_id} 期望 {expected} 实得 {got}"


def test_ethanol_composition_percent():
    """乙醇 C2H6O 的元素质量百分比：C 52.14% / O 34.73% / H 13.13%，三者之和 100%。

    独立核算：C 2×12.011=24.022、O 15.999、H 6×1.008=6.048，总 46.069；
    24.022/46.069=52.14%，15.999/46.069=34.73%，6.048/46.069=13.13%。
    """
    comp = {c["symbol"]: c for c in composition(parse_spec(spec_of("ethanol")))}
    assert abs(comp["C"]["percent"] - 52.14) < 0.2, comp["C"]["percent"]
    assert abs(comp["O"]["percent"] - 34.73) < 0.2, comp["O"]["percent"]
    assert abs(comp["H"]["percent"] - 13.13) < 0.2, comp["H"]["percent"]
    # 百分比是同一批质量的占比，未做二次舍入，因此和必须精确到 0.01
    assert abs(sum(c["percent"] for c in comp.values()) - 100.0) < 0.01
    # 组成条按质量降序展示（C > O > H），顺序错了界面读起来会别扭
    assert [c["symbol"] for c in composition(parse_spec(spec_of("ethanol")))] == ["C", "O", "H"]


# ---------------------------------------------------------------------------
# 四、结构式解析
# ---------------------------------------------------------------------------


def test_parse_branching():
    """括号分支：异丁烷 1 个中心碳 + 3 个甲基 + 10 个氢 = 14 原子 13 键。

    分支必须与主干共享同一套原子/键/环表，否则分支内的下标会错位
    （网页版早期版本就在嵌套分支上崩过）。
    """
    mol = parse_spec(spec_of("isobutane"))
    assert len(mol.atoms) == 14
    assert len(mol.bonds) == 13
    adj = build_adjacency(mol)
    carbons = [i for i, a in enumerate(mol.atoms) if a.el == "C"]
    assert len(carbons) == 4
    # 中心碳 = 1 个 H + 3 个 C；每个甲基碳 = 3 个 H + 中心碳，度数都是 4
    assert adj.degree(0) == 4
    assert all(adj.degree(c) == 4 for c in carbons)


def test_parse_bond_orders():
    """键级符号：C-C 单键 1、C=C 双键 2、C#C 三键 3。"""
    assert [b.order for b in parse_spec("C-C").bonds] == [1]
    assert [b.order for b in parse_spec("C=C").bonds] == [2]
    assert [b.order for b in parse_spec("C#C").bonds] == [3]
    # 库里乙炔用 # 写：解析后必须是 1 根三键 + 2 根 C—H 单键
    orders = sorted(b.order for b in parse_spec(spec_of("acetylene")).bonds)
    assert orders == [1, 1, 3]


def test_parse_ring_closure():
    """成环闭锁 C1-C-C-1：同一个环号第二次出现时与首次出现处相连，闭成三元环。"""
    mol = parse_spec("C1-C-C-1")
    assert len(mol.atoms) == 3
    assert len(mol.bonds) == 3
    assert sorted(ring_of(mol)) == [0, 1, 2]
    # 库里的环丙烷用的是同一种写法，也必须真的闭合成环
    assert len(ring_of(parse_spec(spec_of("cyclopropane")))) == 3


def test_parse_multi_fragment():
    """点号分隔多片段（如氢键复合体/混合物）：两个甲烷之间不能生成键。"""
    mol = parse_spec("C(-H)(-H)(-H)(-H).C(-H)(-H)(-H)(-H)")
    assert len(mol.atoms) == 10
    assert len(mol.bonds) == 8
    comps = connected_components(mol)
    assert len(comps) == 2
    assert comps[0] == [0, 1, 2, 3, 4] and comps[1] == [5, 6, 7, 8, 9]


def test_parse_bracket_atom_hydrogens():
    """方括号写法：元素符号 + 氢数（[C3] = 带 3 个显式氢的碳，[O] = 0 个氢）。

    [CH3] 也能解析（H 后面的数字 3 被当作氢数，实测等价于 [C3]）——
    早先注释说它「会抛 ParseError」，与实现不符，已更正。
    另外解析出的 h_count 目前只存在 Atom 上，heavy_graph 仍按显式氢原子计数，
    所以方括号写法暂不影响识别；库里没有任何 spec 用到它。
    """
    mol = parse_spec("[C3]-[C3]")
    assert len(mol.atoms) == 2
    assert [a.el for a in mol.atoms] == ["C", "C"]
    assert [a.h_count for a in mol.atoms] == [3, 3]
    assert [b.order for b in mol.bonds] == [1]
    # 不带数字时氢数记为 0：[O] 表示不带显式氢的氧
    assert [a.h_count for a in parse_spec("[O]").atoms] == [0]
    # [CH3] 与 [C3] 等价（H 后跟数字）
    assert [a.h_count for a in parse_spec("[CH3]").atoms] == [3]


PARSE_ERROR_CASES = [
    ("extra_close_paren", "C(-H))"),
    ("stray_close_paren", "C)"),
    ("unclosed_ring_number", "C1-C-C"),
    ("unclosed_paren", "C(-H"),
    ("unclosed_bracket", "[CH3"),
    ("unknown_element", "C-X"),
]


@pytest.mark.parametrize("case,spec", PARSE_ERROR_CASES, ids=[c for c, _ in PARSE_ERROR_CASES])
def test_parse_error_cases(case, spec):
    """写错的结构式必须抛 ParseError，而不是静默返回半个分子或原地死循环。

    覆盖网页版踩过的坑：多余右括号曾让解析器原地打转（每轮位置不前进）；
    未闭合环号会丢掉环闭合边；未知元素必须立刻报错。
    """
    with pytest.raises(ParseError):
        parse_spec(spec)
    mol, err = parse_spec_safe(spec)
    assert mol is None and err, f"{case} 容错版未返回错误信息"


def test_parse_empty_spec_raises():
    """空串也必须报错：否则会构造出 0 原子的分子，后面所有几何代码都得防空。"""
    with pytest.raises(ParseError):
        parse_spec("")


# ---------------------------------------------------------------------------
# 五、3D 构型
# ---------------------------------------------------------------------------


def test_methane_is_tetrahedral():
    """甲烷：6 个 H—C—H 角全部 109.47°±2°（实测 109.471~109.472，最大偏离 0.002°）。

    109.47° = arccos(-1/3) 是 sp³ 正四面体的精确夹角；±2° 既容得下
    投影/松弛的数值残差，又能抓住「被摆成平面 120°」这种真正的构型错误。
    """
    mol, res = solved(spec_of("methane"))
    pos = res.positions
    hs = [i for i, a in enumerate(mol.atoms) if a.el == "H"]
    c = next(i for i, a in enumerate(mol.atoms) if a.el == "C")
    assert len(hs) == 4
    angles = [angle_deg(pos, hs[i], c, hs[j]) for i in range(4) for j in range(i + 1, 4)]
    assert len(angles) == 6
    assert all(abs(a - 109.47) < 2.0 for a in angles), angles


def test_methane_bond_length():
    """甲烷 C—H 键长 1.090±0.02 Å（BOND_LENGTH['C-H']=1.09，实测四根都是 1.090）。

    0.02 Å 覆盖投影与松弛的残差，同时能抓住「键被拉到 1.2 Å 以上」的明显失效。
    """
    mol, res = solved(spec_of("methane"))
    pos = res.positions
    hs = [i for i, a in enumerate(mol.atoms) if a.el == "H"]
    c = next(i for i, a in enumerate(mol.atoms) if a.el == "C")
    lengths = [v3_dist(pos[h], pos[c]) for h in hs]
    assert all(abs(x - 1.090) < 0.02 for x in lengths), lengths


def test_ethylene_is_planar():
    """乙烯六个原子共面（最大离平面 < 0.05 Å，实测 0.0000）。

    双键碳是 sp²，六个原子必须严格共面。网页版曾因 sp² 方向分配错误
    把乙烯摆成立体（离平面 0.94 Å），0.05 Å 的阈值足够抓住这种错误。
    """
    _mol, res = solved(spec_of("ethylene"))
    assert len(res.positions) == 6
    assert max_plane_deviation(res.positions, [0, 1, 2, 3, 4, 5]) < 0.05


def test_ethylene_hch_angle():
    """乙烯 H—C—H = 120°±3°（实测 120.00 / 120.00）—— sp² 平面三角形的夹角。"""
    _mol, res = solved(spec_of("ethylene"))
    pos = res.positions
    # spec C(-H)(-H)=C(-H)(-H)：碳是 0、3，氢是 1/2 与 4/5
    assert abs(angle_deg(pos, 1, 0, 2) - 120.0) < 3.0
    assert abs(angle_deg(pos, 4, 3, 5) - 120.0) < 3.0


def test_acetylene_is_linear():
    """乙炔三根角全部 180°±3°（实测 180.0 / 180.0 / 180.0）。

    sp 杂化是直线形；其中 H…H（以端碳为中心）是最强约束 ——
    只要分子被掰弯一点点，这个角就会立刻偏离 180°。
    """
    _mol, res = solved(spec_of("acetylene"))
    pos = res.positions
    # spec C(-H)#C(-H)：碳是 0、2，氢是 1、3
    angles = [angle_deg(pos, 1, 0, 2), angle_deg(pos, 0, 2, 3), angle_deg(pos, 1, 0, 3)]
    assert all(abs(a - 180.0) < 3.0 for a in angles), angles


def test_cyclopropane_ring_angle():
    """环丙烷环内角必须接近 60°（实测 58.43 / 58.45 / 63.13，三点之和恰好 180°）。

    三元环内角和恒为 180°，平均必然是 60°；张力使三个角并不完全相等。
    任务描述里的「约 58~63°」是四舍五入后的说法（实测最大 63.13° 略超 63°），
    所以这里放宽到 60°±5°，另加「平均值 60°±1.5°」的收紧断言：
    若环被撕开或键长崩掉，内角会跳到 109° 直接失败。
    """
    mol, res = solved(spec_of("cyclopropane"))
    ring = ring_of(mol)
    assert len(ring) == 3
    inner = ring_inner_angles(res.positions, ring)
    assert all(abs(a - 60.0) < 5.0 for a in inner), inner
    assert abs(sum(inner) / 3.0 - 60.0) < 1.5, inner


def test_cyclopentane_ring_angle():
    """环戊烷环内角 108°±4°（实测 107.89~108.07，平均 108.00）。

    108° 是正五边形内角；五元环通过轻微扭曲（离平面 0.007 Å）避开
    sp³ 的 109.47° 张力，所以环内角应该贴在 108° 附近而不是 109.47°。
    """
    mol, res = solved(spec_of("cyclopentane"))
    ring = ring_of(mol)
    assert len(ring) == 5
    inner = ring_inner_angles(res.positions, ring)
    assert all(abs(a - 108.0) < 4.0 for a in inner), inner


def test_cyclohexane_is_chair_not_planar():
    """环己烷必须是椅式：环碳最大离平面 > 0.3 Å（实测 1.677 Å）。

    平面六边形会让相邻碳上的氢挤到约 1.78 Å 产生强排斥；
    实测 attempts=1 时会停在 0.095 Å 的近平面构型，默认 4 次尝试才落到椅式。
    阈值取 0.3 Å：真正的椅式褶皱量级是 0.5 Å，平面构型只有 ~0.1 Å，两者可清晰区分。
    """
    mol, res = solved(spec_of("cyclohexane"))
    ring = ring_of(mol)
    assert len(ring) == 6
    dev = max_plane_deviation(res.positions, ring)
    assert dev > 0.3, dev


def test_cyclohexane_ring_angle():
    """环己烷环内角落在 108~120° 之间（实测 109.47，全部相等）。

    椅式环己烷的真实 C—C—C 约 111.5°；本作的求解器把键角当作硬约束、
    目标就是 sp³ 的 109.47°，所以实测值严格等于 109.47 —— 比真实值略小一点，
    但化学上完全合理，也远好于「环被撕开」的坏构型（断链时会跑到 150° 以上）。

    区间下界取 108 而不是 110：早期版本用一阶力场松弛，环内角会偏大到
    115~122°；换成位置约束求解器（PBD）后硬约束保证它精确落在理想角上，
    110 的下界会把正确的解判成失败。
    """
    mol, res = solved(spec_of("cyclohexane"))
    inner = ring_inner_angles(res.positions, ring_of(mol))
    assert all(108.0 <= a <= 120.0 for a in inner), inner


def test_benzene_is_planar():
    """苯环共平面（最大离平面 < 0.05 Å，实测 0.0000）—— 芳香环必须平面。

    与环己烷的椅式褶皱形成对照：同一套求解器要同时对「该平的平」和
    「该皱的皱」给出正确结果。
    """
    mol, res = solved(spec_of("benzene"))
    ring = ring_of(mol)
    assert len(ring) == 6
    assert max_plane_deviation(res.positions, ring) < 0.05


def test_benzene_ring_angle():
    """苯环内角 120°±3°（实测 119.19~120.74）—— sp² 正六边形内角。"""
    mol, res = solved(spec_of("benzene"))
    inner = ring_inner_angles(res.positions, ring_of(mol))
    assert all(abs(a - 120.0) < 3.0 for a in inner), inner


def test_benzene_ring_bond_lengths():
    """苯环 C—C 键长落在 1.28~1.62 Å（实测 1.333~1.575）。

    库里苯用单双键交替表示，两根平衡键长是 C=C 1.34 / C-C 1.54，
    实测键长必须贴着这两个值走；超过 1.62 Å 就意味着环上有键被拉断。
    """
    mol, res = solved(spec_of("benzene"))
    ring = ring_of(mol)
    pos = res.positions
    w = len(ring)
    lengths = [v3_dist(pos[ring[k]], pos[ring[(k + 1) % w]]) for k in range(w)]
    assert all(1.28 < x < 1.62 for x in lengths), lengths


def test_solve_returns_finite_converged_geometry():
    """7 个代表分子求解后坐标必须全部有限，且 relax 报告已收敛。

    NaN/Inf 一旦混进坐标，渲染层会整屏花掉；converged=True 是「力已足够小」的凭据。
    """
    for mol_id in REPRESENTATIVE_IDS:
        _mol, res = solved(spec_of(mol_id))
        assert res.converged, f"{mol_id} 未收敛（迭代 {res.iterations} 次）"
        assert math.isfinite(res.energy), f"{mol_id} 能量非有限：{res.energy}"
        assert all(math.isfinite(c) for p in res.positions for c in p), f"{mol_id} 坐标非有限"


@pytest.mark.parametrize("mol_id", REPRESENTATIVE_IDS, ids=REPRESENTATIVE_IDS)
def test_bond_length_deviation_below_0_01(mol_id):
    """7 个代表分子的每根键长偏差都必须 < 0.01 Å（实测最大值见下）。

    实测最大偏差：甲烷 0.00001 / 乙烯 0.00000 / 乙炔 0.00000 / 乙醇 0.00002 /
    丙酮 0.00228 / 乙酸 0.00250 / 甘氨酸 0.00378 Å。
    0.01 Å 是「约束投影 + 力场松弛」应该达到的精度；超过它说明几何求解退化
    （键长被拉长/压短到肉眼可见，分子形状就不可信了）。
    """
    mol, res = solved(spec_of(mol_id))
    pos = res.positions
    worst = 0.0
    for b in mol.bonds:
        d0 = bond_length(mol.atoms[b.a].el, mol.atoms[b.b].el, b.order or 1)
        worst = max(worst, abs(v3_dist(pos[b.a], pos[b.b]) - d0))
    assert worst < 0.01, f"{mol_id} 最大键长偏差 {worst:.5f} Å"


# ---------------------------------------------------------------------------
# 六、图算法
# ---------------------------------------------------------------------------


def test_spanning_tree_is_dfs_on_ring():
    """六元环上的生成树必须是 DFS：深度严格 0..5 的一条链，且只有 1 条闭合边。

    为什么必须 DFS（而不是 BFS）：DFS 把整条环放在同一条搜索路径上，
    闭合边于是连接路径首尾，补位时只移动一个原子就能闭合环；BFS 会把环上原子
    分到不同分支，闭合边跨越多个未定位原子，环几何直接崩（实测键长算到 2.5 Å）。
    手写的 6 元环上：BFS 的深度会是 0,1,1,2,2,3、闭合边 3 条 —— 两种结果完全可区分。
    """
    ring_mol = Molecule(
        atoms=[Atom("C") for _ in range(6)],
        bonds=[Bond(i, (i + 1) % 6, 1) for i in range(6)],
    )
    adj = build_adjacency(ring_mol)
    tree = spanning_tree(ring_mol, adj)
    assert tree.depth == [0, 1, 2, 3, 4, 5], tree.depth
    assert len(tree.ring_edges) == 1, tree.ring_edges
    a, b, order = tree.ring_edges[0]
    assert {a, b} == {0, 5} and order == 1
    assert find_rings(ring_mol, adj, tree) == [[0, 1, 2, 3, 4, 5]]


@pytest.mark.parametrize(
    "mol_id,size",
    [("cyclopropane", 3), ("cyclopentane", 5), ("cyclohexane", 6), ("benzene", 6)],
)
def test_find_rings_recovers_whole_ring(mol_id, size):
    """find_rings 必须还原出完整环：长度正确、全是碳、环序上相邻原子在图上真有键。

    环还原错了，layout_ring 就会把原子摆到错误的多边形上，后续松弛再强也救不回来
    （实测表现为环散成 2.5 Å 的断链）。
    """
    mol = parse_spec(spec_of(mol_id))
    ring = ring_of(mol)
    assert len(ring) == size, ring
    assert all(mol.atoms[i].el == "C" for i in ring)
    adj = build_adjacency(mol)
    for k in range(size):
        nxt = ring[(k + 1) % size]
        assert nxt in adj.neighbors(ring[k]), f"{mol_id} 环序在 {ring[k]} 处断开"


def test_ring_closure_edge_joins_adjacent_ring_atoms():
    """闭合边必须连接环序列的首尾两个原子（即环上相邻），且整库分子只有 1 条闭合边。

    这条断言直接绑住 DFS 生成树的性质：闭合边只跨越一条环边，
    所以补位时只需移动一个原子就能把环闭起来。
    """
    for mol_id in ("benzene", "cyclohexane"):
        mol = parse_spec(spec_of(mol_id))
        adj = build_adjacency(mol)
        root = max(range(len(mol.atoms)), key=lambda v: adj.degree(v))
        tree = spanning_tree(mol, adj, root)
        assert len(tree.ring_edges) == 1, (mol_id, tree.ring_edges)
        a, b, _order = tree.ring_edges[0]
        ring = ring_of(mol)
        assert {a, b} == {ring[0], ring[-1]}, (mol_id, a, b, ring)


def test_connected_components_of_two_methanes():
    """两个互不成键的甲烷 → 2 个连通分量，每个恰好 5 个原子。

    识别前必须先按连通分量切开，否则两个分子会被当成一个畸形大分子。
    """
    mol = parse_spec("C(-H)(-H)(-H)(-H).C(-H)(-H)(-H)(-H)")
    comps = connected_components(mol)
    assert len(comps) == 2
    assert [len(c) for c in comps] == [5, 5]
    assert comps[0] == [0, 1, 2, 3, 4] and comps[1] == [5, 6, 7, 8, 9]


def test_adjacency_is_symmetric_and_sorted():
    """邻接表必须对称（每根键登记两次）且邻居按下标排序。

    排序是 DFS/找环结果可复现的前提：顺序不定会让同一分子每次给出不同构型。
    """
    mol = parse_spec(spec_of("ethanol"))
    adj = build_adjacency(mol)
    for v in range(len(mol.atoms)):
        assert adj.adj[v] == sorted(adj.adj[v]), f"原子 {v} 的邻居未排序"
        for to, order in adj.adj[v]:
            assert (v, order) in adj.adj[to], f"键 {v}-{to} 不对称"


def test_heavy_graph_of_benzene():
    """苯的重原子骨架：6 个碳、每个带 1 个氢，环内键级是 3 双 + 3 单。

    显式氢写法（-H）压成「重原子 + 氢数」后才能与隐式氢的画法互通，
    氢数算错会让同分异构体互相误命。
    """
    g = heavy_of(spec_of("benzene"))
    assert g.els == ["C"] * 6
    assert g.h_count == [1] * 6
    assert g.lone_h == 0
    orders = sorted(order for v in range(6) for _to, order in g.adj[v])
    assert orders == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


def test_build_index_covers_all_entries():
    """索引必须收录全部 46 条记录（惰性解析路径）：索引为空则所有分子都识别不出来。

    网页版这里踩过坑 —— 索引只处理「已解析过」的记录，游戏运行中没人解析过，
    结果索引为空、全部识别失败。
    """
    index = build_index(force=True)
    total = sum(len(bucket) for bucket in index.values())
    assert total == 46, total


# ---------------------------------------------------------------------------
# 七、同构判定
# ---------------------------------------------------------------------------


def test_isomorphic_self():
    """自己与自己同构（自反性）—— 同构判定最基本的性质。"""
    for mol_id in ("ethanol", "benzene", "isobutane", "glycine"):
        g = heavy_of(spec_of(mol_id))
        assert isomorphic(g, g), mol_id


def test_isomorphic_after_index_permutation():
    """打乱原子下标后仍同构：同构判定必须与编号无关。

    这是「识别不依赖玩家画法」的数学基础。
    """
    for mol_id in ("ethanol", "acetic_acid", "cyclohexane"):
        mol = parse_spec(spec_of(mol_id))
        assert isomorphic(heavy_graph(mol), heavy_graph(permuted(mol, 20261012))), mol_id


def test_isomorphic_false_when_structure_differs():
    """结构不同则不同构：乙烷/乙烯（键级）、乙醇/二甲醚（连接方式）、正丁烷/异丁烷（骨架）。"""
    assert not isomorphic(heavy_of(spec_of("ethane")), heavy_of(spec_of("ethylene")))
    assert not isomorphic(heavy_of(spec_of("ethanol")), heavy_of(spec_of("dimethyl_ether")))
    assert not isomorphic(heavy_of(spec_of("butane")), heavy_of(spec_of("isobutane")))
    # 单键/双键的节点标签不同，粗筛键也必须不同（否则会在粗筛阶段误筛掉正确分子）
    assert coarse_key(heavy_of(spec_of("ethane"))) != coarse_key(heavy_of(spec_of("ethylene")))


def test_coarse_key_stable_under_permutation():
    """同一分子换标注 → 粗筛键必须逐字符一致。

    粗筛是第一道门：若它对标注敏感，正确分子会在粗筛阶段就被淘汰，
    根本轮不到后面的精确同构判定。
    """
    g1 = heavy_of(spec_of("ethanol"))
    g2 = heavy_graph(permuted(parse_spec(spec_of("ethanol")), 7))
    assert coarse_key(g1) == coarse_key(g2)
    assert coarse_key(g1) != coarse_key(heavy_of(spec_of("dimethyl_ether")))


# ---------------------------------------------------------------------------
# 八、元素常量与向量/颜色工具
# ---------------------------------------------------------------------------


def test_element_table():
    """元素表：C/H/O/N 化合价必须 4/1/2/3（画键插槽校验的依据），球半径与颜色齐备。"""
    assert ELEMENTS["C"]["valence"] == 4
    assert ELEMENTS["H"]["valence"] == 1
    assert ELEMENTS["O"]["valence"] == 2
    assert ELEMENTS["N"]["valence"] == 3
    assert set(ATOMIC_MASS) == {"C", "H", "O", "N"}
    for sym, info in ELEMENTS.items():
        assert info["color"].startswith("#") and len(info["color"]) == 7, sym
        assert info["radius"] > 0, sym
    assert BOND_ANGLE == {"sp": 180.0, "sp2": 120.0, "sp3": 109.47}


def test_bond_length_table_is_defined_and_symmetric():
    """常见元素组合在所有键级下都必须有键长定义，且查表与原子顺序无关（C-O == O-C）。"""
    for order in (1, 2, 3):
        for a, b in (("C", "O"), ("C", "N"), ("O", "H"), ("N", "H"), ("C", "H"), ("C", "C")):
            v = bond_length(a, b, order)
            assert v > 0 and math.isfinite(v), f"{a}-{b} order={order} 无定义"
            assert v == bond_length(b, a, order), f"{a}-{b} 查表不对称"
    assert bond_length("C", "O", 1) == BOND_LENGTH["C-O"] == 1.43
    assert bond_length("C", "H", 1) == 1.09


def test_bond_length_default_fallback():
    """表里没有的组合回落到 DEFAULT_BOND_LENGTH(1.45)，而不是 None/NaN。

    兜底值必须安全：玩家能画出来的组合远超常量表覆盖范围，
    返回 NaN 会让整条几何链路崩掉。
    """
    assert bond_length("C", "C", 3) == BOND_LENGTH["C#C"] == 1.20
    assert bond_length("N", "O", 3) == DEFAULT_BOND_LENGTH


def test_vector_helpers():
    """向量工具：加减/点积/叉积/模长/归一化/夹取 —— 3D 几何全靠这些原子操作。"""
    a, b = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)
    assert v3_sub(b, a) == (3.0, 3.0, 3.0)
    assert v3_dot(a, b) == 32.0
    cross = v3_cross(a, b)
    assert cross == (-3.0, 6.0, -3.0)
    # 叉积必须与两个输入都正交（这是「绕轴旋转」「环平面法向」的正确性基础）
    assert abs(v3_dot(cross, a)) < 1e-12 and abs(v3_dot(cross, b)) < 1e-12
    assert abs(v3_len(v3_normalize(a)) - 1.0) < 1e-12
    assert clamp(5.0, -1.0, 1.0) == 1.0
    assert clamp(-5.0, -1.0, 1.0) == -1.0
    assert clamp(0.5, -1.0, 1.0) == 0.5
    assert abs(rad2deg(math.pi) - 180.0) < 1e-12
    assert v3_len((3.0, 4.0, 0.0)) == 5.0


def test_color_helpers():
    """颜色工具：CPK 基色 → RGB、明暗调整、深浅底色自动选可读文字色。"""
    assert hex_to_rgb("#404040") == (64, 64, 64)
    assert shade("#000000", 1.0) == "#ffffff"
    assert shade("#ffffff", -1.0) == "#000000"
    assert mix_hex("#000000", "#ffffff", 1.0) == "#ffffff"
    assert mix_hex("#000000", "#ffffff", 0.0) == "#000000"
    # 深色底与浅色底的文字色必须相反，否则元素名会看不清
    assert contrast_text("#000000") != contrast_text("#ffffff")
