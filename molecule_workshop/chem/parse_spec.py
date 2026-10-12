"""结构式字符串解析器。

把分子库里手写的 spec 字符串解析成分子图（供自校验与库识别使用）。

支持的写法（等价于常用 SMILES 写法）：
    C(-H)(-H)        分支
    C=C  C#C  C-O    键级
    C1-C-C-C-1       成环闭锁（同一个数字表示首尾相连）
    [CH3]  [C3]      方括号原子：元素后面可以写氢数，也可以只写个数
    C.C              用点分隔的多个片段

说明：方括号里写出的氢数只作为「结构式里的显式标注」记录在 Atom.h_count 上，
不参与本作的分子识别 —— 识别用的是重原子骨架 + 与重原子实际成键的氢数。
游戏里玩家摆出的分子一律是显式原子，内置库的结构式也都把氢写全了，
所以两种写法不会造成识别差异。

分支解析用同一个上下文共享原子/键/环表，因此不会出现索引错位或环号串扰。
之所以自己做而不引第三方库：环境断网，且只需覆盖有机物这点语法。
"""
from __future__ import annotations

import re

from .constants import ELEMENTS
from .graph import Atom, Bond, Molecule

# 键级符号
_BOND_CHAR = {"-": 1, "=": 2, "#": 3, ":": 1, "/": 1, "\\": 1}

# 方括号内容：元素 + 任意顺序的「H」和数字，例如 CH3 / C3 / C2H5 / CH
# 早先这里只写 (\d*)，遇到文档里示例的 [CH3] 会直接报错 —— 文档与实现不一致。
_BRACKET_RE = re.compile(r"^([A-Z][a-z]?)(?:H(\d*))?(\d*)$")


class ParseError(ValueError):
    """结构式写错时抛出，带出错位置，便于定位。"""


class _Context:
    __slots__ = ("atoms", "bonds", "rings")

    def __init__(self) -> None:
        self.atoms: list[Atom] = []
        self.bonds: list[Bond] = []
        self.rings: dict[str, int] = {}

    def add_atom(self, el: str, h_count: int | None) -> int:
        self.atoms.append(Atom(el=el, h_count=h_count))
        return len(self.atoms) - 1

    def connect(self, a: int, b: int, order: int) -> None:
        if a < 0 or b < 0 or a == b:
            return
        for bond in self.bonds:
            if (bond.a == a and bond.b == b) or (bond.a == b and bond.b == a):
                if order > bond.order:
                    bond.order = order
                return
        self.bonds.append(Bond(a, b, order))


class _Parser:
    """单次解析的状态：源串 + 当前位置。

    位置用下标推进而不是切片，且严格要求「每次循环都必须前进」——
    网页版曾因为没有这个防护，在括号嵌套写错的结构式上死循环。
    """

    __slots__ = ("src", "pos")

    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0

    # -- 主入口 -----------------------------------------------------------------

    def parse_all(self) -> _Context:
        ctx = _Context()
        guard = 0
        while self.pos < len(self.src):
            before = self.pos
            self._walk(ctx, -1, 1)
            if self.pos <= before:
                ch = self.src[before] if before < len(self.src) else "EOF"
                raise ParseError(
                    f"第 {before} 个字符处解析没有前进（字符 {ch!r}）：结构式可能有误"
                )
            guard += 1
            if guard > len(self.src) * 4 + 64:
                raise ParseError("结构式解析异常：迭代次数超出上限")
        open_rings = sorted(ctx.rings)
        if open_rings:
            raise ParseError("有未闭合的环编号：" + ",".join(open_rings))
        return ctx

    # -- 单层扫描（遇到 ')' 返回）------------------------------------------------

    def _walk(self, ctx: _Context, base_atom: int, base_order: int) -> None:
        current = base_atom
        pending = base_order

        while self.pos < len(self.src):
            ch = self.src[self.pos]

            if ch in " \t\r\n":
                self.pos += 1
                continue
            if ch == ")":
                return
            if ch == ".":
                current = -1
                pending = 1
                self.pos += 1
                continue

            # 分支
            if ch == "(":
                if current < 0:
                    raise ParseError(f"第 {self.pos} 个字符处：分支没有前导原子")
                self.pos += 1                      # 跳过 '('
                self._walk(ctx, current, pending)
                if self.pos >= len(self.src) or self.src[self.pos] != ")":
                    raise ParseError(f"第 {self.pos} 个字符处：括号未闭合")
                self.pos += 1                      # 跳过 ')'
                pending = 1
                continue

            # 方括号原子：[CH3]
            if ch == "[":
                close = self.src.find("]", self.pos)
                if close < 0:
                    raise ParseError("方括号未闭合")
                inner = self.src[self.pos + 1: close]
                m = _BRACKET_RE.match(inner)
                if not m:
                    raise ParseError(f"无法解析方括号内容：[{inner}]")
                sym = m.group(1)
                if sym not in ELEMENTS:
                    raise ParseError(f"未知元素：{sym}")
                # 氢数：'H' 后面没数字算 1 个（[CH] 表示 1 个氢）
                h_group, h_bare = m.group(2), m.group(3)
                if h_group is not None:
                    h_count = int(h_group) if h_group else 1
                elif h_bare:
                    h_count = int(h_bare)
                else:
                    h_count = 0
                idx = ctx.add_atom(sym, h_count)
                if current >= 0:
                    ctx.connect(current, idx, pending)
                current = idx
                pending = 1
                self.pos = close + 1
                continue

            # 键级符号
            if ch in _BOND_CHAR:
                pending = _BOND_CHAR[ch]
                self.pos += 1
                continue

            # 成环闭锁
            if ch.isdigit():
                num = ch
                self.pos += 1
                if self.pos < len(self.src) and self.src[self.pos].isdigit():
                    num += self.src[self.pos]
                    self.pos += 1
                if current < 0:
                    raise ParseError("环闭锁没有前导原子")
                if num in ctx.rings:
                    ctx.connect(ctx.rings[num], current, pending)
                    del ctx.rings[num]
                else:
                    ctx.rings[num] = current
                pending = 1
                continue

            # 元素符号
            if ch.isalpha() and ch.isupper():
                sym = ch
                self.pos += 1
                if self.pos < len(self.src) and self.src[self.pos].islower():
                    sym += self.src[self.pos]
                    self.pos += 1
                if sym not in ELEMENTS:
                    raise ParseError(f"未知元素：{sym}")
                idx = ctx.add_atom(sym, None)
                if current >= 0:
                    ctx.connect(current, idx, pending)
                current = idx
                pending = 1
                continue

            raise ParseError(f"第 {self.pos} 个字符处无法解析：{ch!r}")


def parse_spec(spec: str) -> Molecule:
    """把结构式解析成分子图。

    解析出的原子带 h_count（结构式里显式写出的氢数）；键级已合并。
    """
    if not spec:
        raise ParseError("结构式为空")
    ctx = _Parser(spec).parse_all()
    if not ctx.atoms:
        raise ParseError("没有解析出任何原子")
    return Molecule(atoms=ctx.atoms, bonds=ctx.bonds)


def parse_spec_safe(spec: str) -> tuple[Molecule | None, str | None]:
    """容错版本：返回 (分子, 错误信息)，不抛异常。"""
    try:
        return parse_spec(spec), None
    except ParseError as exc:
        return None, str(exc)
