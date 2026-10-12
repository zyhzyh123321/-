"""一次性转换脚本：把网页版归档里的 molecules.js 转成 Python 数据模块。

为什么用脚本而不是手抄：46 条记录里有大段中文描述，手抄容易出错；
而结构式（spec）是化学正确性的关键，直接读原文转换最可靠。

解析方式：源文件里每条记录形如
    {
      id: 'methane', names: ['甲烷'], en: 'Methane',
      spec: 'C(-H)(-H)(-H)(-H)', formula: 'CH4',
      desc: '...',
      use: '...',
      note: '...'
    },
字段可能两个挤在一行，值里也可能带逗号和冒号，
所以按「已知键名」去切分，而不是按行或按逗号切。
"""
from __future__ import annotations

import ast
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "game_web_archive" / "js" / "data" / "molecules.js"
DST = ROOT / "molecule_workshop" / "data" / "molecules.py"

FIELDS = ("id", "names", "en", "spec", "formula", "desc", "use", "note")
_KEY_RE = re.compile(r"(?:^|[\s,{])(" + "|".join(FIELDS) + r")\s*:\s*", re.M)


def extract_records(text: str) -> list[dict]:
    body_start = text.index("var RAW = [")
    body_end = text.index("\n  ];", body_start)
    body = text[body_start:body_end]

    records: list[dict] = []
    # 每条记录用 `{ ... },` 切出来
    for block in re.findall(r"\{(.*?)\n\s*\},?", body, re.S):
        # 按已知键名切成 [(key, value_text), ...]
        hits = list(_KEY_RE.finditer(block))
        if not hits:
            continue
        rec: dict = {}
        for i, hit in enumerate(hits):
            key = hit.group(1)
            start = hit.end()
            end = hits[i + 1].start() if i + 1 < len(hits) else len(block)
            raw = block[start:end].strip()
            raw = raw.rstrip()
            if raw.endswith(","):
                raw = raw[:-1].strip()
            if not raw:
                continue
            try:
                rec[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
                    rec[key] = raw[1:-1]
                else:
                    rec[key] = raw
        if "id" in rec and "spec" in rec:
            records.append(rec)
    return records


HEADER = '''"""内置分子库（数据即代码）。

由 tools/convert_molecules.py 从网页版的 molecules.js 转换而来。
转换后请用 pytest 做分子式自校验 —— 结构式（spec）写错会立刻暴露。

字段说明：
    id        唯一标识
    names     中文名（可能有别名）
    en        英文名
    spec      结构式字符串，运行时解析成分子图（见 chem/parse_spec.py）
    formula   期望分子式（仅作自校验断言）
    desc      结构特征
    use       常见用途（可选）
    note      备注（可选）
"""

from __future__ import annotations

MOLECULES: list[dict] = [
'''

FOOTER = ''']


def by_id(molecule_id: str) -> dict | None:
    """按 id 查记录。"""
    for rec in MOLECULES:
        if rec["id"] == molecule_id:
            return rec
    return None


def names_of(molecule_id: str) -> list[str]:
    rec = by_id(molecule_id)
    return list(rec["names"]) if rec else []


def all_ids() -> list[str]:
    return [rec["id"] for rec in MOLECULES]
'''


def emit(records: list[dict]) -> str:
    out = io.StringIO()
    out.write(HEADER)
    for rec in records:
        out.write("    {\n")
        for key in FIELDS:
            if key not in rec:
                continue
            out.write(f'        "{key}": {rec[key]!r},\n')
        out.write("    },\n")
    out.write(FOOTER)
    return out.getvalue()


def main() -> int:
    if not SRC.exists():
        print(f"找不到源文件：{SRC}")
        return 1

    records = extract_records(SRC.read_text(encoding="utf-8"))
    print(f"从归档解析出 {len(records)} 条记录")

    incomplete = [r.get("id", "?") for r in records
                  if not r.get("names") or not r.get("formula") or not r.get("spec")]
    if incomplete:
        print("缺少必要字段：" + ", ".join(incomplete[:10]))
        return 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(emit(records), encoding="utf-8")
    print(f"已写出 {DST}（{len(records)} 条）")

    sys.path.insert(0, str(ROOT))
    from molecule_workshop.chem.identify import formula_of
    from molecule_workshop.chem.parse_spec import parse_spec

    ok = bad = 0
    for rec in records:
        try:
            mol = parse_spec(rec["spec"])
        except Exception as exc:                       # noqa: BLE001
            print(f"  x {rec['id']} 解析失败: {exc}")
            bad += 1
            continue
        got = formula_of(mol)
        if got != rec["formula"]:
            print(f"  x {rec['id']} 分子式不符：期望 {rec['formula']} 实得 {got}")
            bad += 1
        else:
            ok += 1
    print(f"分子式自校验：{ok} 条通过，{bad} 条失败")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
