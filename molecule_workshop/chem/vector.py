"""数学、向量与颜色工具。

向量统一用 tuple[float, ...] 表示 —— 不可变、可放心当字典键，且便于调试打印。
（从网页版的 js/utils.js 对齐移植；去掉 DOM 相关部分。）
"""
from __future__ import annotations

import colorsys
import math
from typing import Iterable, Sequence

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]

# ----------------------------------------------------------------------------
# 通用
# ----------------------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - pow(-2.0 * t + 2.0, 3) / 2.0


# ----------------------------------------------------------------------------
# 二维
# ----------------------------------------------------------------------------

def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def normalize2(x: float, y: float) -> Vec2:
    n = math.hypot(x, y)
    if n < 1e-9:
        return (1.0, 0.0)
    return (x / n, y / n)


def rotate2(p: Vec2, angle: float) -> Vec2:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


# ----------------------------------------------------------------------------
# 三维
# ----------------------------------------------------------------------------

def v3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v3_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v3_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v3_len(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v3_dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def v3_normalize(a: Vec3) -> Vec3:
    n = v3_len(a)
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def v3_project_perp(a: Vec3, direction: Vec3) -> Vec3:
    """把 a 投影到与 direction 垂直的平面并归一化（施密特正交化）。"""
    d = v3_dot(a, direction)
    r = v3_sub(a, v3_mul(direction, d))
    if v3_len(r) < 1e-7:
        ref = (1.0, 0.0, 0.0) if abs(direction[0]) < 0.9 else (0.0, 1.0, 0.0)
        d2 = v3_dot(ref, direction)
        r = v3_sub(ref, v3_mul(direction, d2))
    return v3_normalize(r)


def v3_rotate_axis(v: Vec3, axis: Vec3, angle: float) -> Vec3:
    """围绕任意轴旋转向量（罗德里格斯公式）。"""
    c, s = math.cos(angle), math.sin(angle)
    k = v3_normalize(axis)
    kv = v3_cross(k, v)
    kd = v3_dot(k, v)
    return (
        v[0] * c + kv[0] * s + k[0] * kd * (1 - c),
        v[1] * c + kv[1] * s + k[1] * kd * (1 - c),
        v[2] * c + kv[2] * s + k[2] * kd * (1 - c),
    )


def v3_angle_between(a: Vec3, b: Vec3) -> float:
    """两个方向的夹角（弧度）。"""
    return math.acos(clamp(v3_dot(a, b), -1.0, 1.0))


def v3_basis_from(direction: Vec3) -> tuple[Vec3, Vec3]:
    """由给定方向构造一组正交基，用于在圆锥面上铺开方向。"""
    ref = (1.0, 0.0, 0.0) if abs(direction[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = v3_project_perp(ref, direction)
    e2 = v3_normalize(v3_cross(direction, e1))
    return e1, e2


# ----------------------------------------------------------------------------
# 颜色
# ----------------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    n = int(h, 16)
    return ((n >> 16) & 255, (n >> 8) & 255, n & 255)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(clamp(round(r), 0, 255)), int(clamp(round(g), 0, 255)), int(clamp(round(b), 0, 255))
    )


def shade(hex_color: str, amount: float) -> str:
    """明暗调整：amount > 0 变亮，< 0 变暗（-1..1）。"""
    r, g, b = hex_to_rgb(hex_color)
    target = 255.0 if amount > 0 else 0.0
    p = abs(amount)
    return rgb_to_hex(r + (target - r) * p, g + (target - g) * p, b + (target - b) * p)


def rgba(hex_color: str, alpha: float) -> tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, int(clamp(round(alpha * 255), 0, 255)))


def relative_luminance(hex_color: str) -> float:
    r, g, b = hex_to_rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def contrast_text(hex_color: str) -> str:
    """在给定底色上挑一个可读的文字色（深底用白字，浅底用黑字）。"""
    return "#0b1016" if relative_luminance(hex_color) > 0.6 else "#f2f7fa"


def mix_hex(a: str, b: str, t: float) -> str:
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return rgb_to_hex(ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t)


# ----------------------------------------------------------------------------
# 其它
# ----------------------------------------------------------------------------

def hill_order(symbols: Iterable[str]) -> list[str]:
    """Hill 排序：C 第一、H 第二，其余按字母序。"""
    def key(s: str):
        if s == "C":
            return (0, "")
        if s == "H":
            return (1, "")
        return (2, s)
    return sorted(symbols, key=key)


def cell_color_for_fraction(t: float, base_hex: str) -> str:
    """按比例给元素组成条上色（用基色的明暗变化）。"""
    t = clamp(t, 0.0, 1.0)
    return shade(base_hex, -0.25 + 0.5 * t)


def hsv_shift(hex_color: str, dh: float = 0.0, ds: float = 0.0, dv: float = 0.0) -> str:
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h = (h + dh) % 1.0
    s = clamp(s + ds, 0.0, 1.0)
    v = clamp(v + dv, 0.0, 1.0)
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
    return rgb_to_hex(nr * 255, ng * 255, nb * 255)


def point_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """点到线段的距离（命中测试用）。"""
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return dist2(px, py, x1, y1)
    t = clamp(((px - x1) * dx + (py - y1) * dy) / l2, 0.0, 1.0)
    return dist2(px, py, x1 + t * dx, y1 + t * dy)


def format_mass(mass: float | None) -> str:
    return "—" if mass is None else f"{mass:.2f}"
