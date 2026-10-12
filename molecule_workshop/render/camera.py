"""相机与投影。

用正交投影 —— 这是分子视图的教科书惯例，而且让整条管线只剩「点积 + 排序」。

坐标约定：
    编辑平面用 (x, y) 世界坐标；立起来后原子有 pos3。
    相机用球坐标（方位角 azimuth / 仰角 elevation），
    build 模式固定为正视（az=0, el=0），屏幕 x 直接对应世界 x、屏幕 y 对应 -世界 y，
    这样「鼠标拖到哪，原子就到哪」，手感才跟手。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..chem.constants import VIEW_BASE_SCALE, VIEW_MAX_SCALE, VIEW_MIN_SCALE
from ..chem.vector import Vec3


@dataclass
class Camera:
    azimuth: float = 0.0        # 绕 y 轴（弧度）
    elevation: float = 0.0      # 抬高（弧度）
    zoom: float = 1.0
    pan_x: float = 0.0          # 屏幕像素
    pan_y: float = 0.0
    scale: float = VIEW_BASE_SCALE
    width: float = 0.0
    height: float = 0.0

    # --------------------------------------------------------------
    # 基向量
    # --------------------------------------------------------------

    def basis(self) -> tuple[Vec3, Vec3, Vec3]:
        """返回 (right, up, forward)，都是单位向量。

        约定：视线方向 forward 从相机指向场景；right/up 用于投影。
        正视（az=el=0）时 forward = (0,0,-1)，right = (1,0,0)，up = (0,1,0)。
        """
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)

        forward = (-sa * ce, -se, -ca * ce)
        right = (ca, 0.0, -sa)
        # up = forward × right
        up = (
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        )
        right = _norm(right)
        up = _norm(up)
        forward = _norm(forward)
        return right, up, forward

    # --------------------------------------------------------------
    # 投影
    # --------------------------------------------------------------

    def screen_scale(self) -> float:
        return self.scale * self.zoom

    def project(self, p: Vec3) -> tuple[float, float, float]:
        """世界坐标 → (屏幕 x, 屏幕 y, 深度)。

        深度越大越远（用于画家算法排序：远的先画）。
        """
        right, up, forward = self.basis()
        s = self.screen_scale()
        sx = self.width / 2.0 + self.pan_x + (p[0] * right[0] + p[1] * right[1] + p[2] * right[2]) * s
        sy = self.height / 2.0 + self.pan_y - (p[0] * up[0] + p[1] * up[1] + p[2] * up[2]) * s
        depth = p[0] * forward[0] + p[1] * forward[1] + p[2] * forward[2]
        return sx, sy, depth

    def project_many(self, points: list[Vec3]) -> list[tuple[float, float, float]]:
        right, up, forward = self.basis()
        s = self.screen_scale()
        cx = self.width / 2.0 + self.pan_x
        cy = self.height / 2.0 + self.pan_y
        out = []
        for p in points:
            out.append((
                cx + (p[0] * right[0] + p[1] * right[1] + p[2] * right[2]) * s,
                cy - (p[0] * up[0] + p[1] * up[1] + p[2] * up[2]) * s,
                p[0] * forward[0] + p[1] * forward[1] + p[2] * forward[2],
            ))
        return out

    def screen_to_plane(self, sx: float, sy: float) -> tuple[float, float]:
        """屏幕坐标 → 编辑平面（z=0）的世界坐标。

        做法：屏幕点对应一条平行于 forward 的射线，
        与 z=0 平面求交。正视时退化为「平移 + 缩放」。
        """
        right, up, forward = self.basis()
        s = self.screen_scale()
        a = (sx - self.width / 2.0 - self.pan_x) / s      # right 分量
        c = -(sy - self.height / 2.0 - self.pan_y) / s    # up 分量

        # p = a·right + c·up + t·forward，要求 p.z = 0
        pz_const = a * right[2] + c * up[2]
        denom = forward[2]
        t = (-pz_const / denom) if abs(denom) > 1e-6 else 0.0
        return (
            a * right[0] + c * up[0] + t * forward[0],
            a * right[1] + c * up[1] + t * forward[1],
        )

    # --------------------------------------------------------------
    # 操作
    # --------------------------------------------------------------

    def orbit(self, dx_px: float, dy_px: float) -> None:
        """按屏幕位移旋转视角（3D 模式用）。"""
        self.azimuth += dx_px * 0.008
        self.elevation = _clamp(self.elevation + dy_px * 0.008, -1.45, 1.45)

    def zoom_by(self, factor: float) -> None:
        self.zoom = _clamp(self.zoom * factor, VIEW_MIN_SCALE, VIEW_MAX_SCALE)

    def zoom_at(self, factor: float, sx: float, sy: float) -> None:
        """围绕屏幕上的某个点缩放（滚轮缩放时更自然）。"""
        before = self.screen_to_plane(sx, sy)
        self.zoom_by(factor)
        after = self.screen_to_plane(sx, sy)
        s = self.screen_scale()
        self.pan_x += (after[0] - before[0]) * s
        self.pan_y -= (after[1] - before[1]) * s

    def pan(self, dx_px: float, dy_px: float) -> None:
        self.pan_x += dx_px
        self.pan_y += dy_px

    def reset(self, top_down: bool) -> None:
        self.azimuth = 0.0
        self.elevation = 0.0 if top_down else 0.32
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0

    def fit(self, radius: float, margin: float) -> None:
        """按分子半径自动缩放到合适大小（立起来后调用）。"""
        if radius <= 1e-6:
            return
        target = min(self.width, self.height) / (2.0 * radius * margin)
        self.zoom = _clamp(target / self.scale, VIEW_MIN_SCALE, VIEW_MAX_SCALE)


def _norm(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
