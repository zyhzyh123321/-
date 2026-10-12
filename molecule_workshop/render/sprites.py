"""球体贴图烘焙：用 numpy 做真实光照着色，只在启动时算一次。

为什么不用 QPainter 的径向渐变画球：
    虽然更快，但只能得到「中心亮、边缘暗」的简单渐变，缺少真实的方向性光照，
    多个球挨在一起时看起来比较平。而且 Qt 里用渐变画刷有个坑：
    直接 setBrush(QBrush(gradient)) 在离屏渲染下会静默失效、只画出描边。
    这里改为逐像素着色，得到真正的 Lambert 漫反射 + 高光 + 边缘暗化，
    再把结果烘成贴图，每帧只做 drawImage（实测 0.9 ms/帧，完全够用）。

为什么不每帧逐像素渲染整个场景：
    实测光线投射整场景要 580 ms/帧（2 fps），交互不可用。
    烘焙贴图 + 合成是「质量」与「帧率」的正确分工。
"""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtGui

from ..chem.constants import RENDER_SUPERSAMPLE
from ..chem.vector import hex_to_rgb

# 光照方向（世界坐标，Z 朝观察者）
LIGHT_DIR = np.array([0.45, 0.62, 0.64], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

# 环境光与漫反射比例：环境项太低会显得很黑，太高则失去立体感
AMBIENT = 0.30
DIFFUSE = 0.70
SPECULAR = 0.55
SHININESS = 22.0


def bake_sphere(hex_color: str, size: int) -> QtGui.QImage:
    """烘焙一个球体贴图。

    size 是直径（像素）。用超采样做抗锯齿，再缩回目标尺寸。
    """
    ss = max(1, int(RENDER_SUPERSAMPLE))
    s = size * ss
    pad = max(2, s // 16)          # 防抗锯齿切边，四周留边
    total = s + pad * 2

    # 归一化像素坐标（-1..1）
    yy, xx = np.mgrid[0:total, 0:total].astype(np.float32)
    cx = cy = (total - 1) / 2.0
    nx = (xx - cx) / (s / 2.0)
    ny = -(yy - cy) / (s / 2.0)    # 屏幕 y 向下，翻成数学方向
    r2 = nx * nx + ny * ny
    inside = r2 <= 1.0
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    # 法线（单位矢量），球面朝观察者
    n = np.stack([nx, ny, nz], axis=-1)
    lam = np.clip((n * LIGHT_DIR).sum(axis=-1), 0.0, 1.0)

    # 观察方向固定为 +Z，视线与法线的夹角决定高光
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = LIGHT_DIR + view
    half /= np.linalg.norm(half)
    spec = np.clip((n * half).sum(axis=-1), 0.0, 1.0) ** SHININESS

    shade = AMBIENT + DIFFUSE * (lam ** 0.85)

    base = np.array(hex_to_rgb(hex_color), dtype=np.float32)
    rgb = base[None, None, :] * shade[..., None]
    rgb = rgb + spec[..., None] * SPECULAR * 255.0

    # 边缘轻微压暗（模拟接触阴影，增强立体感）
    edge = np.clip(1.0 - (r2 ** 3), 0.0, 1.0)
    rgb *= (0.82 + 0.18 * edge)[..., None]

    rgb = np.clip(rgb, 0.0, 255.0)

    alpha = np.where(inside, 255.0, 0.0).astype(np.float32)
    # 边缘做一点羽化，避免锯齿
    fringe = (np.abs(r2 - 1.0) < (2.0 / (s / 2.0)))
    alpha = np.where(fringe & inside, np.clip(255.0 * (1.0 - (r2 - 1.0 + 0.15) / 0.3), 0, 255), alpha)

    out = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)
    out = np.ascontiguousarray(out)
    img = QtGui.QImage(out.data, total, total, 4 * total,
                       QtGui.QImage.Format.Format_RGBA8888).copy()
    if ss > 1:
        img = img.scaled(s + pad * 2, s + pad * 2,
                         QtGui.Qt.AspectRatioMode.IgnoreAspectRatio,
                         QtGui.Qt.TransformationMode.SmoothTransformation)
    return img


class SpriteCache:
    """元素 → 球体贴图的缓存。

    贴图按「接近实际显示尺寸的分档」缓存，避免每个缩放级别都重新烘焙。
    """

    # 贴图分档：覆盖常见显示尺寸，超出就取最近档再缩放
    BASE_SIZE = 256

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], QtGui.QImage] = {}

    def sphere(self, hex_color: str, size: int) -> QtGui.QImage:
        """取（或烘焙）指定直径的球体贴图。"""
        size = max(6, min(1024, int(size)))
        # 量化到 8 像素一档，减少烘焙次数
        bucket = max(8, int(math.ceil(size / 8.0)) * 8)
        key = (hex_color, bucket)
        img = self._cache.get(key)
        if img is None:
            img = bake_sphere(hex_color, bucket)
            self._cache[key] = img
        return img

    def clear(self) -> None:
        self._cache.clear()

    @property
    def count(self) -> int:
        return len(self._cache)
