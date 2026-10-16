# -*- mode: python ; coding: utf-8 -*-
#
# 入口用 SPECPATH 拼相对路径，**不写死绝对路径**：
# 早先这里写的是 C:/Users/18190/Desktop/...，换电脑或挪动目录后
# `pyinstaller 分子工坊.spec` 会直接报找不到脚本。
import os

a = Analysis(
    [os.path.join(SPECPATH, 'run.py')],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='分子工坊',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
