import os
import customtkinter

ctk_dir = os.path.dirname(customtkinter.__file__)
block_cipher = None

a = Analysis(
    ["coach_app.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("config.yaml", "."),
        ("src", "src"),
        (ctk_dir, "customtkinter"),
    ],
    hiddenimports=[
        "customtkinter",
        "PIL._tkinter_finder",
        "pyttsx3.drivers",
        "pyttsx3.drivers.sapi5",
        "pyttsx3.drivers.nsss",
        "pyttsx3.drivers.espeak",
        "anthropic",
        "mss",
        "mss.windows",
        "cv2",
        "numpy",
        "yaml",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ValorantCoach",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    onefile=True,
)
