import os
import sys
import customtkinter

ctk_dir = os.path.dirname(customtkinter.__file__)
block_cipher = None

_datas = [
    ("config.yaml", "."),
    ("src", "src"),
    (ctk_dir, "customtkinter"),
    ("src/ui/fa-solid-900.ttf", "src/ui"),
    ("src/ui/app_icon.png", "src/ui"),
]

_hidden = [
    "customtkinter",
    "PIL._tkinter_finder",
    "pyttsx3.drivers",
    "pyttsx3.drivers.sapi5",
    "pyttsx3.drivers.nsss",
    "pyttsx3.drivers.espeak",
    "anthropic",
    "mss",
    "mss.windows",
    "mss.darwin",
    "cv2",
    "numpy",
    "yaml",
    "soundcard",
    "scipy",
    "sklearn",
]

a = Analysis(
    ["coach_app.py"],
    pathex=["."],
    binaries=[],
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == "darwin":
    # macOS: onedir mode so .app bundle is valid
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name="ValorantCoach",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        name="ValorantCoach",
    )
    app = BUNDLE(
        coll,
        name="ValorantCoach.app",
        icon="src/ui/app_icon.png",
        bundle_identifier="com.valorant-minimap-coach",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSUIElement": True,
        },
    )
else:
    # Windows / Linux: single-file executable
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
        icon="src/ui/app_icon.png",
    )
