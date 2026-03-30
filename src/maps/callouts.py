from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Zone:
    name: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


CALLOUT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "EN": {
        "one":   "Enemy at {loc}",
        "two":   "Two enemies at {loc}",
        "multi": "{n} enemies at {loc}",
        "split": "{n} enemies - {locs}",
        "stack": "Stack! {n} enemies at {loc}",
    },
    "PL": {
        "one":   "Wróg przy {loc}",
        "two":   "Dwóch wrogów przy {loc}",
        "multi": "{n} wrogów przy {loc}",
        "split": "{n} wrogów - {locs}",
        "stack": "Stos! {n} wrogów przy {loc}",
    },
    "DE": {
        "one":   "Feind bei {loc}",
        "two":   "Zwei Feinde bei {loc}",
        "multi": "{n} Feinde bei {loc}",
        "split": "{n} Feinde - {locs}",
        "stack": "Stack! {n} Feinde bei {loc}",
    },
    "FR": {
        "one":   "Ennemi à {loc}",
        "two":   "Deux ennemis à {loc}",
        "multi": "{n} ennemis à {loc}",
        "split": "{n} ennemis - {locs}",
        "stack": "Stack! {n} ennemis à {loc}",
    },
    "ES": {
        "one":   "Enemigo en {loc}",
        "two":   "Dos enemigos en {loc}",
        "multi": "{n} enemigos en {loc}",
        "split": "{n} enemigos - {locs}",
        "stack": "Stack! {n} enemigos en {loc}",
    },
    "RU": {
        "one":   "Враг на {loc}",
        "two":   "Двое на {loc}",
        "multi": "{n} врагов на {loc}",
        "split": "{n} врагов - {locs}",
        "stack": "Стак! {n} врагов на {loc}",
    },
}

MAP_ZONES: Dict[str, List[Zone]] = {

    # -------------------------------------------------------------------------
    # ASCENT  (CT top, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "ascent": [
        Zone("CT spawn",       (0.40, 0.60), (0.04, 0.12)),
        Zone("D hall",         (0.39, 0.56), (0.12, 0.20)),
        Zone("D plat",         (0.39, 0.55), (0.18, 0.24)),
        Zone("back site B",    (0.06, 0.18), (0.14, 0.22)),
        Zone("Button",         (0.06, 0.16), (0.18, 0.25)),
        Zone("Double",         (0.15, 0.25), (0.19, 0.27)),
        Zone("B stairs",       (0.25, 0.36), (0.15, 0.24)),
        Zone("Triple",         (0.11, 0.22), (0.27, 0.33)),
        Zone("Workshop",       (0.26, 0.36), (0.23, 0.33)),
        Zone("B site",         (0.07, 0.30), (0.22, 0.34)),
        Zone("B lane",         (0.07, 0.27), (0.34, 0.40)),
        Zone("B orb",          (0.18, 0.30), (0.37, 0.44)),
        Zone("B main",         (0.20, 0.38), (0.38, 0.47)),
        Zone("B cubby",        (0.22, 0.38), (0.43, 0.50)),
        Zone("Fish market",    (0.12, 0.28), (0.47, 0.58)),
        Zone("Ticket booth",   (0.14, 0.27), (0.53, 0.62)),
        Zone("B lobby",        (0.14, 0.37), (0.53, 0.63)),
        Zone("Octopus",        (0.14, 0.27), (0.60, 0.68)),
        Zone("B alley",        (0.18, 0.36), (0.64, 0.76)),
        Zone("Church",         (0.29, 0.48), (0.74, 0.86)),
        Zone("Door plat",      (0.33, 0.44), (0.25, 0.35)),
        Zone("Logs",           (0.33, 0.44), (0.33, 0.42)),
        Zone("Boat",           (0.33, 0.44), (0.40, 0.46)),
        Zone("Market",         (0.36, 0.49), (0.26, 0.36)),
        Zone("D conn",         (0.47, 0.59), (0.26, 0.38)),
        Zone("Pizza",          (0.53, 0.63), (0.38, 0.50)),
        Zone("Bottom mid",     (0.42, 0.60), (0.40, 0.50)),
        Zone("Bench",          (0.40, 0.54), (0.44, 0.52)),
        Zone("Fountain",       (0.42, 0.58), (0.52, 0.64)),
        Zone("Library",        (0.52, 0.64), (0.60, 0.70)),
        Zone("Catwalk",        (0.55, 0.67), (0.48, 0.62)),
        Zone("Top mid",        (0.52, 0.65), (0.64, 0.74)),
        Zone("A alley",        (0.52, 0.68), (0.72, 0.80)),
        Zone("Bar",            (0.55, 0.72), (0.76, 0.84)),
        Zone("Heaven ramp",    (0.62, 0.74), (0.10, 0.18)),
        Zone("Heaven hall",    (0.68, 0.84), (0.14, 0.22)),
        Zone("White room",     (0.66, 0.78), (0.22, 0.30)),
        Zone("Heaven pillar",  (0.87, 0.95), (0.22, 0.30)),
        Zone("Heaven",         (0.78, 0.90), (0.22, 0.30)),
        Zone("JumpUp",         (0.74, 0.84), (0.28, 0.34)),
        Zone("Hell",           (0.80, 0.93), (0.30, 0.38)),
        Zone("Generator",      (0.78, 0.88), (0.27, 0.37)),
        Zone("Green box",      (0.88, 0.96), (0.28, 0.36)),
        Zone("A site",         (0.78, 0.97), (0.26, 0.39)),
        Zone("Site pillar",    (0.86, 0.96), (0.34, 0.40)),
        Zone("Glass window",   (0.62, 0.74), (0.30, 0.38)),
        Zone("Garden",         (0.68, 0.80), (0.34, 0.42)),
        Zone("Tree",           (0.65, 0.78), (0.38, 0.46)),
        Zone("Door",           (0.78, 0.90), (0.40, 0.48)),
        Zone("Bricks",         (0.88, 0.96), (0.40, 0.48)),
        Zone("Arch",           (0.86, 0.96), (0.44, 0.52)),
        Zone("A cubby",        (0.65, 0.78), (0.44, 0.52)),
        Zone("A orb",          (0.67, 0.78), (0.48, 0.56)),
        Zone("A long",         (0.73, 0.90), (0.47, 0.56)),
        Zone("Wine",           (0.87, 0.96), (0.47, 0.56)),
        Zone("Deli",           (0.71, 0.84), (0.52, 0.62)),
        Zone("A lobby",        (0.63, 0.84), (0.60, 0.70)),
        Zone("Gelato",         (0.67, 0.80), (0.68, 0.76)),
        Zone("T spawn",        (0.39, 0.59), (0.85, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # BIND  (CT top, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "bind": [
        Zone("CT spawn",           (0.40, 0.60), (0.04, 0.14)),
        Zone("B Hall",             (0.06, 0.24), (0.14, 0.26)),
        Zone("B site",             (0.06, 0.28), (0.24, 0.44)),
        Zone("B Elbow",            (0.02, 0.12), (0.34, 0.48)),
        Zone("B Garden",           (0.12, 0.28), (0.36, 0.54)),
        Zone("B Long",             (0.04, 0.20), (0.44, 0.62)),
        Zone("B Window",           (0.22, 0.36), (0.40, 0.54)),
        Zone("B Exit",             (0.08, 0.22), (0.52, 0.66)),
        Zone("B Short",            (0.22, 0.38), (0.46, 0.60)),
        Zone("B Link",             (0.30, 0.46), (0.32, 0.48)),
        Zone("B Lobby",            (0.08, 0.26), (0.62, 0.76)),
        Zone("Mid",                (0.38, 0.60), (0.44, 0.64)),
        Zone("Attacker Side Cave", (0.42, 0.62), (0.68, 0.82)),
        Zone("A Tower",            (0.76, 0.90), (0.14, 0.28)),
        Zone("A site",             (0.70, 0.94), (0.24, 0.48)),
        Zone("A Lamps",            (0.72, 0.86), (0.30, 0.46)),
        Zone("A Teleporter",       (0.60, 0.76), (0.38, 0.56)),
        Zone("A Cubby",            (0.78, 0.92), (0.44, 0.58)),
        Zone("A Short",            (0.64, 0.80), (0.48, 0.62)),
        Zone("A Link",             (0.52, 0.68), (0.54, 0.68)),
        Zone("A Bath",             (0.86, 0.98), (0.48, 0.64)),
        Zone("A Exit",             (0.88, 0.98), (0.58, 0.72)),
        Zone("A Lobby",            (0.74, 0.94), (0.58, 0.74)),
        Zone("T spawn",            (0.38, 0.60), (0.84, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # BREEZE  (CT top, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "breeze": [
        Zone("CT spawn",       (0.40, 0.62), (0.04, 0.12)),
        Zone("Arches",         (0.10, 0.40), (0.05, 0.14)),
        Zone("Nest",           (0.36, 0.50), (0.18, 0.28)),
        Zone("CT hall",        (0.48, 0.70), (0.18, 0.27)),
        Zone("Bridge",         (0.70, 0.84), (0.18, 0.26)),
        Zone("Backsite B",     (0.04, 0.14), (0.22, 0.30)),
        Zone("B pillar",       (0.08, 0.18), (0.28, 0.38)),
        Zone("Default B",      (0.08, 0.20), (0.37, 0.44)),
        Zone("B site",         (0.04, 0.20), (0.23, 0.44)),
        Zone("Alley",          (0.02, 0.10), (0.22, 0.48)),
        Zone("Ladder",         (0.04, 0.14), (0.42, 0.50)),
        Zone("B pocket",       (0.15, 0.25), (0.38, 0.46)),
        Zone("Wall",           (0.20, 0.32), (0.22, 0.30)),
        Zone("Tunnel",         (0.22, 0.38), (0.28, 0.36)),
        Zone("Grass",          (0.30, 0.44), (0.32, 0.40)),
        Zone("Roti",           (0.40, 0.52), (0.32, 0.40)),
        Zone("Metal doors",    (0.52, 0.64), (0.28, 0.36)),
        Zone("Stairs",         (0.62, 0.72), (0.28, 0.36)),
        Zone("Toilet",         (0.44, 0.54), (0.36, 0.44)),
        Zone("Wood doors",     (0.54, 0.66), (0.36, 0.44)),
        Zone("Tetris",         (0.62, 0.74), (0.35, 0.43)),
        Zone("Triple",         (0.72, 0.82), (0.32, 0.40)),
        Zone("Ramp",           (0.58, 0.70), (0.40, 0.50)),
        Zone("Halls",          (0.58, 0.68), (0.44, 0.54)),
        Zone("Ledge",          (0.54, 0.64), (0.48, 0.56)),
        Zone("Strap",          (0.88, 0.98), (0.28, 0.38)),
        Zone("Ninja",          (0.94, 1.00), (0.38, 0.50)),
        Zone("Inner back A",   (0.74, 0.84), (0.40, 0.48)),
        Zone("Outer back A",   (0.83, 0.92), (0.40, 0.48)),
        Zone("Left tit",       (0.72, 0.82), (0.46, 0.54)),
        Zone("Right tit",      (0.82, 0.92), (0.46, 0.54)),
        Zone("Inner front A",  (0.72, 0.82), (0.52, 0.58)),
        Zone("Default A",      (0.82, 0.92), (0.52, 0.58)),
        Zone("A site",         (0.72, 0.96), (0.38, 0.58)),
        Zone("A cubby",        (0.72, 0.84), (0.58, 0.66)),
        Zone("Shop",           (0.62, 0.76), (0.62, 0.72)),
        Zone("Ropes",          (0.56, 0.68), (0.68, 0.76)),
        Zone("A lobby",        (0.58, 0.76), (0.74, 0.82)),
        Zone("Banana",         (0.74, 0.88), (0.76, 0.86)),
        Zone("B main",         (0.04, 0.22), (0.44, 0.56)),
        Zone("Elbow",          (0.22, 0.34), (0.44, 0.52)),
        Zone("Lighthouse",     (0.40, 0.56), (0.42, 0.52)),
        Zone("Vents",          (0.56, 0.66), (0.55, 0.63)),
        Zone("Double",         (0.56, 0.68), (0.56, 0.64)),
        Zone("Gravel",         (0.38, 0.52), (0.52, 0.60)),
        Zone("Bot mid",        (0.40, 0.58), (0.61, 0.72)),
        Zone("B lobby",        (0.18, 0.36), (0.53, 0.62)),
        Zone("Cannon",         (0.22, 0.36), (0.60, 0.68)),
        Zone("Snake arches",   (0.18, 0.32), (0.62, 0.72)),
        Zone("Courtyard",      (0.04, 0.16), (0.56, 0.66)),
        Zone("2nd street",     (0.16, 0.38), (0.56, 0.64)),
        Zone("1st street",     (0.16, 0.34), (0.62, 0.70)),
        Zone("Sand",           (0.28, 0.56), (0.70, 0.80)),
        Zone("Snake",          (0.18, 0.34), (0.72, 0.80)),
        Zone("Cave",           (0.46, 0.64), (0.78, 0.86)),
        Zone("T spawn",        (0.38, 0.60), (0.86, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # ICEBOX  (CT top, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "icebox": [
        Zone("CT spawn",       (0.42, 0.60), (0.04, 0.14)),
        Zone("B Garage",       (0.38, 0.56), (0.12, 0.24)),
        Zone("B Cubby",        (0.04, 0.16), (0.28, 0.40)),
        Zone("B Yellow",       (0.02, 0.12), (0.36, 0.48)),
        Zone("B Green",        (0.22, 0.40), (0.24, 0.36)),
        Zone("B Tube",         (0.24, 0.40), (0.34, 0.46)),
        Zone("B Orange",       (0.16, 0.32), (0.40, 0.52)),
        Zone("B Snow Pile",    (0.24, 0.40), (0.50, 0.62)),
        Zone("B site",         (0.12, 0.36), (0.36, 0.64)),
        Zone("B Hall",         (0.14, 0.32), (0.54, 0.66)),
        Zone("B Kitchen",      (0.24, 0.42), (0.60, 0.70)),
        Zone("B Fence",        (0.06, 0.20), (0.60, 0.72)),
        Zone("B Back",         (0.20, 0.38), (0.72, 0.82)),
        Zone("B Snowman",      (0.04, 0.18), (0.70, 0.82)),
        Zone("B Hut",          (0.40, 0.56), (0.68, 0.80)),
        Zone("Mid Blue",       (0.38, 0.58), (0.24, 0.36)),
        Zone("Mid Pallet",     (0.38, 0.58), (0.44, 0.56)),
        Zone("Mid Boiler",     (0.42, 0.60), (0.56, 0.68)),
        Zone("A Belt",         (0.86, 0.98), (0.28, 0.40)),
        Zone("A Nest",         (0.80, 0.94), (0.38, 0.50)),
        Zone("A Pipes",        (0.62, 0.78), (0.42, 0.52)),
        Zone("A Screen",       (0.62, 0.78), (0.56, 0.66)),
        Zone("A site",         (0.66, 0.96), (0.36, 0.66)),
        Zone("A Rafters",      (0.78, 0.94), (0.60, 0.74)),
        Zone("T spawn",        (0.42, 0.60), (0.86, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # SPLIT  (CT top, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "split": [
        Zone("CT spawn",       (0.42, 0.60), (0.04, 0.14)),
        Zone("B Lobby",        (0.06, 0.24), (0.08, 0.20)),
        Zone("B Main",         (0.08, 0.28), (0.22, 0.38)),
        Zone("B Link",         (0.22, 0.40), (0.34, 0.46)),
        Zone("B Rafters",      (0.22, 0.38), (0.44, 0.56)),
        Zone("B Tower",        (0.14, 0.30), (0.44, 0.58)),
        Zone("B site",         (0.08, 0.32), (0.42, 0.64)),
        Zone("B Stairs",       (0.26, 0.44), (0.50, 0.62)),
        Zone("B Back",         (0.04, 0.18), (0.56, 0.68)),
        Zone("B Alley",        (0.08, 0.26), (0.66, 0.78)),
        Zone("Mid Top",        (0.40, 0.58), (0.30, 0.42)),
        Zone("Mid Vent",       (0.42, 0.58), (0.42, 0.54)),
        Zone("Mid Mail",       (0.36, 0.52), (0.46, 0.58)),
        Zone("Mid Bottom",     (0.40, 0.58), (0.56, 0.68)),
        Zone("A Lobby",        (0.78, 0.96), (0.08, 0.20)),
        Zone("A Sewer",        (0.64, 0.82), (0.14, 0.26)),
        Zone("A Main",         (0.76, 0.96), (0.20, 0.36)),
        Zone("A Ramps",        (0.66, 0.84), (0.28, 0.40)),
        Zone("A Rafters",      (0.70, 0.86), (0.36, 0.48)),
        Zone("A site",         (0.72, 0.94), (0.38, 0.56)),
        Zone("A Back",         (0.82, 0.96), (0.48, 0.60)),
        Zone("A Tower",        (0.62, 0.78), (0.50, 0.62)),
        Zone("A Screens",      (0.80, 0.94), (0.62, 0.74)),
        Zone("T spawn",        (0.40, 0.60), (0.88, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # HAVEN  (CT top, T bottom; C left, B center, A right)
    # -------------------------------------------------------------------------
    "haven": [
        Zone("CT spawn",       (0.38, 0.62), (0.04, 0.14)),
        Zone("A Lobby",        (0.74, 0.92), (0.08, 0.22)),
        Zone("A Garden",       (0.88, 0.98), (0.14, 0.28)),
        Zone("A Long",         (0.86, 0.98), (0.26, 0.40)),
        Zone("A Short",        (0.72, 0.86), (0.26, 0.42)),
        Zone("A Link",         (0.62, 0.78), (0.40, 0.54)),
        Zone("A Tower",        (0.68, 0.82), (0.52, 0.66)),
        Zone("A site",         (0.74, 0.96), (0.32, 0.56)),
        Zone("B Pillars",      (0.42, 0.60), (0.14, 0.26)),
        Zone("B Main",         (0.40, 0.60), (0.26, 0.42)),
        Zone("B site",         (0.40, 0.62), (0.34, 0.58)),
        Zone("B Back",         (0.40, 0.60), (0.54, 0.66)),
        Zone("C Lobby",        (0.08, 0.28), (0.08, 0.22)),
        Zone("C Long",         (0.02, 0.16), (0.24, 0.40)),
        Zone("C Mound",        (0.28, 0.44), (0.18, 0.32)),
        Zone("C Door",         (0.18, 0.34), (0.28, 0.44)),
        Zone("C Cubby",        (0.12, 0.26), (0.34, 0.48)),
        Zone("C Short",        (0.28, 0.44), (0.40, 0.54)),
        Zone("C Window",       (0.16, 0.30), (0.40, 0.56)),
        Zone("C Link",         (0.26, 0.42), (0.50, 0.64)),
        Zone("C site",         (0.06, 0.30), (0.32, 0.58)),
        Zone("Mid Courtyard",  (0.36, 0.60), (0.46, 0.62)),
        Zone("Mid Window",     (0.40, 0.58), (0.22, 0.36)),
        Zone("T spawn",        (0.36, 0.62), (0.82, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # LOTUS  (T top, CT bottom; A left, B center, C right)
    # -------------------------------------------------------------------------
    "lotus": [
        Zone("T spawn",        (0.38, 0.62), (0.04, 0.14)),
        Zone("A Lobby",        (0.04, 0.22), (0.10, 0.22)),
        Zone("A Rubble",       (0.06, 0.22), (0.18, 0.30)),
        Zone("A Root",         (0.14, 0.30), (0.26, 0.38)),
        Zone("A Door",         (0.02, 0.16), (0.32, 0.46)),
        Zone("A Tree",         (0.02, 0.14), (0.42, 0.54)),
        Zone("A Main",         (0.12, 0.30), (0.36, 0.50)),
        Zone("A Link",         (0.20, 0.38), (0.44, 0.58)),
        Zone("A site",         (0.04, 0.28), (0.40, 0.66)),
        Zone("A Hut",          (0.04, 0.16), (0.56, 0.70)),
        Zone("A Stairs",       (0.10, 0.26), (0.58, 0.72)),
        Zone("A Drop",         (0.04, 0.16), (0.68, 0.80)),
        Zone("A Top",          (0.10, 0.24), (0.70, 0.82)),
        Zone("B Pillars",      (0.36, 0.58), (0.14, 0.28)),
        Zone("B Main",         (0.38, 0.58), (0.28, 0.44)),
        Zone("B site",         (0.36, 0.62), (0.30, 0.58)),
        Zone("B Upper",        (0.34, 0.52), (0.50, 0.64)),
        Zone("C Lobby",        (0.68, 0.88), (0.10, 0.22)),
        Zone("C Mound",        (0.74, 0.92), (0.18, 0.30)),
        Zone("C Bend",         (0.86, 0.98), (0.28, 0.44)),
        Zone("C Door",         (0.62, 0.80), (0.28, 0.42)),
        Zone("C Main",         (0.66, 0.84), (0.36, 0.50)),
        Zone("C Waterfall",    (0.64, 0.80), (0.44, 0.58)),
        Zone("C Link",         (0.58, 0.76), (0.52, 0.66)),
        Zone("C Hall",         (0.70, 0.90), (0.56, 0.70)),
        Zone("C Gravel",       (0.50, 0.70), (0.68, 0.80)),
        Zone("C site",         (0.70, 0.96), (0.36, 0.60)),
        Zone("CT spawn",       (0.38, 0.62), (0.84, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # SUNSET  (CT top, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "sunset": [
        Zone("CT spawn",       (0.40, 0.62), (0.04, 0.14)),
        Zone("B Boba",         (0.06, 0.26), (0.20, 0.36)),
        Zone("B site",         (0.06, 0.28), (0.22, 0.46)),
        Zone("B Market",       (0.22, 0.40), (0.44, 0.58)),
        Zone("B Main",         (0.04, 0.24), (0.52, 0.68)),
        Zone("B Lobby",        (0.08, 0.26), (0.66, 0.78)),
        Zone("Mid Top",        (0.34, 0.56), (0.24, 0.38)),
        Zone("Mid Courtyard",  (0.38, 0.62), (0.38, 0.56)),
        Zone("Mid Tiles",      (0.50, 0.68), (0.50, 0.64)),
        Zone("Mid Bottom",     (0.36, 0.58), (0.54, 0.68)),
        Zone("A Alley",        (0.84, 0.96), (0.12, 0.26)),
        Zone("A Link",         (0.66, 0.84), (0.24, 0.38)),
        Zone("A Elbow",        (0.88, 0.98), (0.36, 0.50)),
        Zone("A site",         (0.70, 0.96), (0.26, 0.52)),
        Zone("A Main",         (0.64, 0.88), (0.46, 0.64)),
        Zone("A Lobby",        (0.72, 0.92), (0.60, 0.74)),
        Zone("T spawn",        (0.38, 0.62), (0.86, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # ABYSS  (T top, CT bottom; A left, B right)
    # -------------------------------------------------------------------------
    "abyss": [
        Zone("T spawn",        (0.38, 0.62), (0.04, 0.14)),
        Zone("A Lobby",        (0.04, 0.22), (0.08, 0.22)),
        Zone("A Main",         (0.04, 0.22), (0.22, 0.38)),
        Zone("A Bridge",       (0.02, 0.12), (0.28, 0.44)),
        Zone("A Tower",        (0.14, 0.30), (0.26, 0.38)),
        Zone("A Vent",         (0.16, 0.32), (0.36, 0.48)),
        Zone("A Link",         (0.22, 0.38), (0.44, 0.56)),
        Zone("A site",         (0.04, 0.30), (0.34, 0.56)),
        Zone("A Security",     (0.08, 0.24), (0.54, 0.66)),
        Zone("A Secret",       (0.04, 0.16), (0.64, 0.76)),
        Zone("Mid Catwalk",    (0.34, 0.52), (0.16, 0.30)),
        Zone("Mid Top",        (0.38, 0.56), (0.28, 0.38)),
        Zone("Mid Library",    (0.36, 0.54), (0.38, 0.50)),
        Zone("Mid Bend",       (0.52, 0.68), (0.46, 0.58)),
        Zone("Mid Bottom",     (0.36, 0.54), (0.50, 0.62)),
        Zone("B Nest",         (0.72, 0.90), (0.12, 0.26)),
        Zone("B Main",         (0.62, 0.82), (0.18, 0.34)),
        Zone("B Danger",       (0.66, 0.84), (0.26, 0.40)),
        Zone("B Link",         (0.58, 0.74), (0.38, 0.50)),
        Zone("B Tower",        (0.72, 0.88), (0.42, 0.54)),
        Zone("B site",         (0.66, 0.92), (0.28, 0.50)),
        Zone("CT spawn",       (0.38, 0.62), (0.84, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # PEARL  (T top, CT bottom; A left, B right)
    # -------------------------------------------------------------------------
    "pearl": [
        Zone("T spawn",                (0.36, 0.62), (0.04, 0.14)),
        Zone("A Lobby",                (0.04, 0.22), (0.08, 0.22)),
        Zone("A Art",                  (0.06, 0.22), (0.14, 0.28)),
        Zone("A Main",                 (0.04, 0.22), (0.22, 0.38)),
        Zone("A Tower",                (0.14, 0.30), (0.24, 0.36)),
        Zone("A Vent",                 (0.16, 0.30), (0.34, 0.46)),
        Zone("A Link",                 (0.22, 0.38), (0.42, 0.56)),
        Zone("A Dugout",               (0.02, 0.14), (0.46, 0.60)),
        Zone("A site",                 (0.04, 0.28), (0.38, 0.60)),
        Zone("A Rafters",              (0.04, 0.18), (0.60, 0.72)),
        Zone("A Garden",               (0.08, 0.24), (0.62, 0.74)),
        Zone("A Window",               (0.18, 0.32), (0.60, 0.72)),
        Zone("A Flowers",              (0.06, 0.20), (0.68, 0.80)),
        Zone("A Secret",               (0.04, 0.16), (0.74, 0.84)),
        Zone("Mid Top",                (0.34, 0.54), (0.14, 0.26)),
        Zone("Mid Catwalk",            (0.38, 0.58), (0.22, 0.34)),
        Zone("Mid Shops",              (0.40, 0.60), (0.14, 0.28)),
        Zone("Mid Courtyard",          (0.38, 0.60), (0.30, 0.46)),
        Zone("Mid Link",               (0.50, 0.68), (0.36, 0.50)),
        Zone("Mid Cubby",              (0.38, 0.56), (0.44, 0.56)),
        Zone("Mid Door",               (0.40, 0.58), (0.44, 0.58)),
        Zone("Mid Pizza",              (0.40, 0.58), (0.58, 0.70)),
        Zone("Mid Market",             (0.44, 0.62), (0.60, 0.72)),
        Zone("Mid Bottom",             (0.36, 0.56), (0.54, 0.66)),
        Zone("Mid Connector",          (0.40, 0.58), (0.66, 0.78)),
        Zone("Defender Side Water",    (0.26, 0.48), (0.70, 0.82)),
        Zone("Defender Side Records",  (0.42, 0.64), (0.72, 0.84)),
        Zone("B Club",                 (0.68, 0.86), (0.08, 0.20)),
        Zone("B Lobby",                (0.76, 0.94), (0.10, 0.22)),
        Zone("B Ramp",                 (0.86, 0.98), (0.14, 0.26)),
        Zone("B Main",                 (0.70, 0.90), (0.22, 0.38)),
        Zone("B Plaza",                (0.60, 0.78), (0.22, 0.36)),
        Zone("B Link",                 (0.58, 0.74), (0.36, 0.50)),
        Zone("B Tower",                (0.72, 0.88), (0.40, 0.54)),
        Zone("B Tunnel",               (0.76, 0.92), (0.46, 0.60)),
        Zone("B site",                 (0.70, 0.96), (0.36, 0.60)),
        Zone("B Screen",               (0.88, 0.98), (0.50, 0.64)),
        Zone("B Hall",                 (0.76, 0.94), (0.58, 0.70)),
        Zone("B Back",                 (0.80, 0.96), (0.58, 0.70)),
        Zone("CT spawn",               (0.38, 0.62), (0.84, 0.96)),
    ],

    # -------------------------------------------------------------------------
    # FRACTURE  (CT top-split, T bottom; B left, A right)
    # -------------------------------------------------------------------------
    "fracture": [
        Zone("CT spawn B",         (0.08, 0.30), (0.04, 0.14)),
        Zone("CT spawn A",         (0.70, 0.92), (0.04, 0.14)),
        Zone("B Tree",             (0.16, 0.30), (0.14, 0.26)),
        Zone("B Tunnel",           (0.28, 0.44), (0.20, 0.32)),
        Zone("B Tower",            (0.12, 0.26), (0.22, 0.34)),
        Zone("B Canteen",          (0.22, 0.38), (0.30, 0.44)),
        Zone("B Generator",        (0.14, 0.30), (0.38, 0.52)),
        Zone("B Link",             (0.28, 0.42), (0.32, 0.46)),
        Zone("B site",             (0.10, 0.34), (0.22, 0.50)),
        Zone("B Arch",             (0.02, 0.12), (0.38, 0.52)),
        Zone("B Elbow",            (0.02, 0.12), (0.48, 0.62)),
        Zone("B Arcade",           (0.10, 0.26), (0.52, 0.66)),
        Zone("B Bench",            (0.18, 0.32), (0.60, 0.72)),
        Zone("B Main",             (0.04, 0.22), (0.56, 0.70)),
        Zone("B Lobby",            (0.08, 0.26), (0.68, 0.78)),
        Zone("Mid Window",         (0.38, 0.56), (0.16, 0.28)),
        Zone("Mid Top",            (0.38, 0.56), (0.28, 0.38)),
        Zone("Mid Stairs",         (0.38, 0.56), (0.38, 0.48)),
        Zone("Attacker Side Bridge", (0.32, 0.68), (0.46, 0.60)),
        Zone("Mid Bottom",         (0.36, 0.56), (0.50, 0.62)),
        Zone("A Crane",            (0.82, 0.96), (0.12, 0.24)),
        Zone("A Hall",             (0.68, 0.84), (0.14, 0.26)),
        Zone("A Door",             (0.66, 0.82), (0.24, 0.36)),
        Zone("A Rope",             (0.68, 0.82), (0.30, 0.42)),
        Zone("A Link",             (0.58, 0.74), (0.30, 0.42)),
        Zone("A Pocket",           (0.86, 0.96), (0.36, 0.50)),
        Zone("A Yard",             (0.60, 0.78), (0.40, 0.54)),
        Zone("A site",             (0.68, 0.96), (0.22, 0.50)),
        Zone("A Drop",             (0.66, 0.82), (0.52, 0.66)),
        Zone("A Elbow",            (0.87, 0.97), (0.44, 0.58)),
        Zone("A Dish",             (0.62, 0.78), (0.64, 0.78)),
        Zone("A Gate",             (0.56, 0.72), (0.66, 0.80)),
        Zone("A Main",             (0.68, 0.92), (0.54, 0.68)),
        Zone("A Lobby",            (0.72, 0.92), (0.66, 0.78)),
        Zone("T spawn",            (0.36, 0.62), (0.86, 0.96)),
    ],
}

_GRID = [
    ["top-left",    "top-center",    "top-right"],
    ["mid-left",    "center",        "mid-right"],
    ["bottom-left", "bottom-center", "bottom-right"],
]


def pos_to_zone(x: float, y: float, map_name: str) -> str:
    for zone in MAP_ZONES.get(map_name, []):
        if (zone.x_range[0] <= x < zone.x_range[1]
                and zone.y_range[0] <= y < zone.y_range[1]):
            return zone.name
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    col = min(int(x * 3), 2)
    row = min(int(y * 3), 2)
    return _GRID[row][col]


def enemies_to_callout(
    positions: List[Tuple[float, float]],
    map_name: str,
    lang: str = "EN",
) -> str:
    if not positions:
        return ""
    t = CALLOUT_TEMPLATES.get(lang, CALLOUT_TEMPLATES["EN"])
    zones = list(dict.fromkeys(pos_to_zone(x, y, map_name) for x, y in positions))
    n = len(positions)
    if n == 1:
        return t["one"].format(loc=zones[0])
    if n == 2:
        if len(zones) == 1:
            return t["two"].format(loc=zones[0])
        return t["split"].format(n=2, locs=f"{zones[0]} - {zones[1]}")
    return (
        t["multi"].format(n=n, loc=zones[0]) if len(zones) == 1
        else t["split"].format(n=n, locs=" - ".join(zones[:3]))
    )


def stack_callout(zone: str, count: int, map_name: str, lang: str = "EN") -> str:
    t = CALLOUT_TEMPLATES.get(lang, CALLOUT_TEMPLATES["EN"])
    return t["stack"].format(n=count, loc=zone)
