"""Lane B test inputs: the kit table layout the vision tests' geometry was written against.

config/scene.yaml, config/objects.yaml and config/fake_scene.json describe the real table and props (the
props changed on 2026-10-18, the camera pose and calibration markers were tape-measured that day, and
config/homography.npy is replaced by the measured D6 calibration). Tests whose expectations depend on where
things land in the synthetic image (which calibration marker a raised item hides, which pixel a check spot
projects to) use this frozen copy instead, so a re-taped table or a new prop never changes what they check.

The values are the kit layout as committed before the prop change (c7e95fd): the placeholder camera at
(0.20, -0.05, 0.65), calibration markers 2/3 at x = 0.27 and 6/7 at y = +-0.17, the 0.20 m cereal box at
(0.28, 0.07), which hides calibration marker 6 in the synthetic image, and the 0.07 m tray at (0.30, 0.22).
They are literals, not `git show`, so later commits cannot move them.

`write_kit_layout(directory)` writes the three files plus the kit camera's synthetic homography;
`install_kit_layout(monkeypatch, directory)` also points the module path globals that `load_scene_cfg`,
`marker_table`, `load_objects` and `load_fake_scene` read at call time at those files, so a default
`FakeCamera()` (what `scripts/capture_table.py --fake` opens) renders the kit layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision import calibrate, cameras, markers

KIT_SCENE_YAML = """\
# Test fixture (tests/vision_kit.py): the kit layout of config/scene.yaml before the 2026-10-18 measurements.
workspace: {x: [0.10, 0.40], y: [-0.25, 0.25], z: [0.02, 0.30]}
zones:
  tray:
    x: 0.30
    y: 0.22
    r: 0.07
    wall: 0.035
    slots: [[0.275, 0.195], [0.325, 0.195], [0.275, 0.245], [0.325, 0.245]]
  handoff: {x: 0.28, y: -0.08, z: 0.10}
camera:
  Hc: 0.65
  c: [0.20, -0.05]
calib_markers:
  0: [0.08, 0.34]
  1: [0.08, -0.34]
  2: [0.27, 0.34]
  3: [0.27, -0.34]
  4: [0.46, 0.34]
  5: [0.46, -0.34]
  6: [0.46, 0.17]
  7: [0.46, -0.17]
home_q: [0.0, -1.35, 0.02, 1.45, 0.0]
"""

KIT_OBJECTS_YAML = """\
# Test fixture (tests/vision_kit.py): the kit props of config/objects.yaml before the 2026-10-18 prop change.
10: {name: pill_bottle, kind: object, shape: cylinder, size: [0.035, 0.035], h: 0.07, held_min: 0.7}
11: {name: juice_box, kind: object, shape: box, size: [0.063, 0.04], h: 0.105, held_min: 7.4}
12: {name: tea_tin, kind: object, shape: cylinder, size: [0.04, 0.04], h: 0.09, held_min: 16.1}
30: {name: cereal_box, kind: obstacle, shape: box, size: [0.19, 0.06], h: 0.2}
40: {name: hand, kind: hand, h: 0.03}
"""

KIT_FAKE_SCENE: dict[str, Any] = {
    "t": 1792209600.12,
    "objects": {
        "pill_bottle": {
            "marker": 10,
            "x": 0.18,
            "y": -0.12,
            "yaw": 0.0,
            "h": 0.07,
            "size": [0.035, 0.035],
            "seen": True,
        },
        "juice_box": {
            "marker": 11,
            "x": 0.24,
            "y": -0.04,
            "yaw": 0.3,
            "h": 0.105,
            "size": [0.063, 0.04],
            "seen": True,
        },
        "tea_tin": {
            "marker": 12,
            "x": 0.16,
            "y": -0.02,
            "yaw": 0.0,
            "h": 0.09,
            "size": [0.04, 0.04],
            "seen": True,
        },
    },
    "obstacles": {
        "cereal_box": {
            "marker": 30,
            "x": 0.28,
            "y": 0.07,
            "yaw": 0.0,
            "h": 0.2,
            "size": [0.19, 0.06],
            "seen": True,
        }
    },
    "hand": {"marker": 40, "x": 0.28, "y": -0.08, "z": 0.03, "seen": False},
    "zones": {
        "tray": {
            "x": 0.3,
            "y": 0.22,
            "r": 0.07,
            "wall": 0.035,
            "slots": [[0.275, 0.195], [0.325, 0.195], [0.275, 0.245], [0.325, 0.245]],
        },
        "handoff": {"x": 0.28, "y": -0.08, "z": 0.1},
    },
}


@dataclass(frozen=True)
class KitLayout:
    """The kit layout's files in a temporary directory."""

    scene_path: Path
    objects_path: Path
    fake_scene_path: Path
    homography_path: Path  # the kit camera's synthetic homography (what `calibrate --fake` would write)

    def cfg(self) -> dict[str, Any]:
        return calibrate.load_scene_cfg(self.scene_path)

    def table(self) -> dict[int, dict[str, Any]]:
        return markers.marker_table(self.objects_path, self.scene_path)

    def fake_scene(self) -> dict[str, Any]:
        return cameras.load_fake_scene(self.fake_scene_path)


def write_kit_layout(directory: Path) -> KitLayout:
    """Write the kit scene.yaml, objects.yaml, fake_scene.json and synthetic homography.npy into
    `directory` (created) and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layout = KitLayout(
        scene_path=directory / "scene.yaml",
        objects_path=directory / "objects.yaml",
        fake_scene_path=directory / "fake_scene.json",
        homography_path=directory / "homography.npy",
    )
    layout.scene_path.write_text(KIT_SCENE_YAML, encoding="utf-8")
    layout.objects_path.write_text(KIT_OBJECTS_YAML, encoding="utf-8")
    layout.fake_scene_path.write_text(json.dumps(KIT_FAKE_SCENE, indent=2), encoding="utf-8")
    calibrate.save_homography(calibrate.synthetic_homography(layout.cfg()), layout.homography_path)
    return layout


def install_kit_layout(monkeypatch: Any, directory: Path) -> KitLayout:
    """`write_kit_layout(directory)`, then make the default config readers load it for this test:
    vision.calibrate.SCENE_PATH (camera pose, calibration markers, zones), vision.markers.SCENE_PATH and
    OBJECTS_PATH (the marker table) and vision.cameras.FAKE_SCENE_PATH (the scene a default FakeCamera
    renders). config/homography.npy is not redirected: pass `homography_path` explicitly."""
    layout = write_kit_layout(directory)
    monkeypatch.setattr(calibrate, "SCENE_PATH", layout.scene_path)
    monkeypatch.setattr(markers, "SCENE_PATH", layout.scene_path)
    monkeypatch.setattr(markers, "OBJECTS_PATH", layout.objects_path)
    monkeypatch.setattr(cameras, "FAKE_SCENE_PATH", layout.fake_scene_path)
    return layout
