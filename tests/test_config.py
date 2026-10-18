"""Config sanity checks (A11 and B4/B5 keep these green)."""

import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
OBJECTS = yaml.safe_load((ROOT / "config" / "objects.yaml").read_text())
SCENE = yaml.safe_load((ROOT / "config" / "scene.yaml").read_text())


def test_objects_yaml_entries():
    names = set()
    kinds = {"object": 0, "obstacle": 0, "hand": 0}
    for marker, entry in OBJECTS.items():
        assert isinstance(marker, int) and 10 <= marker <= 49, marker
        assert entry["kind"] in kinds, entry
        kinds[entry["kind"]] += 1
        assert entry["name"] not in names, entry["name"]
        names.add(entry["name"])
        assert entry["h"] > 0
        if entry["kind"] in ("object", "obstacle"):
            assert entry["shape"] in ("box", "cylinder")
            assert len(entry["size"]) == 2 and all(v > 0 for v in entry["size"])
        if entry["kind"] == "object":
            assert 0.06 <= entry["h"] <= 0.115, f"{entry['name']}: height must be 6 to 11.5 cm"
            assert entry["held_min"] > 0
    assert kinds["object"] >= 3 and kinds["obstacle"] >= 1 and kinds["hand"] == 1


def test_scene_yaml_has_workspace_zones_and_calibration():
    ws = SCENE["workspace"]
    assert ws == {"x": [0.10, 0.40], "y": [-0.25, 0.25], "z": [0.02, 0.30]}
    assert {"tray", "handoff"} <= set(SCENE["zones"])
    assert sorted(SCENE["calib_markers"]) == list(range(8))
    assert len(SCENE["home_q"]) == 5


def test_fake_scene_matches_config():
    fake = json.loads((ROOT / "config" / "fake_scene.json").read_text())
    assert fake["zones"] == SCENE["zones"]
    for marker, entry in OBJECTS.items():
        group = {"object": "objects", "obstacle": "obstacles"}.get(entry["kind"])
        if group:
            item = fake[group][entry["name"]]
            assert item["marker"] == marker
            assert item["h"] == entry["h"] and item["size"] == entry["size"]
