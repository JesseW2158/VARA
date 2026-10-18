"""The scaffold (S0): every module of the repo map imports under FAKE=1 with its public names, and
`scripts/e2e_fake.py "put the juice box in the tray"` runs end to end as a subprocess (the one slow test).

The end-to-end run uses a copy of skills/ as shipped (starters and backups, without skills learned or loaded
on this laptop) and an empty runs/ under tmp_path (VARA_SKILLS_DIR, VARA_RUNS_DIR), so the repo's own folders
stay untouched; a second run with `--reset` removes the learned skill again.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "scripts" / "e2e_fake.py"
E2E_LIMIT_S = 120.0

# Public names per module (brief sections 4, 8 and 9), for every module of the CLAUDE.md repo map.
PUBLIC_NAMES: dict[str, list[str]] = {
    "arm.driver": [
        "limits",
        "JOINTS",
        "TICK",
        "MAX_STEP_RAD",
        "MARGIN_RAD",
        "Driver",
        "FakeDriver",
        "get_driver",
        "EscWatcher",
    ],
    "arm.ik": [
        "MODEL_PATH",
        "load_model",
        "fk",
        "fk_tip",
        "jaw_axes",
        "PITCH_LADDER",
        "solve",
        "solve_exact",
        "fk_grasp",
        "solve_grasp",
        "grasp_offset",
    ],
    "arm.sim_robot": ["SimTimeLimit", "SIM_TIME_LIMIT", "DT", "SimRobot"],
    "arm.replay": ["validate", "load_limits", "load_box", "bridge", "replay"],
    "arm.real_robot": ["home", "freeze", "carry_to_hand", "cancelled", "clear_cancel"],
    "vision.cameras": ["open_camera", "Camera", "FakeCamera", "get_camera"],
    "vision.markers": ["DICT", "detector", "detect", "marker_table", "draw"],
    "vision.calibrate": ["fit_homography", "pixel_to_table", "height_correct", "calibration_error"],
    "vision.scene": ["SceneTracker", "FakeSceneTracker", "get_tracker"],
    "vision.verify": ["verify", "take_after"],
    "agents.llm": ["chat", "router_model", "codegen_model"],
    "agents.router": ["route", "keyword_route"],
    "agents.candidates": [
        "HINTS",
        "Candidate",
        "generate",
        "repair",
        "routine_candidate",
        "extract_code",
        "check_source",
    ],
    "agents.library": ["Library", "make_skill", "routine_spec"],
    "agents.sandbox_pool": ["LocalRunner", "SandboxPool", "get_pool", "build_snapshot"],
    "agents.planner": ["plan", "PlanError"],
    "agents.orchestrator": ["app", "orch", "Orchestrator", "BUDGETS"],
    "sim.candidate_check": ["check_source"],
    "sim.success": ["evaluate", "resolve_object", "already_true"],
    "sim.run_candidate": ["main"],
    "sim.scoring": ["perturb_scene", "trial_result", "summarize", "rank"],
    "rl.handoff_env": ["HandoffEnv", "build_observation"],
    "rl.train": ["train", "EnvFactory", "RewardWrapper"],
    "rl.evaluate": ["evaluate", "run_episodes", "load_policy"],
    "rl.policy_runner": ["run_policy"],
    "rl.rewards.write_rewards": ["write_rewards"],
    "rl.train_parallel": ["train_parallel", "build_rl_snapshot"],
}
SCRIPTS = ["e2e_fake.py", "daytona_smoke.py", "make_markers.py", "touch_markers.py"]
PAGES = [
    "web/index.html",
    "web/voice.html",
    "web/static/app.js",
    "web/static/voice.js",
    "web/static/style.css",
]
STAGE_LINE = re.compile(r"^(\w+)\s{2,}")


@pytest.fixture(autouse=True)
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE", "1")


@pytest.mark.parametrize("module", sorted(PUBLIC_NAMES))
def test_module_imports_with_its_public_names(module: str) -> None:
    mod = importlib.import_module(module)
    missing = [name for name in PUBLIC_NAMES[module] if not hasattr(mod, name)]
    assert not missing, f"{module} lacks {missing}"


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses_and_has_a_main(script: str) -> None:
    source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=script)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "main" in names, script
    assert "sys.path.insert(0, str(ROOT))" in source, f"{script} must put the repo root on sys.path"


@pytest.mark.parametrize("page", PAGES)
def test_web_page_exists(page: str) -> None:
    assert (ROOT / page).is_file()


def e2e_env(skills: Path, runs: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {"FAKE": "1", "VARA_SKILLS_DIR": str(skills), "VARA_RUNS_DIR": str(runs), "PYTHONIOENCODING": "utf-8"}
    )
    env.pop("DEMO_MODE", None)
    return env


def run_e2e(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(E2E), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=E2E_LIMIT_S,
    )


def stage_lines(stdout: str) -> list[str]:
    stages = []
    for line in stdout.splitlines():
        match = STAGE_LINE.match(line)
        if match and match.group(1) not in ("task", "saved_skill"):
            stages.append(match.group(1))
    return stages


def copy_shipped_skills(src: Path, dst: Path) -> None:
    """Copy skills/ as shipped: `_starter`, `_backup` and the starter live files.

    Skills learned or loaded from the backup on this laptop stay behind, so a demo rehearsal that left
    one in the live folder does not change what the fake learn starts from.
    """
    starters = {path.name for path in (src / "_starter").glob("*.json")}

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = {name for name in names if name.endswith(".tmp")}
        if Path(directory) == src:
            skipped |= {name for name in names if name.endswith(".json") and name not in starters}
        return skipped

    shutil.copytree(src, dst, ignore=ignore)


def test_e2e_fake_learns_and_saves_the_juice_box_skill(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    copy_shipped_skills(ROOT / "skills", skills)
    runs = tmp_path / "runs"
    env = e2e_env(skills, runs)
    result = run_e2e(["put the juice box in the tray"], env)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-3000:]}"
    stages = stage_lines(result.stdout)
    assert stages, result.stdout
    assert stages[-1] == "saved", stages
    assert stages[:2] == ["routing", "spec_ready"]
    assert stages.count("writing_code") == 6 and stages.count("candidate_scored") == 6
    assert "selected" in stages and "executing" in stages and "verifying" in stages
    assert "saved skill:" in result.stdout
    learned = skills / "juice_box_to_tray.json"
    assert learned.is_file()
    assert sorted(p.stem for p in skills.glob("*.json")) == [
        "hand_over",
        "juice_box_to_tray",
        "place_in_tray",
    ]
    assert (runs / "t-0001" / "events.jsonl").is_file()
    assert not (ROOT / "skills" / "juice_box_to_tray.json").exists()
    reset = run_e2e(["--reset"], env)
    assert reset.returncode == 0, reset.stdout + reset.stderr[-2000:]
    assert "reset demo, live skills: hand_over, place_in_tray" in reset.stdout
    assert sorted(p.stem for p in skills.glob("*.json")) == ["hand_over", "place_in_tray"]
    assert list((runs / "archive").rglob("juice_box_to_tray.json"))
