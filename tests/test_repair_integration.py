"""C7: actual failed MuJoCo scores trigger a genuine fake-LLM repair and a successful learn.

Only initial generation is deliberately replaced with the known colliding fixture. Repair, both local
scoring rounds, selection, fake replay, verification and saving use their production implementations.
All devices/services are fake and every skill/run artifact stays in the temporary test directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from agents import candidates, orchestrator, sandbox_pool
from arm import driver
from vision import scene

ROOT = Path(__file__).resolve().parents[1]


def test_failed_first_round_is_repaired_by_real_local_scoring(tmp_path, monkeypatch):
    skills, runs = tmp_path / "skills", tmp_path / "runs"
    shutil.copytree(ROOT / "skills" / "_starter", skills / "_starter")
    monkeypatch.setenv("FAKE", "1")
    monkeypatch.setenv("VARA_SKILLS_DIR", str(skills))
    monkeypatch.setenv("VARA_RUNS_DIR", str(runs))
    monkeypatch.delenv("DEMO_MODE", raising=False)
    source = (
        ROOT / "tests" / "fixtures" / "llm" / "candidates" / "juice_box_to_tray" / "cand_1.py"
    ).read_text(encoding="utf-8")
    assert candidates.check_source(source) == []

    def failing_generation(spec, camera_scene, saved_skills, *, task_dir, n, **kwargs):
        assert spec["name"] == "juice_box_to_tray"
        out = []
        for index in range(1, n + 1):
            path = Path(task_dir) / f"cand_{index}.py"
            path.write_text(source, encoding="utf-8")
            out.append(candidates.Candidate(index, candidates.HINTS[index - 1], source, [], 1, path))
        return out

    monkeypatch.setattr(candidates, "generate", failing_generation)
    driver.reset_driver()
    scene.reset_tracker()
    sandbox_pool.reset_pool()
    try:
        orch = orchestrator.Orchestrator()
        monkeypatch.setattr(orchestrator, "orch", orch)
        with TestClient(orchestrator.app) as client:
            reply = client.post("/command", json={"text": "put the juice box in the tray"})
            assert reply.status_code == 200, reply.text
            task_id = reply.json()["task_id"]
            finished = orchestrator.wait_for_stage(orch, task_id, orchestrator.FINAL_STAGES, 120.0)
            assert finished is not None, orch.recent_events()
            assert finished["stage"] == "done" and finished["detail"]["ok"], finished
            assert any(event["stage"] == "saved" for event in orch.recent_events())

        task_dir = runs / task_id
        first = [json.loads((task_dir / f"cand_{i}" / "score.json").read_text()) for i in range(1, 7)]
        repaired = [
            json.loads((task_dir / "repair" / f"cand_{i}" / "score.json").read_text()) for i in range(1, 7)
        ]
        assert all(not score["passed"] and score["collisions"] > 0 for score in first), first
        assert all("cereal_box" in score["fail_reason"] for score in first)
        passing = [score for score in repaired if score["passed"]]
        assert passing, repaired
        assert all(score["successes"] == 5 and score["collisions"] == 0 for score in passing)
        prompt = (task_dir / "prompts" / "repair_1.md").read_text(encoding="utf-8")
        assert "cereal_box" in prompt and "Score line:" in prompt
        assert prompt.count("### Failed program") == 3
        saved = json.loads((skills / "juice_box_to_tray.json").read_text(encoding="utf-8"))
        assert saved["sim"] == {"successes": 5, "trials": 5}
        assert saved["code"] != source
        scored = [event for event in orch.recent_events() if event["stage"] == "candidate_scored"]
        assert len(scored) == 12 and all(event["detail"]["runner"] == "local" for event in scored)
        print(
            f"C7 evidence: {task_dir}; first round 0/6 passing; "
            f"repair {len(passing)}/6 passing; terminal={finished['stage']}"
        )
    finally:
        driver.reset_driver()
        scene.reset_tracker()
        sandbox_pool.reset_pool()
