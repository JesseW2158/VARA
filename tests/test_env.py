"""D1/D2 Done-when: HandoffEnv runs a random policy across resets, the spaces and the observation layout hold,
+1 moves a joint by at most 2 degrees per step, success and falls trigger when forced, reset(seed) is
deterministic, the randomizations are applied, the baseline reward passes check_source and computes its
formula, and rl.train / rl.evaluate run end to end as subprocesses."""

from __future__ import annotations

import builtins
import csv
import json
import math
import pathlib
import pickle
import subprocess
import sys
import time
import types

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from arm import ik
from rl import handoff_env as he
from rl.handoff_env import HandoffEnv
from sim.candidate_check import check_source
from sim.restricted import load_entry

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE = ROOT / "rl" / "rewards" / "baseline.py"
DEG = math.pi / 180


@pytest.fixture
def env():
    env = HandoffEnv()
    yield env
    env.close()


def inside_box(p) -> bool:
    return all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(p, he.TARGET_BOX, strict=True))


def test_random_policy_runs_1000_steps_across_resets(env):
    env.action_space.seed(1)
    obs, info = env.reset(seed=1)
    episodes = 0
    for _ in range(1000):
        assert obs.shape == (16,) and obs.dtype == np.float32 and np.all(np.isfinite(obs))
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert math.isfinite(reward)
        assert isinstance(info["success"], bool) and isinstance(info["fell"], bool)
        assert info["dist"] >= 0.0 and inside_box(info["target"])
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()
    assert episodes >= 5  # 1000 steps of 150-step episodes


def test_spaces_and_observation_layout(env):
    assert env.action_space.shape == (4,) and env.observation_space.shape == (16,)
    assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)
    obs, info = env.reset(seed=3)
    assert obs.shape == (16,) and obs.dtype == np.float32
    q = obs[he.OBS_LAYOUT["joints"]]
    assert np.all(np.abs(q - np.array(ik.home_q())) < (5.0 + 2.0 + 2.5) * DEG)  # jitter + offset + noise
    assert np.all(obs[he.OBS_LAYOUT["prev_action"]] == 0.0)
    tip = obs[he.OBS_LAYOUT["tip"]]
    assert np.linalg.norm(tip - np.array(ik.fk_tip(q))) < 0.006  # fk of the measured joints (+1 mm noise)
    target = obs[he.OBS_LAYOUT["target"]]
    assert inside_box(np.array(info["target"]))
    assert np.linalg.norm(target - np.array(info["target"])) < 0.006  # no lag yet, 1 mm noise
    assert abs(obs[15] - np.linalg.norm(tip - target)) < 1e-5
    assert np.linalg.norm(np.array(info["tip"]) - np.array(ik.fk_tip(env.joint_angles()))) < 1e-9
    obs2, _, _, _, info2 = env.step([0.3, -0.2, 0.1, 0.0])
    assert np.allclose(obs2[he.OBS_LAYOUT["prev_action"]], [0.3, -0.2, 0.1, 0.0], atol=1e-6)
    assert info2["step"] == 1 and abs(info2["dist"] - math.dist(info2["tip"], info2["target"])) < 1e-9
    check_env(env, warn=False)


def test_plus_one_moves_a_joint_by_at_most_two_degrees_per_step(env):
    for seed in range(4):
        env.reset(seed=seed)
        q_start = np.array(env.joint_angles())
        q_prev = q_start.copy()
        cmd_prev = np.array(env.command())
        n = 12
        for _ in range(n):
            env.step([1.0, 0.0, 0.0, 0.0])
            cmd = np.array(env.command())
            q = np.array(env.joint_angles())
            assert abs((cmd[0] - cmd_prev[0]) - he.MAX_STEP_RAD) < 1e-12  # the command moves exactly 2 deg
            assert np.allclose(cmd[1:], cmd_prev[1:])  # the other joints hold
            assert q[0] - q_prev[0] <= 3.0 * DEG  # the servo tracks the ramp with a little overshoot
            cmd_prev, q_prev = cmd, q
        moved = q[0] - q_start[0]
        delay = env.params.action_delay
        assert moved <= n * he.MAX_STEP_RAD + 1.0 * DEG
        assert moved >= (n - delay - 1) * he.MAX_STEP_RAD - 1.0 * DEG
    # a negative action on another joint, and the clamp 3 degrees inside the limit
    env.reset(seed=9)
    cmd0 = np.array(env.command())
    env.step([0.0, 0.0, 0.0, -1.0])
    assert abs((np.array(env.command())[3] - cmd0[3]) + he.MAX_STEP_RAD) < 1e-12
    hi = ik.limits()["shoulder_pan"][1] - 3 * DEG
    for _ in range(80):
        env.step([1.0, 0.0, 0.0, 0.0])
    assert abs(env.command()[0] - hi) < 1e-9
    assert env.joint_angles()[0] <= ik.limits()["shoulder_pan"][1]


def test_action_delay_holds_back_the_applied_command(env):
    seen = set()
    for seed in range(30):
        env.reset(seed=seed)
        delay = env.params.action_delay
        seen.add(delay)
        before = env.applied_command()
        env.step([1.0, 0.0, 0.0, 0.0])
        if delay == 0:
            assert env.applied_command() == env.command()
        else:
            assert env.applied_command() == before
            for _ in range(delay):
                env.step([0.0, 0.0, 0.0, 0.0])
            assert env.applied_command() == env.command()
    assert seen == {0, 1, 2}


def test_success_when_the_target_is_set_to_the_tip(env):
    obs, info = env.reset(seed=7)
    env.set_target(ik.fk_tip(env.joint_angles()))
    assert info["success"] is False
    steps_taken = 0
    for _ in range(40):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))
        steps_taken += 1
        assert info["dist"] < he.SUCCESS_RADIUS
        if terminated:
            break
    assert terminated and not truncated and info["success"] is True and info["fell"] is False
    assert steps_taken == he.SUCCESS_STEPS and info["streak"] == he.SUCCESS_STEPS
    assert reward > 0.9  # the +1 bonus minus a tiny distance


def test_fall_terminates_with_the_fell_flag(env):
    env.reset(seed=11)
    q, ok = ik.solve((0.22, 0.0, 0.02), -math.pi / 2)  # jaws straight down, tip 2 cm above the floor
    assert ok
    env.set_joints(q)
    assert ik.fk_tip(env.joint_angles())[2] < he.FALL_Z
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated and info["fell"] is True and info["success"] is False
    assert info["tip"][2] < he.FALL_Z
    assert reward < he.FALL_PENALTY  # the built-in reward carries the fall penalty
    # and a high tip never counts as fallen
    env.reset(seed=11)
    _, _, terminated, _, info = env.step(np.zeros(4))
    assert not terminated and info["fell"] is False


def test_reset_seed_is_deterministic():
    a, b = HandoffEnv(), HandoffEnv()
    try:
        obs_a, info_a = a.reset(seed=42)
        obs_b, info_b = b.reset(seed=42)
        assert np.array_equal(obs_a, obs_b)
        assert info_a == info_b
        actions = np.random.default_rng(0).uniform(-1, 1, size=(30, 4))
        first = [a.step(act) for act in actions]
        obs_again, info_again = a.reset(seed=42)
        assert np.array_equal(obs_again, obs_a) and info_again == info_a
        second = [a.step(act) for act in actions]
        for (o1, r1, t1, u1, i1), (o2, r2, t2, u2, i2) in zip(first, second, strict=True):
            assert np.array_equal(o1, o2) and r1 == r2 and t1 == t2 and u1 == u2 and i1 == i2
        assert not np.array_equal(a.reset(seed=43)[0], obs_a)
    finally:
        a.close()
        b.close()


def test_randomizations_are_applied(env):
    params = [env.reset(seed=seed)[1]["params"] for seed in range(40)]
    gains = {round(p["gain_scale"], 9) for p in params}
    assert len(gains) > 1 and all(0.8 <= g <= 1.2 for g in gains)
    offsets = {tuple(p["joint_offset"]) for p in params}
    assert len(offsets) > 1 and all(len(o) == 5 and max(abs(v) for v in o) <= 2 * DEG for o in offsets)
    delays = {p["action_delay"] for p in params}
    assert delays == {0, 1, 2}
    lags = {p["obs_lag_steps"] for p in params}
    assert len(lags) > 1 and all(3 <= lag <= 15 for lag in lags)
    assert all(0.1 <= p["obs_lag_s"] <= 0.5 for p in params)
    drifting = [p["drifting"] for p in params]
    assert any(drifting) and not all(drifting)
    assert all((p["target_speed"] > 0) == p["drifting"] for p in params)
    assert all(0.01 <= p["target_speed"] <= 0.05 for p in params if p["drifting"])
    assert env.params.as_dict() == params[-1]


def test_the_gain_scale_reaches_the_actuators(env):
    env.reset(seed=0)
    kp0 = env._kp0[:4]
    for seed in range(5):
        env.reset(seed=seed)
        scale = env.params.gain_scale
        gain = env._model.actuator_gainprm[env._act[:4], 0]
        bias = env._model.actuator_biasprm[env._act[:4], 1]
        assert np.allclose(gain, kp0 * scale) and np.allclose(bias, -kp0 * scale)


def test_target_drifts_and_the_observation_lags_behind_it(env):
    for seed in range(60):
        obs, info = env.reset(seed=seed)
        if env.params.drifting:
            break
    assert env.params.drifting
    lag = env.params.obs_lag_steps
    targets = [np.array(info["target"])]
    for _ in range(40):
        obs, _, _, _, info = env.step(np.zeros(4))
        targets.append(np.array(info["target"]))
        assert inside_box(info["target"])
        observed = obs[he.OBS_LAYOUT["target"]]
        stale = targets[max(0, len(targets) - 1 - lag)]
        assert np.linalg.norm(observed - stale) < 0.006  # the lagged target plus 1 mm noise
    moved = np.linalg.norm(targets[-1] - targets[0])
    assert 0.4 * 40 * he.DT * 0.01 < moved <= 40 * he.DT * 0.05 + 1e-9
    # a target that is not drifting stays put and is observed where it is
    for seed in range(60):
        obs, info = env.reset(seed=seed)
        if not env.params.drifting:
            break
    first = np.array(info["target"])
    for _ in range(20):
        obs, _, _, _, info = env.step(np.zeros(4))
    assert np.array_equal(np.array(info["target"]), first)


def test_measured_joints_carry_the_offset_and_noise(env):
    env.reset(seed=5)
    offset = np.array(env.params.joint_offset)
    residuals = []
    for _ in range(60):
        obs, _, _, _, _ = env.step(np.zeros(4))
        residuals.append(obs[he.OBS_LAYOUT["joints"]] - np.array(env.joint_angles()) - offset)
    residuals = np.array(residuals)
    assert np.all(np.abs(residuals.mean(axis=0)) < 0.4 * DEG)  # noise averages out, the offset is exact
    assert 0.2 * DEG < residuals.std() < 0.9 * DEG  # about 0.5 degrees of noise


def test_build_observation_matches_the_env_layout():
    q = list(ik.home_q())
    obs = he.build_observation(q, [0.1, 0.2, 0.3, 0.4], [0.3, 0.0, 0.1])
    assert obs.shape == (16,) and obs.dtype == np.float32
    assert np.allclose(obs[:5], q) and np.allclose(obs[5:9], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(obs[9:12], ik.fk_tip(q), atol=1e-6)
    assert np.allclose(obs[12:15], [0.3, 0.0, 0.1])
    assert abs(obs[15] - math.dist(ik.fk_tip(q), (0.3, 0.0, 0.1))) < 1e-5


def test_baseline_reward_passes_check_source_and_computes_the_formula(env):
    src = BASELINE.read_text(encoding="utf-8")
    assert check_source(src, entry="reward") == []
    reward = load_entry(src, entry="reward")
    obs = [0.0] * 16
    far = {"dist": 0.10, "success": False, "fell": False}
    assert reward(obs, [0.5, 0.0, 0.0, 0.0], [0.0] * 4, far) == pytest.approx(-0.10 - 0.01 * 0.25)
    near = {"dist": 0.02, "success": False, "fell": False}
    assert reward(obs, [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], near) == pytest.approx(-0.02 - 0.02 + 1.0)
    assert reward(obs, [0.2] * 4, [0.2] * 4, {"dist": 0.03}) == pytest.approx(-0.03)  # d == 0.03 is no bonus
    # the env's built-in reward is the same formula on a real step (no fall)
    env.reset(seed=2)
    prev = [0.0] * 4
    for action in ([0.4, -0.3, 0.2, 0.1], [-0.5, 0.5, 0.0, 0.0]):
        obs_t, built_in, _, _, info = env.step(action)
        assert built_in == pytest.approx(reward([float(v) for v in obs_t], action, prev, info))
        prev = action


def test_check_cli_runs(capsys):
    assert he.main(["--check", "--steps", "40", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "obs shape (16,)" in out and "successes" in out


def test_reward_wrapper_tracks_prev_action_and_keeps_the_fall_penalty():
    from rl.train import RewardWrapper

    calls = []

    def stand_in(obs, action, prev_action, info):  # a test stand-in, not AI-written code
        calls.append((list(action), list(prev_action)))
        return 0.5 if not info["fell"] else 0.0

    wrapper = RewardWrapper(HandoffEnv(), stand_in)
    try:
        wrapper.reset(seed=4)
        _, reward, _, _, info = wrapper.step([0.3, 0.0, -0.2, 0.0])
        assert reward == 0.5 and calls[-1] == ([0.3, 0.0, -0.2, 0.0], [0.0, 0.0, 0.0, 0.0])
        wrapper.step([1.5, 0.0, 0.0, 0.0])  # clipped before the reward sees it
        assert calls[-1] == ([1.0, 0.0, 0.0, 0.0], [0.3, 0.0, -0.2, 0.0])
        wrapper.reset(seed=11)
        assert calls and wrapper.env.joint_angles()  # the inner env is the HandoffEnv
        q, ok = ik.solve((0.22, 0.0, 0.02), -math.pi / 2)
        assert ok
        wrapper.env.set_joints(q)
        _, reward, terminated, _, info = wrapper.step(np.zeros(4))
        assert terminated and info["fell"] and reward == he.FALL_PENALTY
        assert calls[-1][1] == [0.0, 0.0, 0.0, 0.0]  # prev_action restarts at reset
    finally:
        wrapper.close()


def test_reward_wrapper_rejects_a_non_finite_reward():
    from rl.train import RewardWrapper

    wrapper = RewardWrapper(HandoffEnv(), lambda obs, action, prev_action, info: math.nan)
    try:
        wrapper.reset(seed=0)
        with pytest.raises(ValueError, match="reward"):
            wrapper.step(np.zeros(4))
    finally:
        wrapper.close()


def test_train_rejects_a_bad_reward_file(tmp_path):
    from rl import train

    bad = tmp_path / "bad.py"
    bad_src = "import os\n\ndef reward(obs, action, prev_action, info):\n    return 0.0\n"
    bad.write_text(bad_src, encoding="utf-8")
    with pytest.raises(ValueError, match="reward rejected"):
        train.train(bad, 64, tmp_path / "out", n_envs=1, eval_episodes=0)
    assert train.rollout_sizes(1024, 1) == (1024, 256)
    assert train.rollout_sizes(500_000, 8) == (1024, 256)
    assert train.rollout_sizes(100, 1) == (100, 100)  # no standard batch divides 100: one batch
    assert train.rollout_sizes(10, 1) == (64, 64)
    assert train.rollout_sizes(4096, 2) == (1024, 256)


def test_train_and_evaluate_smoke(tmp_path):
    out = tmp_path / "smoke"
    cmd = [sys.executable, "-m", "rl.train", "--reward", str(BASELINE), "--steps", "1024", "--n-envs", "1"]
    cmd += ["--out", str(out)]
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert elapsed < 90.0, f"train.py took {elapsed:.0f} s"
    assert (out / "policy.zip").is_file()
    with (out / "progress.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and set(rows[0]) == {"step", "mean_reward", "success_rate"}
    assert int(rows[-1]["step"]) >= 1024
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert 0.0 <= summary["success_rate"] <= 1.0 and summary["episodes"] == 100
    assert summary["mean_steps_to_success"] is None or summary["mean_steps_to_success"] > 0
    last_line = proc.stdout.strip().splitlines()[-1]
    assert json.loads(last_line)["success_rate"] == summary["success_rate"]

    cmd = [sys.executable, "-m", "rl.evaluate", "--policy", str(out / "policy.zip"), "--episodes", "3"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["episodes"] == 3 and 0.0 <= result["success_rate"] <= 1.0
    assert "mean_steps_to_success" in result


# ------------------------------------------------------------------ regressions from the review of round 1


def test_the_action_delay_reaches_the_actuators(env):
    """The simulator's ctrl holds the command from `action_delay` steps ago minus the joint offset, on every
    step: a delay that only existed in the applied_command() bookkeeping would not slow the arm down."""
    rng = np.random.default_rng(0)
    seen = set()
    for seed in range(30):
        env.reset(seed=seed)
        delay = env.params.action_delay
        seen.add(delay)
        offset = np.array(env.params.joint_offset)
        commands = [np.array(env.command())]
        assert np.allclose(env.actuator_targets(), commands[0] - offset, atol=1e-12)
        for _ in range(12):
            env.step(rng.uniform(-1.0, 1.0, size=4))
            commands.append(np.array(env.command()))
            expected = commands[max(0, len(commands) - 1 - delay)] - offset
            assert np.allclose(env.actuator_targets(), expected, atol=1e-12)
            assert np.allclose(env.actuator_targets(), np.array(env.applied_command()) - offset, atol=1e-12)
            if delay > 0:  # the current command is NOT what the actuators track yet
                assert not np.allclose(env.actuator_targets(), commands[-1] - offset, atol=1e-9)
    assert seen == {0, 1, 2}


def test_success_is_judged_on_the_true_target_not_the_observed_one(env):
    """The lag history can disagree with the true target for `obs_lag_steps` steps: success (and info['dist'])
    follow the true one, the observation the lagged one."""
    for seed in range(60):
        obs, info = env.reset(seed=seed)
        if env.params.obs_lag_steps >= 5:
            break
    assert env.params.obs_lag_steps >= 5
    far = (0.35, 0.15, 0.12)
    tip = ik.fk_tip(env.joint_angles())
    env.set_target(tip, observed=far)  # true target on the tip, the policy still sees it far away
    steps = 0
    for _ in range(40):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))
        steps += 1
        if steps == 1:
            assert obs[15] > 0.1 and info["dist"] < he.SUCCESS_RADIUS and reward > 0.9
        if terminated:
            break
    assert terminated and info["success"] is True and steps == he.SUCCESS_STEPS
    # the reverse: the observation says near while the true target is far -> never a success
    env.reset(seed=seed)
    env.set_target(far, observed=ik.fk_tip(env.joint_angles()))
    for step in range(1, 21):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert not terminated and info["success"] is False and info["streak"] == 0
        assert info["dist"] > 0.1 and reward < -0.1
        if step == 1:
            assert obs[15] < 0.05  # the observed target sits on the tip for the first `lag` steps
    assert np.allclose(env.target(), far)


def test_test_hooks_reject_bad_input_and_use_before_reset(env):
    fresh = HandoffEnv()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            fresh.applied_command()
        with pytest.raises(RuntimeError, match="reset"):
            fresh.actuator_targets()
        with pytest.raises(RuntimeError, match="reset"):
            fresh.step(np.zeros(4))
    finally:
        fresh.close()
    env.reset(seed=0)
    env.set_target([0.30, 0.0, 0.10], velocity=[0.02, 0.0, 0.0])
    for bad in ([math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0], [1.0, 2.0], "abc", [None, 0.0, 0.0]):
        with pytest.raises(ValueError, match="velocity"):
            env.set_target([0.31, 0.0, 0.10], velocity=bad)
        with pytest.raises(ValueError, match="target"):
            env.set_target(bad)
        with pytest.raises(ValueError, match="observed target"):
            env.set_target([0.31, 0.0, 0.10], observed=bad)
    assert env.target() == [0.30, 0.0, 0.10]  # a rejected call changes nothing
    obs, _, _, _, info = env.step(np.zeros(4))
    assert np.allclose(info["target"], [0.30 + 0.02 * he.DT, 0.0, 0.10])  # the velocity set before survived
    assert np.linalg.norm(obs[he.OBS_LAYOUT["target"]] - [0.30, 0.0, 0.10]) < 0.006  # and so did the history
    assert np.all(np.isfinite(obs)) and math.isfinite(info["dist"])


def test_reward_wrapper_gives_the_reward_a_copy_of_info():
    """A reward file that writes info['success'] (accepted by check_source) must not forge the training
    metric that Monitor, ProgressCallback and progress.csv read."""
    from rl.train import RewardWrapper, copy_info

    hostile = (
        "import math\n\n"
        "def reward(obs, action, prev_action, info):\n"
        "    info['success'] = True\n"
        "    info['dist'] = 0.0\n"
        "    info['fell'] = True\n"
        "    info['target'][0] = 9.0\n"
        "    info['params']['gain_scale'] = 9.0\n"
        "    obs[0] = 9.0\n"
        "    action[0] = 9.0\n"
        "    return 1.0\n"
    )
    assert check_source(hostile, entry="reward") == []
    wrapper = RewardWrapper(HandoffEnv(), load_entry(hostile, entry="reward"))
    try:
        wrapper.reset(seed=4)
        obs, reward, terminated, truncated, info = wrapper.step([0.3, 0.0, -0.2, 0.0])
        assert reward == 1.0  # the env's own `fell` (False) decides the penalty, not the forged flag
        assert info["success"] is False and info["fell"] is False and not terminated
        assert info["dist"] > 0.05 and info["target"][0] < 1.0 and info["params"]["gain_scale"] < 2.0
        assert obs[0] < 5.0
        _, _, _, _, info2 = wrapper.step([0.0, 0.0, 0.0, 0.0])
        assert info2["step"] == 2 and info2["success"] is False
    finally:
        wrapper.close()
    # the action list the reward may scribble on is not the prev_action of the next step
    seen = []

    def scribbler(obs, action, prev_action, info):  # a test stand-in, not AI-written code
        seen.append(list(prev_action))
        action[0] = 9.0
        return 0.0

    wrapper = RewardWrapper(HandoffEnv(), scribbler)
    try:
        wrapper.reset(seed=4)
        wrapper.step([0.3, 0.0, -0.2, 0.0])
        wrapper.step([0.0, 0.0, 0.0, 0.0])
        assert seen == [[0.0, 0.0, 0.0, 0.0], [0.3, 0.0, -0.2, 0.0]]
    finally:
        wrapper.close()
    original = {"a": [1, 2], "b": {"c": [3], "d": 4}, "e": 5}
    copied = copy_info(original)
    copied["a"].append(9)
    copied["b"]["c"].append(9)
    copied["b"]["d"] = 9
    copied["e"] = 9
    assert original == {"a": [1, 2], "b": {"c": [3], "d": 4}, "e": 5}


def test_reward_wrapper_rejects_non_numeric_and_raising_rewards():
    from rl.train import RewardWrapper

    def check(fn, message):
        wrapper = RewardWrapper(HandoffEnv(), fn)
        try:
            wrapper.reset(seed=0)
            with pytest.raises(ValueError, match=message):
                wrapper.step(np.zeros(4))
        finally:
            wrapper.close()

    check(lambda obs, action, prev_action, info: "0.5", "not a number")
    check(lambda obs, action, prev_action, info: None, "not a number")
    check(lambda obs, action, prev_action, info: [0.1], "not a number")
    check(lambda obs, action, prev_action, info: 1 / 0, "raised ZeroDivisionError")
    check(lambda obs, action, prev_action, info: info["missing"], "raised KeyError")
    exits = "import math\n\ndef reward(obs, action, prev_action, info):\n    raise SystemExit(0)\n"
    check(load_entry(exits, entry="reward"), "raised SystemExit")
    # a numpy scalar and an int are numbers
    wrapper = RewardWrapper(HandoffEnv(), lambda obs, action, prev_action, info: np.float32(0.25))
    try:
        wrapper.reset(seed=0)
        assert wrapper.step(np.zeros(4))[1] == 0.25
    finally:
        wrapper.close()


def test_train_validates_its_arguments_and_writes_null_eval_keys(tmp_path):
    from rl import evaluate, train

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="seed"):
        train.train(BASELINE, 64, out, n_envs=1, seed=-1)
    with pytest.raises(ValueError, match="eval_episodes"):
        train.train(BASELINE, 64, out, n_envs=1, eval_episodes=-3)
    with pytest.raises(ValueError, match="seed"):
        evaluate.run_episodes(None, 1, seed=-1)
    with pytest.raises(ValueError, match="episodes"):
        evaluate.run_episodes(None, 0)
    assert not out.exists()  # nothing was written before the arguments were checked
    summary = train.train(BASELINE, 64, out, n_envs=1, eval_episodes=0)  # one tiny update, no evaluation
    assert set(evaluate.RESULT_KEYS) <= set(summary)
    assert summary["episodes"] == 0 and summary["success_rate"] is None and summary["mean_final_dist"] is None
    written = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert written["success_rate"] is None and written["episodes"] == 0 and written["steps"] >= 64
    assert (out / "policy.zip").is_file()
    real = evaluate.run_episodes(evaluate.load_policy(out / "policy.zip"), 1)
    assert set(real) == set(evaluate.RESULT_KEYS) == set(evaluate.empty_result())


@pytest.mark.parametrize("n_envs", [1, 2])
def test_train_cli_reports_a_raising_reward_as_an_error(tmp_path, n_envs):
    """A reward that raises at call time ends the run with one `error:` line and exit 2, under DummyVecEnv
    and under SubprocVecEnv (where the worker dies), without a hang or a second traceback from the cleanup."""
    bad = tmp_path / "raises.py"
    src = "import math\n\ndef reward(obs, action, prev_action, info):\n    return 1 / 0\n"
    bad.write_text(src, encoding="utf-8")
    out = tmp_path / f"out{n_envs}"
    cmd = [sys.executable, "-m", "rl.train", "--reward", str(bad), "--steps", "128", "--n-envs", str(n_envs)]
    cmd += ["--out", str(out), "--eval-episodes", "1"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 2, proc.stderr[-3000:]
    last = proc.stderr.strip().splitlines()[-1]
    assert last.startswith("error: "), last
    assert "ZeroDivisionError" in proc.stderr
    assert proc.stdout.strip() == "" and not (out / "policy.zip").exists()


# ------------------------------------------------------------------ literal pins (final hardening)

LITERAL_BOX = ((0.20, 0.35), (-0.15, 0.15), (0.06, 0.12))  # x, y, z (lo, hi) in m, straight from D1
FAR = (0.35, 0.15, 0.12)  # a box corner about 0.32 m from the home tip: never a success, never a fall


def settle(env, steps=5):
    """Step the env with zero actions so the servos sit still, then return the true tip."""
    for _ in range(steps):
        env.step(np.zeros(4))
    return np.array(ik.fk_tip(env.joint_angles()))


def test_d1_constants_are_the_literal_task_numbers():
    assert he.EPISODE_STEPS == 150
    assert he.SUCCESS_RADIUS == 0.03
    assert he.SUCCESS_STEPS == 10
    assert he.FALL_Z == 0.03
    assert he.FALL_PENALTY == -50.0
    assert he.TARGET_BOX == LITERAL_BOX
    assert he.MAX_STEP_RAD == math.radians(2.0)
    assert he.MARGIN_RAD == math.radians(3.0)
    assert he.HZ == 30 and he.DT == 1.0 / 30
    assert he.OBS_SIZE == 16 and he.ACTION_SIZE == 4
    assert he.DRIFT_PROB == 0.3 and he.DRIFT_SPEED == (0.01, 0.05)
    assert he.OFFSET_MAX == math.radians(2.0) and he.ANGLE_NOISE == math.radians(0.5)
    assert he.POS_NOISE == 0.001 and he.DELAY_MAX == 2
    assert he.GAIN_RANGE == (0.8, 1.2) and he.LAG_RANGE == (0.1, 0.5)


def test_targets_are_sampled_across_the_literal_box(env):
    targets = np.array([env.reset(seed=seed)[1]["target"] for seed in range(300)])
    for axis, (lo, hi) in enumerate(LITERAL_BOX):
        assert np.all(targets[:, axis] >= lo) and np.all(targets[:, axis] <= hi)
        assert targets[:, axis].min() < lo + 0.01 and targets[:, axis].max() > hi - 0.01


def test_an_episode_truncates_at_exactly_150_steps(env):
    env.reset(seed=3)
    env.set_target(FAR)
    for step in range(1, 150):
        _, _, terminated, truncated, info = env.step(np.zeros(4))
        assert not terminated and not truncated and info["step"] == step
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert truncated and not terminated and info["step"] == 150
    assert info["success"] is False and info["fell"] is False and info["dist"] > 0.1


def test_success_radius_is_the_literal_three_centimeters(env):
    env.reset(seed=7)
    tip = settle(env)
    env.set_target(tip + np.array([0.0, 0.029, 0.0]))  # 1 mm inside the radius
    for step in range(1, 11):
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert info["streak"] == step and terminated is (step == 10), step
    assert info["success"] is True and not truncated and reward > 0.9
    env.reset(seed=7)
    tip = settle(env)
    env.set_target(tip + np.array([0.0, 0.031, 0.0]))  # 1 mm outside
    for _ in range(12):
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert info["streak"] == 0 and not terminated and reward < 0.0
    assert info["success"] is False


def test_success_needs_ten_consecutive_steps_and_leaving_the_radius_restarts_the_count(env):
    env.reset(seed=2)
    q, ok = ik.solve((0.25, 0.0, 0.10), -math.pi / 2)  # a tip inside the box, so the drift never bounces
    assert ok
    env.set_joints(q)
    tip = settle(env)
    assert all(lo < v < hi for v, (lo, hi) in zip(tip, LITERAL_BOX, strict=True))
    # 2.75 cm away and drifting 1 mm per step further: inside, inside, outside, outside
    env.set_target(tip + np.array([0.0, 0.0275, 0.0]), velocity=[0.0, 0.03, 0.0])
    streaks = [env.step(np.zeros(4))[4]["streak"] for _ in range(4)]
    assert streaks == [1, 2, 0, 0]


def test_fall_line_is_the_literal_three_centimeters(env):
    for z, fell in ((0.028, True), (0.032, False)):
        env.reset(seed=11)
        q, ok = ik.solve((0.22, 0.0, z), -math.pi / 2)
        assert ok
        env.set_joints(q)
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert abs(info["tip"][2] - z) < 0.001
        assert info["fell"] is fell and terminated is fell and not truncated
        assert (reward < -49.0) is fell


def test_fall_penalty_is_the_literal_minus_fifty():
    from rl.train import RewardWrapper

    assert he.baseline_reward(0.10, np.zeros(4), np.zeros(4), fell=True) == pytest.approx(-50.10)
    assert he.baseline_reward(0.10, np.zeros(4), np.zeros(4), fell=False) == pytest.approx(-0.10)
    wrapper = RewardWrapper(HandoffEnv(), lambda obs, action, prev_action, info: 2.0)
    try:
        wrapper.reset(seed=11)
        q, ok = ik.solve((0.22, 0.0, 0.02), -math.pi / 2)
        assert ok
        wrapper.env.set_joints(q)
        _, reward, terminated, _, info = wrapper.step(np.zeros(4))
        assert info["fell"] and terminated and reward == -48.0
    finally:
        wrapper.close()


def test_actions_outside_minus_one_to_one_are_clipped_by_the_env(env):
    env.reset(seed=4)
    cmd0 = np.array(env.command())
    obs, _, _, _, _ = env.step([2.5, -7.0, 0.0, 0.0])
    delta = np.array(env.command()) - cmd0
    assert abs(delta[0] - math.radians(2.0)) < 1e-12 and abs(delta[1] + math.radians(2.0)) < 1e-12
    assert np.allclose(delta[2:], 0.0)
    assert np.allclose(obs[he.OBS_LAYOUT["prev_action"]], [1.0, -1.0, 0.0, 0.0])
    obs, _, _, _, _ = env.step(np.zeros(4))
    assert np.all(obs[he.OBS_LAYOUT["prev_action"]] == 0.0)


# `memoryview` is removed from the restricted builtins but is not a name check_source rejects, so a reward
# file can probe the namespace it runs in; `open` cannot be named by a reward, so the test inspects it below.
RESTRICTED_PROBE = (
    "import math\n\n"
    "def reward(obs, action, prev_action, info):\n"
    "    try:\n"
    "        memoryview\n"
    "    except NameError:\n"
    "        return 1.0\n"
    "    return -1.0\n"
)
ACTION_PROBE = (
    "import math\n\ndef reward(obs, action, prev_action, info):\n    return action[0] + 0.5 * action[1]\n"
)


def test_env_factory_loads_the_reward_through_the_restricted_namespace(tmp_path):
    from rl.train import EnvFactory, RewardWrapper

    path = tmp_path / "probe.py"
    path.write_text(RESTRICTED_PROBE, encoding="utf-8")
    factory = EnvFactory(path, seed=3)
    assert factory.reward_path == str(path) and factory.seed == 3
    copied = pickle.loads(pickle.dumps(factory))  # SubprocVecEnv ships the factory, never a function
    assert copied.reward_path == str(path) and copied.seed == 3
    env = factory()
    try:
        assert isinstance(env, Monitor) and isinstance(env.env, RewardWrapper)
        assert isinstance(env.unwrapped, HandoffEnv)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(4))
        assert reward == 1.0 and info["fell"] is False  # memoryview raised NameError: the trimmed builtins
        fn = env.env._reward_fn
        assert fn.__module__ == "candidate"  # defined by exec in sim.restricted.namespace(), never imported
        seen = fn.__globals__["__builtins__"]
        assert isinstance(seen, dict)
        assert "open" not in seen and "memoryview" not in seen and "exec" not in seen and "type" not in seen
        assert seen["__import__"] is not builtins.__import__ and "len" in seen and "range" in seen
        assert fn.__globals__["math"] is math and fn.__globals__["__name__"] == "candidate"
    finally:
        env.close()
    # the file is read, checked and loaded by the call, not by the constructor
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import os\n\ndef reward(obs, action, prev_action, info):\n    return 0.0\n", encoding="utf-8"
    )
    lazy = EnvFactory(bad)
    with pytest.raises(ValueError, match="rejected"):
        lazy()


def test_env_factory_env_clips_actions_before_the_reward_and_the_arm(tmp_path):
    from rl.train import EnvFactory

    path = tmp_path / "action.py"
    path.write_text(ACTION_PROBE, encoding="utf-8")
    env = EnvFactory(path)()
    try:
        env.reset(seed=0)
        cmd0 = np.array(env.unwrapped.command())
        obs, reward, _, _, _ = env.step([2.5, -7.0, 0.0, 0.0])
        assert reward == 0.5  # 1.0 + 0.5 * -1.0: the reward sees the clipped action
        delta = np.array(env.unwrapped.command()) - cmd0
        assert abs(delta[0] - math.radians(2.0)) < 1e-12 and abs(delta[1] + math.radians(2.0)) < 1e-12
        assert np.allclose(obs[he.OBS_LAYOUT["prev_action"]], [1.0, -1.0, 0.0, 0.0])
        _, reward, _, _, _ = env.step([0.25, 0.5, 0.0, 0.0])
        assert reward == 0.5  # in range: untouched
    finally:
        env.close()


class TipTargetEnv(HandoffEnv):
    """A test stand-in: after every reset the target sits on the tip (or at FAR, per the `on_tip` plan,
    its last entry repeating), so a zero-action policy succeeds at exactly step 10 or runs the episode out."""

    def __init__(self, on_tip=(True,)):
        super().__init__()
        self.on_tip = list(on_tip)
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        on_tip = self.on_tip[min(self.resets, len(self.on_tip) - 1)]
        self.resets += 1
        self.set_target(ik.fk_tip(self.joint_angles()) if on_tip else FAR)
        return obs, info


class ZeroActionTipEnv(TipTargetEnv):
    """TipTargetEnv that also ignores the policy's action, so PPO's random rollouts succeed as well."""

    def step(self, action):
        return super().step(np.zeros(4))


class ZeroPolicy:
    """`predict` like a PPO model: always the zero action."""

    def predict(self, obs, deterministic=True):
        return np.zeros(4, dtype=np.float32), None


def test_run_episodes_counts_a_real_success():
    from rl.evaluate import RESULT_KEYS, run_episodes

    env = TipTargetEnv()
    try:
        result = run_episodes(ZeroPolicy(), 1, seed=0, env=env)
        assert set(result) == set(RESULT_KEYS)
        assert result["episodes"] == 1 and result["successes"] == 1 and result["success_rate"] == 1.0
        assert result["mean_steps_to_success"] == 10.0 and result["falls"] == 0
        assert 0.0 <= result["mean_final_dist"] < 0.03
    finally:
        env.close()
    env = TipTargetEnv(on_tip=(True, False, True))
    try:
        result = run_episodes(ZeroPolicy(), 3, seed=5, env=env)
        assert result["episodes"] == 3 and result["successes"] == 2 and result["falls"] == 0
        assert result["success_rate"] == pytest.approx(2 / 3)
        assert result["mean_steps_to_success"] == 10.0  # the 150-step miss does not count
        assert 0.09 < result["mean_final_dist"] < 0.12  # (~0 + ~0.32 + ~0) / 3
        assert env.resets == 3
    finally:
        env.close()


def test_progress_callback_counts_a_real_success(tmp_path):
    """Drive ProgressCallback through its public hooks with the infos of a real success and a real miss."""
    from rl.train import PROGRESS_COLUMNS, ProgressCallback

    env = TipTargetEnv(on_tip=(True, False))
    callback = ProgressCallback(tmp_path / "progress.csv", every=4)
    model = types.SimpleNamespace(num_timesteps=0)
    callback.model = model
    callback.on_training_start({}, {})
    totals = []
    try:
        for _episode in range(2):
            env.reset(seed=0)
            total = 0.0
            done = False
            while not done:
                _, reward, terminated, truncated, info = env.step(np.zeros(4))
                total += reward
                done = terminated or truncated
                info = dict(info)
                if done:
                    info["episode"] = {"r": total, "l": info["step"], "t": 0.0}  # what Monitor adds
                model.num_timesteps += 1
                callback.update_locals({"infos": [info], "dones": [done]})
                assert callback.on_step() is True
            totals.append(total)
            assert info["fell"] is False
        callback.on_training_end()
    finally:
        env.close()
    assert model.num_timesteps == 160 and totals[0] > 9.0 and totals[1] < -40.0
    with (tmp_path / "progress.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0]) == PROGRESS_COLUMNS == ("step", "mean_reward", "success_rate")
    steps = [int(row["step"]) for row in rows]
    assert steps[:3] == [4, 8, 12] and steps[-1] == 160 and steps == sorted(steps)
    assert rows[0]["success_rate"] == "nan" and rows[0]["mean_reward"] == "nan"  # no episode done yet
    by_step = {int(row["step"]): row for row in rows}
    assert float(by_step[12]["success_rate"]) == 1.0  # the first episode, a success, ended at step 10
    assert float(by_step[12]["mean_reward"]) == pytest.approx(totals[0])
    assert float(by_step[160]["success_rate"]) == 0.5
    assert float(by_step[160]["mean_reward"]) == pytest.approx(sum(totals) / 2)
    assert callback.rows[-1] == {
        "step": 160,
        "mean_reward": pytest.approx(sum(totals) / 2),
        "success_rate": 0.5,
    }


def test_train_reports_real_successes_in_progress_and_summary(tmp_path, monkeypatch):
    from rl import evaluate, train

    monkeypatch.setattr(train, "HandoffEnv", ZeroActionTipEnv)
    out = tmp_path / "out"
    summary = train.train(BASELINE, 64, out, n_envs=1, eval_episodes=0)
    final = summary["final_progress"]
    assert final["step"] == 64 and final["success_rate"] == 1.0 and 8.0 < final["mean_reward"] <= 10.0
    with (out / "progress.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[-1]["success_rate"]) == 1.0 and int(rows[-1]["step"]) == 64
    result = evaluate.run_episodes(evaluate.load_policy(out / "policy.zip"), 2, env=ZeroActionTipEnv())
    assert result["episodes"] == 2 and result["successes"] == 2 and result["success_rate"] == 1.0
    assert result["falls"] == 0 and result["mean_steps_to_success"] == 10.0
