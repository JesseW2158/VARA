"""The dashboard video must contain a browser-supported stream, not merely decode in OpenCV."""

import av
import numpy as np
import pytest

from rl import render_video as video


class Scene:
    def __init__(self, **kwargs):
        self.frames = 0

    def reset(self, **kwargs):
        return np.zeros(16), {"step": 0, "dist": 0.1}

    def target(self):
        return np.zeros(3)

    def set_target(self, *args, **kwargs):
        pass

    def render(self):
        self.frames += 1
        return np.full((480, 640, 3), self.frames * 20, dtype=np.uint8)

    def step(self, action):
        return np.zeros(16), 0, False, False, {"step": self.frames, "dist": 0.1}

    def close(self):
        pass


class Policy:
    def predict(self, obs, **kwargs):
        return np.zeros(4), None


def test_dashboard_video_has_h264_frames_and_correct_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "HandoffEnv", Scene)
    out = tmp_path / "clip.mp4"
    result = video.render_video("unused", out, policy=Policy(), seconds=0.2)
    with av.open(out) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
        frames = list(container.decode(video=0))
        assert len(frames) == result["frames"] == 6
        assert float(stream.duration * stream.time_base) == pytest.approx(0.2)
        assert (frames[0].width, frames[0].height) == (640, 480)
    assert not list(tmp_path.glob(".render-*"))


def test_failed_render_preserves_previous_dashboard_video(tmp_path, monkeypatch):
    class BrokenScene(Scene):
        def render(self):
            if self.frames:
                raise RuntimeError("lost renderer")
            return super().render()

    monkeypatch.setattr(video, "HandoffEnv", BrokenScene)
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"previous video")
    with pytest.raises(RuntimeError, match="lost renderer"):
        video.render_video("unused", out, policy=Policy(), seconds=0.2)
    assert out.read_bytes() == b"previous video"
    assert not list(tmp_path.glob(".render-*"))


@pytest.mark.parametrize(
    "kwargs", [{"seconds": float("inf")}, {"fps": 1.5}, {"drift": float("nan")}, {"seed": -1}]
)
def test_bad_video_parameters_fail_before_policy_or_renderer_access(tmp_path, kwargs):
    with pytest.raises(ValueError):
        video.render_video("missing-policy.zip", tmp_path / "clip.mp4", **kwargs)
