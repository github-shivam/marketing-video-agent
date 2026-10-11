"""Trimming a finished video to the length that was asked for, with real ffmpeg.

These run the real binary on real files and read the result back with ffprobe,
because "the file is N seconds long" is exactly the thing a mock cannot prove.
They are skipped where ffmpeg is not installed.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from agents.publisher import video_node
from providers.stub import StubVideoProvider
from tools.media import shrink_to_fit, trim_video

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="needs ffmpeg and ffprobe on PATH",
)


def probe(path: Path) -> dict:
    """Duration in seconds and the stream types of a media file."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height",
            "-of", "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    fields = [line.split("=", 1) for line in out.strip().splitlines()]
    return {
        "duration": float(next(v for k, v in fields if k == "duration")),
        "streams": [v for k, v in fields if k == "codec_type"],
        "size": [(k, int(v)) for k, v in fields if k in ("width", "height")],
    }


def make_clip(tmp_path: Path, seconds: int) -> Path:
    """A small real video with picture and sound, of exactly ``seconds``.

    Small and flat so that encoding it is cheap; trimming does not care what the
    picture is. The full-size placeholder is exercised separately, end to end.
    """
    clip = tmp_path / f"clip-{seconds}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x203040:s=180x320:r=24:d={seconds}",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:d={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-g", "48",  # a keyframe every 2s, so a stream copy could not cut exactly
            "-c:a", "aac", "-shortest", str(clip),
        ],
        check=True,
    )
    return clip


class TestTrimVideo:
    async def test_cuts_a_video_to_the_requested_length(self, settings, tmp_path):
        clip = make_clip(tmp_path, 22)
        assert probe(clip)["duration"] == pytest.approx(22, abs=0.2)

        assert await trim_video(clip, 20) is True

        assert probe(clip)["duration"] == pytest.approx(20, abs=0.15)

    async def test_the_file_is_replaced_in_place_and_nothing_is_left_behind(
        self, settings, tmp_path
    ):
        clip = make_clip(tmp_path, 12)
        await trim_video(clip, 10)

        assert clip.exists()
        assert not list(tmp_path.glob("*.trim.mp4")), "no temporary file left over"

    async def test_audio_and_picture_survive_the_cut(self, settings, tmp_path):
        clip = make_clip(tmp_path, 12)
        before = probe(clip)
        await trim_video(clip, 9)
        after = probe(clip)

        assert sorted(after["streams"]) == sorted(before["streams"]) == ["audio", "video"]
        assert after["size"] == before["size"], "720x1280 stays 720x1280"

    @pytest.mark.parametrize("seconds", [5, 11, 25])
    async def test_it_is_frame_accurate_not_keyframe_accurate(
        self, settings, tmp_path, seconds
    ):
        """A stream copy can only cut on a keyframe and misses by a second or more."""
        clip = make_clip(tmp_path, 29)
        await trim_video(clip, seconds)

        assert probe(clip)["duration"] == pytest.approx(seconds, abs=0.15)

    async def test_a_file_ffmpeg_cannot_read_is_left_untouched(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"this is not a video")

        assert await trim_video(broken, 5) is False

        assert broken.read_bytes() == b"this is not a video"
        assert not list(tmp_path.glob("*.trim.mp4"))

    async def test_without_ffmpeg_it_says_so_and_changes_nothing(
        self, settings, tmp_path, monkeypatch
    ):
        clip = make_clip(tmp_path, 12)
        original = clip.read_bytes()
        monkeypatch.setattr("tools.media.shutil.which", lambda _name: None)

        assert await trim_video(clip, 8) is False
        assert clip.read_bytes() == original


class TestTheVideoNodeEndToEnd:
    """The node, the stub provider and the trim together, on real files."""

    @pytest.mark.parametrize(
        ("asked", "made", "generated"),
        [(5, 5, 6), (20, 20, 22), (30, 29, 29)],
    )
    async def test_the_finished_file_is_as_long_as_the_plan_says(
        self, settings, tmp_path, asked, made, generated
    ):
        cfg = dataclasses.replace(settings, output_dir=tmp_path)
        result = await video_node(cfg, StubVideoProvider(cfg))(
            {"prompts": ["a", "b", "c", "d"], "duration_seconds": asked}
        )

        file = Path(result["media"].local_files[0])
        assert result["media"].duration_seconds == made
        assert probe(file)["duration"] == pytest.approx(made, abs=0.2), (
            f"asked for {asked}s, generated {generated}s, expected {made}s"
        )


def make_heavy_clip(tmp_path: Path, seconds: int, bitrate: str = "6M") -> Path:
    """A real clip at a deliberately high bitrate, so it is large for its length."""
    clip = tmp_path / "heavy.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=360x640:rate=24:duration={seconds}",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000:d={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-b:v", bitrate,
            "-minrate", bitrate, "-maxrate", bitrate, "-bufsize", "2M",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip),
        ],
        check=True,
    )
    return clip


class TestShrinkToFit:
    """WhatsApp takes at most 16 MB; a copy is made small enough for the chat."""

    async def test_a_file_that_fits_is_returned_as_it_is(self, tmp_path):
        clip = make_clip(tmp_path, 4)
        assert await shrink_to_fit(clip, 10_000_000, 4) == clip
        assert not list(tmp_path.glob("*.chat.mp4"))

    async def test_a_large_file_gets_a_smaller_copy_that_fits(self, tmp_path):
        clip = make_heavy_clip(tmp_path, 8)
        limit = clip.stat().st_size // 3
        assert clip.stat().st_size > limit

        small = await shrink_to_fit(clip, limit, 8)

        assert small != clip and small.exists()
        assert small.stat().st_size <= limit
        assert probe(small)["duration"] == pytest.approx(8, abs=0.3), "same length"
        assert sorted(probe(small)["streams"]) == ["audio", "video"]

    async def test_the_original_is_left_untouched_for_instagram(self, tmp_path):
        clip = make_heavy_clip(tmp_path, 8)
        before = clip.read_bytes()

        await shrink_to_fit(clip, clip.stat().st_size // 3, 8)

        assert clip.read_bytes() == before

    async def test_a_limit_that_cannot_be_met_returns_the_original(self, tmp_path):
        clip = make_heavy_clip(tmp_path, 8)

        assert await shrink_to_fit(clip, 1_000, 8) == clip
        assert not list(tmp_path.glob("*.chat.mp4")), "no half-made copy left over"

    async def test_without_ffmpeg_the_original_comes_back(self, tmp_path, monkeypatch):
        clip = make_heavy_clip(tmp_path, 4)
        monkeypatch.setattr("tools.media.shutil.which", lambda _name: None)

        assert await shrink_to_fit(clip, 1_000, 4) == clip

    async def test_a_missing_file_is_left_for_the_caller_to_report(self, tmp_path):
        ghost = tmp_path / "gone.mp4"
        assert await shrink_to_fit(ghost, 1_000, 8) == ghost
