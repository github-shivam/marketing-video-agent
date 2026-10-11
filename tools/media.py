"""Trim a finished video to the length that was asked for. No LLM.

Veo makes fixed lengths (a 4, 6 or 8 second base clip, then 7 second extensions),
so a video of 20 seconds is generated as 22 and cut back to 20. The screenplay
is written to finish inside the kept seconds, so the cut removes only the part
nobody planned.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

#: The audio fades over this long before the cut, so it does not end on a click.
FADE_SECONDS = 0.25


async def trim_video(path: Path, seconds: int) -> bool:
    """Cut ``path`` down to ``seconds``, in place.

    The video is re-encoded rather than stream-copied, because a copy can only
    cut on a keyframe and would miss the requested length by a second or more.

    Args:
        path: The MP4 to shorten. It is replaced only if the cut succeeds.
        seconds: The length to keep.

    Returns:
        True if the file now ends at ``seconds``. False if it could not be cut
        (no ffmpeg, or ffmpeg failed); the file is then left untouched, still
        longer than asked, and the caller should say so.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found - cannot trim %s to %ds", path.name, seconds)
        return False

    trimmed = path.with_name(f"{path.stem}.trim{path.suffix}")
    fade_from = max(0.0, seconds - FADE_SECONDS)
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-y",
        "-i", str(path),
        "-t", str(seconds),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-af", f"afade=t=out:st={fade_from}:d={FADE_SECONDS}",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        str(trimmed),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0 or not trimmed.exists() or trimmed.stat().st_size == 0:
        logger.warning(
            "ffmpeg could not trim %s to %ds: %s",
            path.name, seconds, stderr.decode(errors="replace")[-300:],
        )
        trimmed.unlink(missing_ok=True)
        return False

    os.replace(trimmed, path)
    logger.info("Trimmed %s to %ds", path.name, seconds)
    return True


#: Share of the size limit the video is aimed at. Bitrate control is approximate,
#: so aim under the line rather than at it.
_TARGET_SHARE = 0.9

#: Audio keeps this much of the budget; the picture gets the rest.
_AUDIO_BITS_PER_SECOND = 96_000

#: Below this the picture is not worth sending.
_MIN_VIDEO_BITS_PER_SECOND = 150_000


async def _encode_to_bitrate(
    ffmpeg: str, source: Path, target: Path, video_bits_per_second: int
) -> bool:
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-y",
        "-i", str(source),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-b:v", str(video_bits_per_second),
        "-maxrate", str(int(video_bits_per_second * 1.3)),
        "-bufsize", str(video_bits_per_second * 2),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", str(_AUDIO_BITS_PER_SECOND),
        "-movflags", "+faststart",
        str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        logger.warning("ffmpeg could not shrink %s: %s", source.name, stderr.decode(errors="replace")[-300:])
        return False
    return target.exists() and target.stat().st_size > 0


async def shrink_to_fit(path: Path, max_bytes: int, seconds: float) -> Path:
    """Return a copy of ``path`` that fits in ``max_bytes``, or ``path`` itself.

    WhatsApp rejects a video over 16 MB, and a 30-second reel can come close to
    that. The original is never touched: it stays full quality for Instagram, and
    only the copy sent into a chat is compressed.

    Args:
        path: The finished video.
        max_bytes: The size it must fit in.
        seconds: Its length, to work out how much picture each second can afford.

    Returns:
        ``path`` when it already fits. Otherwise a smaller copy beside it. If no
        copy can be made to fit (no ffmpeg, or it fails) the original comes back
        and the caller has to deal with it being too big.
    """
    if not path.is_file() or path.stat().st_size <= max_bytes:
        return path  # a missing file is for the caller to report, not for us

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or seconds <= 0:
        logger.warning("Cannot shrink %s: no ffmpeg, or no length to aim for", path.name)
        return path

    smaller = path.with_name(f"{path.stem}.chat{path.suffix}")
    budget = max_bytes * 8 * _TARGET_SHARE / seconds - _AUDIO_BITS_PER_SECOND
    video_bits = max(_MIN_VIDEO_BITS_PER_SECOND, int(budget))

    # Bitrate control is approximate, so check the result and, if it is still
    # over, try once more at a lower rate.
    for attempt in (1, 2):
        if await _encode_to_bitrate(ffmpeg, path, smaller, video_bits) and (
            smaller.stat().st_size <= max_bytes
        ):
            logger.info(
                "Shrank %s from %.1f to %.1f MB for the chat",
                path.name, path.stat().st_size / 1e6, smaller.stat().st_size / 1e6,
            )
            return smaller
        video_bits = max(_MIN_VIDEO_BITS_PER_SECOND, int(video_bits * 0.7))

    smaller.unlink(missing_ok=True)
    logger.warning("Could not shrink %s under %d bytes", path.name, max_bytes)
    return path


# ── Preparing uploads to be read ─────────────────────────────────────────

#: Uploads are cut down before a model sees them. A model reads a video a frame or
#: two a second, so full resolution only costs size, and a request has a size limit.
MAX_VIDEO_SECONDS = 60
MAX_AUDIO_SECONDS = 120


async def _run_ffmpeg(*args: str) -> tuple[bool, str]:
    """Run ffmpeg and say whether it succeeded, and what it complained about."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg is not installed"
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-v", "error", "-y", *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    return process.returncode == 0, stderr.decode(errors="replace")[-300:]


async def probe_seconds(path: Path) -> float | None:
    """The length of a media file, or None if it cannot be read."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    process = await asyncio.create_subprocess_exec(
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await process.communicate()
    try:
        return float(out.decode().strip())
    except ValueError:
        return None


async def extract_frames(video: Path, out_dir: Path, count: int = 3) -> list[Path]:
    """Stills from evenly spaced moments of a video, as JPEGs.

    Taken away from the very start and end, which are often a fade or a title.

    Args:
        video: The video file.
        out_dir: Where to write ``frame-1.jpg``, ``frame-2.jpg``, ...
        count: How many stills.

    Returns:
        The stills that were made, possibly fewer than asked for.
    """
    length = await probe_seconds(video)
    if not length or length <= 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for i in range(1, count + 1):
        moment = length * i / (count + 1)
        frame = out_dir / f"frame-{i}.jpg"
        ok, _ = await _run_ffmpeg(
            "-ss", f"{moment:.2f}", "-i", str(video), "-frames:v", "1",
            "-vf", "scale='min(1024,iw)':-2", "-q:v", "3", str(frame),
        )
        if ok and frame.exists() and frame.stat().st_size > 0:
            frames.append(frame)
    return frames


async def shrink_video_for_reading(src: Path, dst: Path) -> bool:
    """A small, low-frame-rate copy of a video, with its sound, for a model to read."""
    ok, why = await _run_ffmpeg(
        "-i", str(src), "-t", str(MAX_VIDEO_SECONDS),
        "-vf", "scale=-2:480,fps=2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "1", "-b:a", "48k", "-movflags", "+faststart", str(dst),
    )
    if not ok:
        logger.warning("Could not prepare %s for reading: %s", src.name, why)
    return ok


async def audio_for_reading(src: Path, dst: Path) -> bool:
    """Mono 16 kHz WAV of the sound in ``src``: small, and any model can decode it."""
    ok, why = await _run_ffmpeg(
        "-i", str(src), "-vn", "-t", str(MAX_AUDIO_SECONDS),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
    )
    if not ok:
        logger.warning("Could not prepare %s for reading: %s", src.name, why)
    return ok
