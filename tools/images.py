"""Read the pixels of an uploaded photo. No LLM.

The colours of a person's material are a measurement, not an opinion, so they are
worked out here from the pixels. A model is good at saying what a picture *feels*
like, and unreliable at reporting an exact colour code.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

#: The longest side a photo is kept at. Plenty for a model to see a face and a
#: room, and small enough to send as a reference without hitting size limits.
MAX_SIDE = 1024

#: Photos are sampled at this size to find their colours: fast, and detail
#: finer than this does not change which colours dominate.
_SAMPLE = 64

#: Colours closer than this (in RGB distance) count as one colour.
_MERGE_DISTANCE = 56


def normalize_image(src: Path, dst: Path, max_side: int = MAX_SIDE) -> Path:
    """Write a clean, upright, reasonably sized JPEG copy of a photo.

    Phones store a photo sideways and record the turn in metadata. A model that
    ignores the metadata sees a sideways person, so the turn is applied here.
    Transparency is flattened onto white, because JPEG has none.

    Args:
        src: The uploaded image, in any format Pillow reads.
        dst: Where to write the JPEG.
        max_side: The longest side the copy may have.

    Returns:
        ``dst``.

    Raises:
        ValueError: If the file is not an image.
    """
    try:
        with Image.open(src) as opened:
            image = ImageOps.exif_transpose(opened)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                flat = Image.new("RGB", image.size, (255, 255, 255))
                flat.paste(image, mask=image.getchannel("A"))
                image = flat
            else:
                image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.save(dst, "JPEG", quality=88)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{src.name} is not an image I can read: {exc}") from exc
    return dst


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def dominant_colors(paths: list[Path], count: int = 5) -> list[str]:
    """The colours that dominate a set of images, most common first.

    Every image is reduced to a handful of colours, the colours are pooled across
    images by how much of the picture they fill, and near-identical shades are
    merged so five results are five different colours and not five browns.

    Args:
        paths: Image files. Ones that cannot be read are skipped.
        count: How many colours to return at most.

    Returns:
        ``#RRGGBB`` strings. Empty if no image could be read.
    """
    weight: dict[tuple[int, int, int], float] = defaultdict(float)
    for path in paths:
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image = image.resize((_SAMPLE, _SAMPLE))
        except (OSError, ValueError):
            logger.warning("Could not read colours from %s", path.name)
            continue
        quantized = image.quantize(colors=10, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette() or []
        total = _SAMPLE * _SAMPLE
        for pixels, index in quantized.getcolors() or []:
            colour = tuple(palette[index * 3 : index * 3 + 3])
            if len(colour) == 3:
                # Every image counts equally, however many pixels it has.
                weight[colour] += pixels / total  # type: ignore[index]

    chosen: list[tuple[float, float, float]] = []
    for colour, _ in sorted(weight.items(), key=lambda kv: kv[1], reverse=True):
        if all(_distance(colour, other) > _MERGE_DISTANCE for other in chosen):
            chosen.append(colour)  # type: ignore[arg-type]
        if len(chosen) == count:
            break
    return [f"#{int(r):02X}{int(g):02X}{int(b):02X}" for r, g, b in chosen]
