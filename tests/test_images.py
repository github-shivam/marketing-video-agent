"""Reading an uploaded photo: straightening it, shrinking it, and measuring its colours."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools.images import MAX_SIDE, dominant_colors, normalize_image


def save(path: Path, image: Image.Image, **kwargs) -> Path:
    image.save(path, **kwargs)
    return path


def solid(path: Path, colour: tuple[int, int, int], size=(120, 80)) -> Path:
    return save(path, Image.new("RGB", size, colour), format="PNG")


def close(hex_colour: str, expected: tuple[int, int, int], tolerance: int = 12) -> bool:
    got = tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return all(abs(a - b) <= tolerance for a, b in zip(got, expected))


class TestNormalizeImage:
    def test_a_sideways_phone_photo_is_turned_upright(self, tmp_path):
        """Phones record the turn in metadata. A model that ignores it sees a
        person lying down."""
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
        src = save(tmp_path / "p.jpg", Image.new("RGB", (200, 100), "red"), format="JPEG", exif=exif)

        out = normalize_image(src, tmp_path / "out.jpg")

        with Image.open(out) as image:
            assert image.size == (100, 200), "landscape on disk, portrait on screen"

    def test_a_large_photo_is_shrunk_to_the_longest_side(self, tmp_path):
        src = save(tmp_path / "big.png", Image.new("RGB", (3000, 2000), "blue"))

        out = normalize_image(src, tmp_path / "out.jpg")

        with Image.open(out) as image:
            assert max(image.size) == MAX_SIDE
            assert image.size == (1024, 683), "the proportions are kept"

    def test_a_small_photo_is_not_enlarged(self, tmp_path):
        src = solid(tmp_path / "small.png", (10, 20, 30), size=(300, 200))
        with Image.open(normalize_image(src, tmp_path / "out.jpg")) as image:
            assert image.size == (300, 200)

    def test_the_result_is_a_jpeg(self, tmp_path):
        src = solid(tmp_path / "a.png", (1, 2, 3))
        with Image.open(normalize_image(src, tmp_path / "out.jpg")) as image:
            assert image.format == "JPEG" and image.mode == "RGB"

    def test_transparency_is_flattened_onto_white_not_black(self, tmp_path):
        """JPEG has no transparency; the usual failure is a black box."""
        src = save(tmp_path / "t.png", Image.new("RGBA", (20, 20), (0, 0, 0, 0)))
        with Image.open(normalize_image(src, tmp_path / "out.jpg")) as image:
            assert all(channel > 240 for channel in image.getpixel((10, 10)))

    def test_the_folder_is_created(self, tmp_path):
        src = solid(tmp_path / "a.png", (1, 2, 3))
        out = normalize_image(src, tmp_path / "deep" / "er" / "out.jpg")
        assert out.exists()

    def test_a_file_that_is_not_an_image_is_a_clear_error(self, tmp_path):
        bad = tmp_path / "notes.jpg"
        bad.write_text("this is not a picture")

        with pytest.raises(ValueError, match="notes.jpg is not an image"):
            normalize_image(bad, tmp_path / "out.jpg")


class TestDominantColors:
    def test_a_single_colour_image_gives_that_colour(self, tmp_path):
        (only,) = dominant_colors([solid(tmp_path / "a.png", (200, 30, 40))])
        assert close(only, (200, 30, 40))

    def test_colours_are_returned_as_hex(self, tmp_path):
        (only,) = dominant_colors([solid(tmp_path / "a.png", (255, 0, 0))])
        assert only == only.upper() and len(only) == 7 and only.startswith("#")

    def test_the_colour_that_fills_more_of_the_picture_comes_first(self, tmp_path):
        image = Image.new("RGB", (100, 100), (0, 0, 255))
        image.paste((255, 0, 0), (0, 0, 100, 25))  # a quarter red, three quarters blue
        colours = dominant_colors([save(tmp_path / "a.png", image)])

        assert close(colours[0], (0, 0, 255)) and close(colours[1], (255, 0, 0))

    def test_near_identical_shades_count_as_one_colour(self, tmp_path):
        image = Image.new("RGB", (100, 100), (200, 30, 40))
        image.paste((205, 33, 44), (0, 0, 100, 50))  # a barely different red
        colours = dominant_colors([save(tmp_path / "a.png", image)])

        assert len(colours) == 1, f"five browns are not a palette: {colours}"

    def test_colours_are_pooled_across_images(self, tmp_path):
        paths = [
            solid(tmp_path / "a.png", (255, 0, 0)),
            solid(tmp_path / "b.png", (0, 255, 0)),
            solid(tmp_path / "c.png", (0, 0, 255)),
        ]
        colours = dominant_colors(paths)

        assert len(colours) == 3
        assert any(close(c, (255, 0, 0)) for c in colours)
        assert any(close(c, (0, 255, 0)) for c in colours)
        assert any(close(c, (0, 0, 255)) for c in colours)

    def test_every_image_counts_equally_whatever_its_size(self, tmp_path):
        big = solid(tmp_path / "big.png", (255, 0, 0), size=(2000, 2000))
        small = solid(tmp_path / "small.png", (0, 0, 255), size=(40, 40))

        colours = dominant_colors([big, small], count=2)

        assert len(colours) == 2, "the small image's colour is not drowned out"

    def test_at_most_count_colours(self, tmp_path):
        paths = [solid(tmp_path / f"{i}.png", (i * 40, 255 - i * 40, 100)) for i in range(6)]
        assert len(dominant_colors(paths, count=3)) <= 3

    def test_an_unreadable_file_is_skipped_not_fatal(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_text("nope")
        colours = dominant_colors([bad, solid(tmp_path / "ok.png", (10, 200, 10))])

        assert len(colours) == 1 and close(colours[0], (10, 200, 10))

    def test_nothing_readable_gives_no_colours(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_text("nope")
        assert dominant_colors([bad]) == []
        assert dominant_colors([]) == []

    def test_a_sideways_photo_is_measured_the_same(self, tmp_path):
        """Colours do not depend on the metadata turn."""
        exif = Image.Exif()
        exif[0x0112] = 6
        path = save(tmp_path / "s.jpg", Image.new("RGB", (80, 40), (30, 90, 200)), format="JPEG", exif=exif)
        (only,) = dominant_colors([path])
        assert close(only, (30, 90, 200), tolerance=20)
