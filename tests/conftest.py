"""Shared helper for the tests: makes small fake images (no dataset needed)."""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_image_bytes():
    def _make(mode="RGB", size=(300, 200), file_format="PNG"):
        """Return the file content of a random image in the given colour mode."""
        rng = np.random.default_rng(0)
        pixels = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(pixels).convert(mode)
        buffer = BytesIO()
        image.save(buffer, format=file_format)
        return buffer.getvalue()
    return _make