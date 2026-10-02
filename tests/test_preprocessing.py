"""Tests for the preprocessing step."""
import numpy as np
import pytest

from src import config
from src.data.preprocessing import preprocess_bytes


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_every_colour_mode_gives_the_same_shape(make_image_bytes, mode):
    image = preprocess_bytes(make_image_bytes(mode)).numpy()
    assert image.shape == (*config.IMAGE_SIZE, 3)
    assert image.dtype == np.float32


def test_jpeg_files_work_too(make_image_bytes):
    image = preprocess_bytes(make_image_bytes("RGB", file_format="JPEG")).numpy()
    assert image.shape == (*config.IMAGE_SIZE, 3)


def test_pixel_values_stay_in_0_255(make_image_bytes):
    image = preprocess_bytes(make_image_bytes("RGB")).numpy()
    assert image.min() >= -0.01 and image.max() <= 255.01   # tiny rounding from resizing


def test_three_channels_are_identical(make_image_bytes):
    image = preprocess_bytes(make_image_bytes("RGB")).numpy()
    assert np.allclose(image[..., 0], image[..., 1])
    assert np.allclose(image[..., 1], image[..., 2])