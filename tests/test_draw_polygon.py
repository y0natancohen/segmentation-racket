"""Tests for draw_polygon_on_image — polygon overlay drawing utility."""

import numpy as np
import pytest

from segmentation.segmentation import draw_polygon_on_image


class TestDrawPolygonOnImage:
    """Unit tests for draw_polygon_on_image."""

    @pytest.fixture
    def blank_image(self):
        """A 200×300 black BGR image."""
        return np.zeros((200, 300, 3), dtype=np.uint8)

    @pytest.fixture
    def square_polygon(self):
        """A 100×100 square polygon centered in the image."""
        return np.array([
            [100, 50],
            [200, 50],
            [200, 150],
            [100, 150],
        ], dtype=np.float32)

    # ---- outline mode -------------------------------------------------------

    def test_outline_draws_on_image(self, blank_image, square_polygon):
        """Outline mode should add non-zero pixels along the polygon edges."""
        result = draw_polygon_on_image(blank_image, square_polygon, thickness=2)
        assert result.shape == blank_image.shape
        # Some pixels should have been drawn (not entirely black)
        assert np.any(result > 0)

    def test_outline_preserves_shape(self, blank_image, square_polygon):
        """Output image dimensions must match input."""
        result = draw_polygon_on_image(blank_image, square_polygon, thickness=2)
        assert result.shape == blank_image.shape
        assert result.dtype == blank_image.dtype

    # ---- fill mode -----------------------------------------------------------

    def test_fill_mode_blends(self, blank_image, square_polygon):
        """Fill mode (thickness=-1) should fill interior pixels."""
        result = draw_polygon_on_image(
            blank_image, square_polygon,
            color=(0, 255, 0), thickness=-1, fill_alpha=0.5,
        )
        # Interior point (150, 100) should have non-zero green channel
        assert result[100, 150, 1] > 0

    def test_fill_alpha_zero_leaves_image_unchanged(self, blank_image, square_polygon):
        """fill_alpha=0 means fully transparent → image unchanged."""
        result = draw_polygon_on_image(
            blank_image, square_polygon,
            thickness=-1, fill_alpha=0.0,
        )
        np.testing.assert_array_equal(result, blank_image)

    # ---- guard: None or short polygon ----------------------------------------

    def test_none_polygon_returns_image_unchanged(self, blank_image):
        """None polygon should return the original image unchanged."""
        result = draw_polygon_on_image(blank_image, None)
        np.testing.assert_array_equal(result, blank_image)

    def test_short_polygon_returns_image_unchanged(self, blank_image):
        """Polygon with < 3 vertices should return image unchanged."""
        short = np.array([[10, 10], [20, 20]], dtype=np.float32)
        result = draw_polygon_on_image(blank_image, short)
        np.testing.assert_array_equal(result, blank_image)

    def test_empty_polygon_returns_image_unchanged(self, blank_image):
        """Zero-length polygon array should return image unchanged."""
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        result = draw_polygon_on_image(blank_image, empty)
        np.testing.assert_array_equal(result, blank_image)

    # ---- color parameter -----------------------------------------------------

    def test_custom_color(self, blank_image, square_polygon):
        """Drawing with a red outline should produce red pixels."""
        result = draw_polygon_on_image(
            blank_image, square_polygon,
            color=(0, 0, 255), thickness=2,  # BGR red
        )
        # At least some pixels should have red channel > 0
        assert np.any(result[:, :, 2] > 0)
