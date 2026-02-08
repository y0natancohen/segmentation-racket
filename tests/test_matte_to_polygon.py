"""Tests for matte_to_polygon: converting a segmentation alpha matte into a
simplified polygon (Nx2 float32 array of vertices)."""

import numpy as np
import pytest

from segmentation.segmentation import matte_to_polygon


class TestMatteToPolygon:
    """Unit tests for matte_to_polygon."""

    def test_person_mask_returns_polygon(self, synthetic_mask_person):
        """A clear person-shaped mask should produce a valid polygon."""
        poly = matte_to_polygon(synthetic_mask_person)
        assert poly is not None
        assert isinstance(poly, np.ndarray)
        assert poly.ndim == 2
        assert poly.shape[1] == 2
        # Should have at least 3 vertices (triangle)
        assert poly.shape[0] >= 3

    def test_empty_mask_returns_none(self, synthetic_mask_empty):
        """An empty mask (no foreground) should return None."""
        poly = matte_to_polygon(synthetic_mask_empty)
        assert poly is None

    def test_tiny_blob_below_min_area(self):
        """A very small blob (< min_area) should return None."""
        mask = np.zeros((480, 640), dtype=np.float32)
        # 5×5 blob → area=25, well below default min_area=2000
        mask[200:205, 300:305] = 1.0
        poly = matte_to_polygon(mask, min_area=2000)
        assert poly is None

    def test_threshold_controls_sensitivity(self, synthetic_mask_person):
        """Higher threshold should still work on a strong mask."""
        poly_high = matte_to_polygon(synthetic_mask_person, threshold=0.9)
        poly_low = matte_to_polygon(synthetic_mask_person, threshold=0.1)
        # Both should return something on a clear mask
        assert poly_high is not None
        assert poly_low is not None

    def test_epsilon_controls_vertex_count(self, synthetic_mask_person):
        """Larger epsilon → fewer vertices (coarser polygon)."""
        poly_fine = matte_to_polygon(synthetic_mask_person, epsilon_ratio=0.001)
        poly_coarse = matte_to_polygon(synthetic_mask_person, epsilon_ratio=0.05)
        assert poly_fine is not None
        assert poly_coarse is not None
        assert poly_fine.shape[0] >= poly_coarse.shape[0]

    def test_return_mask_flag(self, synthetic_mask_person):
        """With return_mask=True, should return (polygon, mask) tuple."""
        result = matte_to_polygon(synthetic_mask_person, return_mask=True)
        assert isinstance(result, tuple)
        poly, mask = result
        assert poly is not None
        assert mask.shape == synthetic_mask_person.shape[:2]

    def test_vertices_within_image_bounds(self, synthetic_mask_person):
        """All polygon vertices should be within the image dimensions."""
        poly = matte_to_polygon(synthetic_mask_person)
        assert poly is not None
        H, W = synthetic_mask_person.shape[:2]
        assert np.all(poly[:, 0] >= 0) and np.all(poly[:, 0] <= W)
        assert np.all(poly[:, 1] >= 0) and np.all(poly[:, 1] <= H)

    def test_full_white_mask(self):
        """A fully-white mask should produce a rectangle-like polygon."""
        mask = np.ones((480, 640), dtype=np.float32)
        poly = matte_to_polygon(mask)
        assert poly is not None
        assert poly.shape[0] >= 4  # at least a quad
