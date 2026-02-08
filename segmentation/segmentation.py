"""
Segmentation utilities — shared by segmentation_server.py and debug tools.

Public API:
    TimingStats        — rolling timing statistics tracker
    matte_to_polygon   — convert an alpha matte to a simplified polygon
    draw_polygon_on_image — draw a polygon overlay on an image
"""

import time
from collections import deque

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class TimingStats:
    """Track rolling timing statistics for different processing steps."""

    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self._timings: dict[str, deque] = {}

    def add_timing(self, step_name: str, duration_ms: float):
        if step_name not in self._timings:
            self._timings[step_name] = deque(maxlen=self.max_samples)
        self._timings[step_name].append(duration_ms)

    def get_average_timings(self) -> dict[str, float]:
        return {
            step: (sum(times) / len(times)) if times else 0.0
            for step, times in self._timings.items()
        }


# ---------------------------------------------------------------------------
# Matte → Polygon
# ---------------------------------------------------------------------------

def matte_to_polygon(pha, threshold=0.5, min_area=2000, epsilon_ratio=0.015,
                     return_mask=False, timing_stats=None):
    """Convert an alpha matte to a simplified polygon via contour detection.

    Args:
        pha: 2D float32 array in [0, 1] (alpha matte).
        threshold: binarization threshold for foreground.
        min_area: ignore contours below this many pixels.
        epsilon_ratio: Douglas-Peucker epsilon as a fraction of perimeter.
        return_mask: if True, return (polygon, thresholded_mask) tuple.
        timing_stats: optional TimingStats to record measurements.

    Returns:
        Nx2 float32 array of (x, y) polygon vertices, or None.
        If return_mask is True, returns (polygon, mask) tuple.
    """
    # 1) Smooth and threshold
    t0 = time.time()
    m = cv2.GaussianBlur(pha, (0, 0), 1.2)
    mask = (m >= threshold).astype(np.uint8) * 255
    if timing_stats:
        timing_stats.add_timing('thresholding', (time.time() - t0) * 1000)

    # 2) Morphology to clean noise & holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # 3) Contour extraction
    t1 = time.time()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if timing_stats:
            timing_stats.add_timing('find_contour', (time.time() - t1) * 1000)
        return (None, mask) if return_mask else None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        if timing_stats:
            timing_stats.add_timing('find_contour', (time.time() - t1) * 1000)
        return (None, mask) if return_mask else None

    # 4) Simplify with Douglas-Peucker
    t2 = time.time()
    peri = cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    poly = poly.reshape(-1, 2).astype(np.float32)
    if timing_stats:
        timing_stats.add_timing('generate_polygon', (time.time() - t2) * 1000)
        timing_stats.add_timing('find_contour', (time.time() - t1) * 1000)

    return (poly, mask) if return_mask else poly


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_polygon_on_image(image, polygon, color=(0, 255, 0), thickness=2,
                          fill_alpha=0.3):
    """Draw a polygon on an image with optional semi-transparent fill.

    Args:
        image: BGR image array.
        polygon: Nx2 array of (x, y) vertices.
        color: BGR colour tuple.
        thickness: line thickness (-1 for filled).
        fill_alpha: transparency for fill (0.0 = transparent, 1.0 = opaque).

    Returns:
        Image with polygon drawn.
    """
    if polygon is None or len(polygon) < 3:
        return image

    pts = polygon.astype(np.int32)

    if thickness == -1:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        return cv2.addWeighted(image, 1 - fill_alpha, overlay, fill_alpha, 0)

    cv2.polylines(image, [pts], True, color, thickness)
    return image
