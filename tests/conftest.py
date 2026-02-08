"""Shared fixtures for the segmentation game test suite."""

import os
import sys

import numpy as np
import pytest

# Ensure project root + RVM submodule on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "segmentation", "rvm"))


@pytest.fixture
def synthetic_mask_person():
    """A 480×640 float32 mask simulating a person-shaped blob (torso+head)."""
    mask = np.zeros((480, 640), dtype=np.float32)
    # Head (circle-ish)
    cy_head, cx_head, r_head = 120, 320, 50
    yy, xx = np.ogrid[:480, :640]
    head = ((yy - cy_head) ** 2 + (xx - cx_head) ** 2) <= r_head ** 2
    mask[head] = 1.0
    # Torso (rectangle-ish)
    mask[170:380, 260:380] = 1.0
    return mask


@pytest.fixture
def synthetic_mask_empty():
    """A completely empty mask (no person detected)."""
    return np.zeros((480, 640), dtype=np.float32)


@pytest.fixture
def sample_jpeg_bytes(synthetic_mask_person):
    """A JPEG-encoded 640×480 BGR image (person-shaped white blob on black)."""
    import cv2

    img = (synthetic_mask_person * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()
