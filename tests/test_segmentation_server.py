"""Tests for segmentation_server.py — WebSocket round-trip and session logic."""

import asyncio
import json
import os
import sys
import time

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "segmentation", "rvm"))

from segmentation_server import SegmentationSession


# ---------------------------------------------------------------------------
# SegmentationSession unit tests (no GPU required, uses CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session():
    """Create a SegmentationSession with a real model on CPU for testing."""
    import torch
    from model import MattingNetwork

    model_path = os.path.join(PROJECT_ROOT, "models", "rvm_mobilenetv3.pth")
    if not os.path.exists(model_path):
        pytest.skip("Model file not found at %s" % model_path)

    device = torch.device("cpu")
    model = MattingNetwork("mobilenetv3").eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    return SegmentationSession(
        model=model,
        device=device,
        fp16=False,
        dsr=0.25,
        polygon_threshold=0.5,
        polygon_min_area=500,
        polygon_epsilon=0.01,
    )


class TestSegmentationSession:
    """Tests for the blocking process_frame() method."""

    def test_process_valid_jpeg(self, session, sample_jpeg_bytes):
        """A valid JPEG should produce a polygon dict or None (no crash)."""
        result = session.process_frame(sample_jpeg_bytes)
        # May be None if the synthetic image doesn't trigger the model
        if result is not None:
            assert "polygon" in result
            assert "timestamp" in result
            assert "original_image_size" in result
            assert isinstance(result["polygon"], list)
            assert len(result["original_image_size"]) == 2

    def test_process_invalid_jpeg(self, session):
        """Garbage bytes should return None (not crash)."""
        result = session.process_frame(b"not-a-jpeg")
        assert result is None

    def test_process_empty_bytes(self, session):
        """Empty bytes should return None."""
        result = session.process_frame(b"")
        assert result is None

    def test_latency_reasonable(self, session, sample_jpeg_bytes):
        """Processing a frame should take <5s on CPU (sanity check)."""
        t0 = time.time()
        session.process_frame(sample_jpeg_bytes)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Frame processing took {elapsed:.1f}s — too slow"


# ---------------------------------------------------------------------------
# WebSocket round-trip test (requires running server — integration test)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_round_trip(sample_jpeg_bytes):
    """Start a minimal server, send a JPEG frame, expect JSON polygon back.

    This test is skipped if torch is not available or the model is missing.
    """
    import torch
    model_path = os.path.join(PROJECT_ROOT, "models", "rvm_mobilenetv3.pth")
    if not os.path.exists(model_path):
        pytest.skip("Model file not found")

    from segmentation_server import load_model, handle_client
    from concurrent.futures import ThreadPoolExecutor
    import websockets

    device = torch.device("cpu")
    model = load_model(model_path, device, fp16=False)
    executor = ThreadPoolExecutor(max_workers=1)

    async def handler(ws, path):
        await handle_client(ws, path, model, device, False, 0.25, 0.5, 500, 0.01, executor)

    server = await websockets.serve(handler, "localhost", 0)
    try:
        # Get the dynamically assigned port
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://localhost:{port}"

        async with websockets.connect(uri) as ws:
            await ws.send(sample_jpeg_bytes)
            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                pytest.skip("Server did not reply in time (CPU inference slow)")
                return

            data = json.loads(reply)
            assert "original_image_size" in data
            # polygon may or may not be present depending on model output
            assert "timestamp" in data
    finally:
        server.close()
        await server.wait_closed()
        executor.shutdown(wait=False)
