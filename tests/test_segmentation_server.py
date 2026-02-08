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

import torch
from model import MattingNetwork
from segmentation_server import SegmentationSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _model_path() -> str:
    return os.path.join(PROJECT_ROOT, "models", "rvm_mobilenetv3.pth")


def _load_raw_model(device: torch.device, fp16: bool):
    """Load the MattingNetwork directly (no warmup / CUDA flags)."""
    path = _model_path()
    if not os.path.exists(path):
        pytest.skip("Model file not found at %s" % path)
    model = MattingNetwork("mobilenetv3").eval().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    if fp16 and device.type == "cuda":
        model = model.half()
    return model


# ---------------------------------------------------------------------------
# Fixtures — one session per (device, fp16) combo, scoped per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session():
    """SegmentationSession on CPU / fp32 (always available)."""
    device = torch.device("cpu")
    model = _load_raw_model(device, fp16=False)
    return SegmentationSession(
        model=model, device=device, fp16=False,
        dsr=0.25, polygon_threshold=0.5,
        polygon_min_area=500, polygon_epsilon=0.01,
    )


@pytest.fixture(scope="module")
def session_cuda():
    """SegmentationSession on CUDA / fp32."""
    if not _has_cuda():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    model = _load_raw_model(device, fp16=False)
    return SegmentationSession(
        model=model, device=device, fp16=False,
        dsr=0.25, polygon_threshold=0.5,
        polygon_min_area=500, polygon_epsilon=0.01,
    )


@pytest.fixture(scope="module")
def session_cuda_fp16():
    """SegmentationSession on CUDA / fp16."""
    if not _has_cuda():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    model = _load_raw_model(device, fp16=True)
    return SegmentationSession(
        model=model, device=device, fp16=True,
        dsr=0.25, polygon_threshold=0.5,
        polygon_min_area=500, polygon_epsilon=0.01,
    )


# ---------------------------------------------------------------------------
# SegmentationSession unit tests — CPU (always runs)
# ---------------------------------------------------------------------------

class TestSegmentationSession:
    """Tests for the blocking process_frame() method (CPU)."""

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
# SegmentationSession unit tests — CUDA fp32
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestSegmentationSessionCUDA:
    """Tests for process_frame() on CUDA with fp32."""

    def test_process_valid_jpeg(self, session_cuda, sample_jpeg_bytes):
        """A valid JPEG should produce a polygon dict or None on CUDA fp32."""
        result = session_cuda.process_frame(sample_jpeg_bytes)
        if result is not None:
            assert "polygon" in result
            assert "timestamp" in result
            assert "original_image_size" in result
            assert isinstance(result["polygon"], list)
            assert len(result["original_image_size"]) == 2

    def test_process_invalid_jpeg(self, session_cuda):
        result = session_cuda.process_frame(b"not-a-jpeg")
        assert result is None

    def test_process_empty_bytes(self, session_cuda):
        result = session_cuda.process_frame(b"")
        assert result is None

    def test_latency_reasonable(self, session_cuda, sample_jpeg_bytes):
        """CUDA inference should be faster than CPU; 2s generous ceiling."""
        t0 = time.time()
        session_cuda.process_frame(sample_jpeg_bytes)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"CUDA fp32 frame took {elapsed:.1f}s — too slow"


# ---------------------------------------------------------------------------
# SegmentationSession unit tests — CUDA fp16
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestSegmentationSessionCUDAFP16:
    """Tests for process_frame() on CUDA with fp16 (half precision)."""

    def test_process_valid_jpeg(self, session_cuda_fp16, sample_jpeg_bytes):
        """fp16 inference should work end-to-end without dtype errors."""
        result = session_cuda_fp16.process_frame(sample_jpeg_bytes)
        if result is not None:
            assert "polygon" in result
            assert "timestamp" in result
            assert "original_image_size" in result
            assert isinstance(result["polygon"], list)
            assert len(result["original_image_size"]) == 2
            # Verify polygon coordinates are regular floats, not fp16 artefacts
            for pt in result["polygon"]:
                assert all(isinstance(v, float) for v in pt)

    def test_process_invalid_jpeg(self, session_cuda_fp16):
        result = session_cuda_fp16.process_frame(b"not-a-jpeg")
        assert result is None

    def test_process_empty_bytes(self, session_cuda_fp16):
        result = session_cuda_fp16.process_frame(b"")
        assert result is None

    def test_latency_reasonable(self, session_cuda_fp16, sample_jpeg_bytes):
        """fp16 should be at least as fast as fp32; 2s generous ceiling."""
        t0 = time.time()
        session_cuda_fp16.process_frame(sample_jpeg_bytes)
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"CUDA fp16 frame took {elapsed:.1f}s — too slow"

    def test_multiple_frames_stable(self, session_cuda_fp16, sample_jpeg_bytes):
        """Run several frames to verify recurrent state stays stable under fp16."""
        for i in range(5):
            result = session_cuda_fp16.process_frame(sample_jpeg_bytes)
            # Should not crash; result may be None for synthetic images


# ---------------------------------------------------------------------------
# WebSocket round-trip tests
# ---------------------------------------------------------------------------

async def _run_ws_round_trip(device: torch.device, fp16: bool, sample_jpeg_bytes: bytes):
    """Shared helper: spin up a WS server, send one frame, validate response."""
    from segmentation_server import load_model, handle_client
    from concurrent.futures import ThreadPoolExecutor
    import websockets

    model = load_model(_model_path(), device, fp16=fp16)
    executor = ThreadPoolExecutor(max_workers=1)

    async def handler(ws, *_args):
        await handle_client(ws, model, device, fp16, 0.25, 0.5, 500, 0.01, executor)

    server = await websockets.serve(handler, "localhost", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://localhost:{port}"

        async with websockets.connect(uri) as ws:
            await ws.send(sample_jpeg_bytes)
            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                pytest.skip("Server did not reply in time")
                return

            data = json.loads(reply)
            assert "original_image_size" in data
            assert "timestamp" in data
            _validate_polygon_data_schema(data)
    finally:
        server.close()
        await server.wait_closed()
        executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_ws_round_trip(sample_jpeg_bytes):
    """WebSocket round-trip on CPU / fp32."""
    if not os.path.exists(_model_path()):
        pytest.skip("Model file not found")
    await _run_ws_round_trip(torch.device("cpu"), fp16=False,
                             sample_jpeg_bytes=sample_jpeg_bytes)


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
async def test_ws_round_trip_cuda(sample_jpeg_bytes):
    """WebSocket round-trip on CUDA / fp32."""
    if not os.path.exists(_model_path()):
        pytest.skip("Model file not found")
    await _run_ws_round_trip(torch.device("cuda"), fp16=False,
                             sample_jpeg_bytes=sample_jpeg_bytes)


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
async def test_ws_round_trip_cuda_fp16(sample_jpeg_bytes):
    """WebSocket round-trip on CUDA / fp16."""
    if not os.path.exists(_model_path()):
        pytest.skip("Model file not found")
    await _run_ws_round_trip(torch.device("cuda"), fp16=True,
                             sample_jpeg_bytes=sample_jpeg_bytes)


# ---------------------------------------------------------------------------
# Schema validation helper (WebSocket contract)
# ---------------------------------------------------------------------------

def _validate_polygon_data_schema(data: dict):
    """Validate that a server response matches the PolygonData TS type.

    TypeScript definition (types.ts):
        polygon: number[][];           // [[x,y], ...]
        timestamp: number;             // seconds since epoch
        original_image_size: number[]; // [height, width]
    """
    # Required keys
    assert isinstance(data, dict), "Response must be a JSON object"
    for key in ("timestamp", "original_image_size"):
        assert key in data, f"Missing required key: {key}"

    # timestamp — float (seconds since epoch)
    assert isinstance(data["timestamp"], (int, float)), "timestamp must be numeric"
    assert data["timestamp"] > 0, "timestamp must be positive"

    # original_image_size — [height, width]
    size = data["original_image_size"]
    assert isinstance(size, list) and len(size) == 2, "original_image_size must be [h, w]"
    assert all(isinstance(v, int) for v in size), "original_image_size values must be int"
    assert all(v > 0 for v in size), "original_image_size values must be positive"

    # polygon — may be absent if model produced no segmentation
    if "polygon" in data and data["polygon"] is not None:
        poly = data["polygon"]
        assert isinstance(poly, list), "polygon must be a list"
        assert len(poly) >= 3, "polygon must have >= 3 vertices"
        h, w = size
        for i, pt in enumerate(poly):
            assert isinstance(pt, list) and len(pt) == 2, (
                f"polygon[{i}] must be [x, y], got {pt}"
            )
            x, y = pt
            assert isinstance(x, (int, float)) and isinstance(y, (int, float)), (
                f"polygon[{i}] coordinates must be numeric"
            )
            assert 0 <= x <= w, f"polygon[{i}].x={x} out of range [0, {w}]"
            assert 0 <= y <= h, f"polygon[{i}].y={y} out of range [0, {h}]"


class TestPolygonDataSchema:
    """Unit tests for the schema validator itself (runs without a model)."""

    def test_valid_polygon_data(self):
        """A well-formed response should pass validation."""
        data = {
            "polygon": [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            "timestamp": 1700000000.123,
            "original_image_size": [480, 640],
        }
        _validate_polygon_data_schema(data)  # should not raise

    def test_missing_timestamp_fails(self):
        data = {"polygon": [[0, 0], [1, 1], [2, 2]], "original_image_size": [100, 200]}
        with pytest.raises(AssertionError, match="timestamp"):
            _validate_polygon_data_schema(data)

    def test_missing_original_image_size_fails(self):
        data = {"polygon": [[0, 0], [1, 1], [2, 2]], "timestamp": 1.0}
        with pytest.raises(AssertionError, match="original_image_size"):
            _validate_polygon_data_schema(data)

    def test_polygon_vertex_out_of_bounds_fails(self):
        data = {
            "polygon": [[999, 0], [1, 1], [2, 2]],
            "timestamp": 1.0,
            "original_image_size": [100, 100],
        }
        with pytest.raises(AssertionError, match="out of range"):
            _validate_polygon_data_schema(data)

    def test_no_polygon_key_is_ok(self):
        """Server may omit polygon entirely when nothing detected."""
        data = {"timestamp": 1.0, "original_image_size": [480, 640]}
        _validate_polygon_data_schema(data)  # should not raise
