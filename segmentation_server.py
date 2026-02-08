#!/usr/bin/env python3
"""
Segmentation WebSocket Server

Unified server that:
  1. Accepts a WebSocket connection from the browser.
  2. Receives JPEG frames as binary WebSocket messages.
  3. Runs RVM (Robust Video Matting) segmentation on each frame.
  4. Generates a polygon from the segmentation mask.
  5. Sends the polygon back as a JSON text message.

Design principles:
  - Single-slot frame buffer (latest frame only, no accumulation).
  - Non-blocking inference via asyncio thread-pool executor.
  - Frame-drop over accumulated latency.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch

from segmentation.rvm.model import MattingNetwork
from segmentation.segmentation import matte_to_polygon, TimingStats
import websockets

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("segmentation_server")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device, fp16: bool):
    """Load the RVM MobileNetV3 model."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    if not os.path.exists(model_path):
        logger.info("Downloading RVM MobileNetV3 model weights...")
        import urllib.request
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
        urllib.request.urlretrieve(url, model_path)
        logger.info("Model weights downloaded.")

    model = MattingNetwork("mobilenetv3").eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    if fp16 and device.type == "cuda":
        model = model.half()
    logger.info("RVM model loaded on %s (fp16=%s)", device, fp16)
    return model


def to_torch_image(frame_bgr: np.ndarray, device: torch.device, fp16: bool):
    """Convert an OpenCV BGR frame to a batched torch tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    if fp16 and device.type == "cuda":
        t = t.half()
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Per-connection segmentation state
# ---------------------------------------------------------------------------

class SegmentationSession:
    """Holds the recurrent state and timing stats for one client session."""

    def __init__(self, model, device: torch.device, fp16: bool, dsr: float,
                 polygon_threshold: float, polygon_min_area: int,
                 polygon_epsilon: float):
        self.model = model
        self.device = device
        self.fp16 = fp16
        self.dsr = dsr
        self.polygon_threshold = polygon_threshold
        self.polygon_min_area = polygon_min_area
        self.polygon_epsilon = polygon_epsilon

        # RVM recurrent state
        self.rec = [None, None, None, None]

        # Timing / stats
        self.timing = TimingStats(max_samples=60)
        self.frame_count = 0
        self.polygon_count = 0
        self.null_polygon_count = 0
        self.last_stats_time = time.time()

    # ---- runs in thread pool (blocking) ----
    def process_frame(self, jpeg_bytes: bytes) -> Optional[dict]:
        """Decode JPEG, run segmentation, return polygon dict or None."""
        t0 = time.time()

        # Decode JPEG
        if len(jpeg_bytes) == 0:
            logger.debug("[frame] Received empty JPEG bytes (0 bytes), skipping")
            return None

        logger.debug("[frame] Received %d bytes JPEG", len(jpeg_bytes))

        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("[frame] cv2.imdecode FAILED for %d bytes", len(jpeg_bytes))
            return None

        t_decode = time.time()
        decode_ms = (t_decode - t0) * 1000
        self.timing.add_timing("data_prep", decode_ms)
        logger.debug("[frame] Decoded %dx%d frame in %.1fms", frame.shape[1], frame.shape[0], decode_ms)

        # Model inference
        src = to_torch_image(frame, self.device, self.fp16)
        with torch.inference_mode():
            fgr, pha, *self.rec = self.model(
                src, *self.rec, self.dsr
            )

        t_infer = time.time()
        infer_ms = (t_infer - t_decode) * 1000
        self.timing.add_timing("model_inference", infer_ms)

        # Alpha matte to polygon
        pha_np = pha[0, 0].cpu().numpy()  # float32 [0,1]

        # Log alpha matte statistics so we can diagnose threshold issues
        pha_min = float(pha_np.min())
        pha_max = float(pha_np.max())
        pha_mean = float(pha_np.mean())
        logger.debug(
            "[frame] Inference %.1fms | alpha matte: min=%.3f max=%.3f mean=%.3f (threshold=%.2f)",
            infer_ms, pha_min, pha_max, pha_mean, self.polygon_threshold,
        )

        polygon = matte_to_polygon(
            pha_np,
            threshold=self.polygon_threshold,
            min_area=self.polygon_min_area,
            epsilon_ratio=self.polygon_epsilon,
            timing_stats=self.timing,
        )

        t_poly = time.time()
        total_ms = (t_poly - t0) * 1000
        self.timing.add_timing("total_frame", total_ms)

        # Stats bookkeeping
        self.frame_count += 1

        if polygon is None:
            self.null_polygon_count += 1
            # Detailed reason logging
            if pha_max < self.polygon_threshold:
                reason = f"alpha max ({pha_max:.3f}) below threshold ({self.polygon_threshold})"
            else:
                reason = f"contour area < {self.polygon_min_area} or no contours found"
            logger.debug("[frame] polygon=None reason: %s (total %.1fms)", reason, total_ms)
        else:
            self.polygon_count += 1
            logger.debug(
                "[frame] polygon=%d vertices (total %.1fms)",
                len(polygon), total_ms,
            )

        # Periodic summary every 5 seconds (always at INFO level)
        now = time.time()
        if now - self.last_stats_time >= 5.0:
            elapsed = now - self.last_stats_time
            fps = self.frame_count / elapsed
            avgs = self.timing.get_average_timings()
            logger.info(
                "FPS=%.1f | frames=%d polygons=%d nulls=%d | "
                "decode=%.1fms infer=%.1fms poly=%.1fms total=%.1fms",
                fps, self.frame_count, self.polygon_count, self.null_polygon_count,
                avgs.get("data_prep", 0),
                avgs.get("model_inference", 0),
                avgs.get("generate_polygon", 0),
                avgs.get("total_frame", 0),
            )
            self.frame_count = 0
            self.polygon_count = 0
            self.null_polygon_count = 0
            self.last_stats_time = now

        if polygon is None:
            return None

        h, w = frame.shape[:2]
        return {
            "polygon": polygon.tolist(),
            "timestamp": time.time(),
            "original_image_size": [h, w],
        }


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_client(websocket, model, device, fp16, dsr,
                        polygon_threshold, polygon_min_area,
                        polygon_epsilon, executor):
    """Handle a single WebSocket client connection."""
    addr = websocket.remote_address
    logger.info("Client connected: %s", addr)

    session = SegmentationSession(
        model, device, fp16, dsr,
        polygon_threshold, polygon_min_area, polygon_epsilon,
    )

    # Single-slot frame buffer & processing flag
    latest_frame: Optional[bytes] = None
    is_processing = False
    lock = asyncio.Lock()
    frames_received = 0
    frames_dropped = 0

    loop = asyncio.get_running_loop()

    async def process_latest():
        """Process the latest buffered frame if available."""
        nonlocal latest_frame, is_processing
        async with lock:
            if latest_frame is None or is_processing:
                return
            frame_bytes = latest_frame
            latest_frame = None  # consume
            is_processing = True

        try:
            result = await loop.run_in_executor(
                executor, session.process_frame, frame_bytes
            )
            if result is not None:
                msg = json.dumps(result)
                await websocket.send(msg)
                logger.debug(
                    "[ws] Sent polygon (%d verts, %d bytes JSON) to %s",
                    len(result["polygon"]), len(msg), addr,
                )
            else:
                logger.debug("[ws] No polygon to send for this frame")
        except Exception:
            logger.exception("Error processing frame")
        finally:
            async with lock:
                is_processing = False
            # If a newer frame arrived while we were busy, process it now
            if latest_frame is not None:
                asyncio.create_task(process_latest())

    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                frames_received += 1
                async with lock:
                    was_overwritten = latest_frame is not None
                    latest_frame = bytes(message)  # single-slot overwrite
                if was_overwritten:
                    frames_dropped += 1
                    logger.debug(
                        "[ws] Frame overwritten (dropped) — received=%d dropped=%d from %s",
                        frames_received, frames_dropped, addr,
                    )
                else:
                    logger.debug(
                        "[ws] Frame buffered (%d bytes) — received=%d from %s",
                        len(message), frames_received, addr,
                    )
                # Kick off processing (no-op if already running)
                asyncio.create_task(process_latest())
            elif isinstance(message, str):
                logger.debug("Text message from client: %s", message[:120])
            else:
                logger.warning("Unknown message type: %s", type(message))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info(
            "Client disconnected: %s (total frames received=%d, dropped=%d)",
            addr, frames_received, frames_dropped,
        )


# ---------------------------------------------------------------------------
# Server entry-point
# ---------------------------------------------------------------------------

async def _log_memory_periodically(interval: float = 30.0):
    """Log memory usage every `interval` seconds when tracemalloc is enabled."""
    import resource
    while True:
        await asyncio.sleep(interval)
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        msg = f"RSS={rss_kb / 1024:.1f}MB"
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            msg += f" | traced={current / 1024 / 1024:.1f}MB peak={peak / 1024 / 1024:.1f}MB"
        logger.info("Memory: %s", msg)


async def run_server(args):
    """Start the WebSocket server and block until shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    model = load_model(args.model_path, device, args.fp16)

    # Thread pool for blocking model inference (1 thread per client is enough)
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    # Start periodic memory logging
    asyncio.create_task(_log_memory_periodically(30.0))

    async def handler(websocket, *_handler_args):
        await handle_client(
            websocket, model, device, args.fp16, args.dsr,
            args.polygon_threshold, args.polygon_min_area,
            args.polygon_epsilon, executor,
        )

    stop = asyncio.get_running_loop().create_future()

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(sig):
        logger.info("Received %s, shutting down...", sig.name)
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, _shutdown, sig)

    server = await websockets.serve(
        handler, args.host, args.port,
        max_size=10 * 1024 * 1024,  # 10 MB max message
        ping_interval=20,
        ping_timeout=20,
    )
    logger.info("Segmentation server listening on ws://%s:%d", args.host, args.port)

    await stop  # wait for shutdown signal

    server.close()
    await server.wait_closed()
    executor.shutdown(wait=False)
    logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Segmentation WebSocket Server")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model_path", default="models/rvm_mobilenetv3.pth")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA")
    p.add_argument("--dsr", type=float, default=0.25, help="RVM downsample ratio")
    p.add_argument("--polygon_threshold", type=float, default=0.5)
    p.add_argument("--polygon_min_area", type=int, default=2000)
    p.add_argument("--polygon_epsilon", type=float, default=0.002,
                    help="Douglas-Peucker epsilon ratio")
    p.add_argument("--max_workers", type=int, default=2,
                    help="Thread pool size for inference")
    p.add_argument("--tracemalloc", action="store_true",
                    help="Enable tracemalloc for memory debugging")
    p.add_argument("--debug", action="store_true",
                    help="Enable DEBUG-level logging (very verbose, per-frame)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger("segmentation_server").setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled (per-frame logs)")
    if args.tracemalloc:
        tracemalloc.start()
        logger.info("tracemalloc enabled")
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
