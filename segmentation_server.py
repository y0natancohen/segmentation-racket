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
import struct
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.io import decode_jpeg

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

    # CUDA performance knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA perf flags: cudnn.benchmark=True, TF32 enabled")

    model = MattingNetwork("mobilenetv3").eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    if fp16 and device.type == "cuda":
        model = model.half()
    logger.info("RVM model loaded on %s (fp16=%s)", device, fp16)

    # JIT script + freeze — eliminates Python dispatch overhead per forward
    # call and optimises module attribute lookups (matches official RVM
    # inference_speed_test.py).
    if device.type == "cuda":
        try:
            model = torch.jit.script(model)
            model = torch.jit.freeze(model)
            logger.info("Model JIT-scripted and frozen")
        except Exception as e:
            logger.warning("JIT script/freeze failed, falling back to eager: %s", e)

    # Warmup pass — stabilizes CUDA kernel selection and JIT caches
    if device.type == "cuda":
        _warmup_model(model, device, fp16)

    return model


def _warmup_model(model, device: torch.device, fp16: bool, n_frames: int = 5):
    """Run a few dummy frames through the model to warm up CUDA/cuDNN."""
    logger.info("Running %d warmup inference passes...", n_frames)
    h, w = 288, 512  # match default client capture size
    rec = [None, None, None, None]
    for i in range(n_frames):
        dummy = torch.randn(1, 3, h, w, device=device)
        if fp16:
            dummy = dummy.half()
        with torch.inference_mode():
            _fgr, _pha, *rec = model(dummy, *rec, 0.25)
    # Sync to ensure all kernels finished
    torch.cuda.synchronize()
    logger.info("Warmup complete.")


def to_torch_image(frame_bgr: np.ndarray, device: torch.device, fp16: bool):
    """Convert an OpenCV BGR frame to a batched torch tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device, non_blocking=True).permute(2, 0, 1).float() / 255.0
    if fp16 and device.type == "cuda":
        t = t.half()
    return t.unsqueeze(0)


def _gpu_decode_jpeg(jpeg_bytes: bytes, device: torch.device, fp16: bool):
    """Decode JPEG bytes directly on GPU via nvJPEG (torchvision).

    Returns (src, h, w) where src is [1,3,H,W] float/half on *device*,
    or None if decoding fails.
    """
    try:
        jpeg_tensor = torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8)
        # decode_jpeg with device='cuda' uses nvJPEG — returns [3,H,W] uint8 RGB
        decoded = decode_jpeg(jpeg_tensor, device=device)
        h, w = decoded.shape[1], decoded.shape[2]
        if fp16:
            src = decoded.half().div_(255.0).unsqueeze(0)
        else:
            src = decoded.float().div_(255.0).unsqueeze(0)
        return src, h, w
    except Exception as e:
        logger.debug("[frame] GPU JPEG decode failed, will fall back to CPU: %s", e)
        return None


# ---------------------------------------------------------------------------
# Per-connection segmentation state
# ---------------------------------------------------------------------------

class SegmentationSession:
    """Holds the recurrent state and timing stats for one client session."""

    def __init__(self, model, device: torch.device, fp16: bool, dsr: float,
                 polygon_threshold: float, polygon_min_area: int,
                 polygon_epsilon: float, profile: bool = False):
        self.model = model
        self.device = device
        self.fp16 = fp16
        self.dsr = dsr
        self.polygon_threshold = polygon_threshold
        self.polygon_min_area = polygon_min_area
        self.polygon_epsilon = polygon_epsilon
        self.profile = profile

        # RVM recurrent state
        self.rec = [None, None, None, None]

        # Timing / stats
        self.timing = TimingStats(max_samples=60)
        self.frame_count = 0
        self.polygon_count = 0
        self.null_polygon_count = 0
        self.last_stats_time = time.time()

        # GPU profiling (CUDA Event timing) — only active when --profile
        self._gpu_kernel_times: list[float] = []  # ms per frame

    # ---- runs in thread pool (blocking) ----
    def process_frame(self, jpeg_bytes: bytes, frame_timestamp: float = 0) -> Optional[dict]:
        """Decode JPEG, run segmentation, return polygon dict or None."""
        t0 = time.time()

        # Decode JPEG
        if len(jpeg_bytes) == 0:
            logger.debug("[frame] Received empty JPEG bytes (0 bytes), skipping")
            return None

        logger.debug("[frame] Received %d bytes JPEG", len(jpeg_bytes))

        # Decode JPEG — use GPU (nvJPEG) on CUDA, fall back to CPU
        gpu_decoded = None
        if self.device.type == "cuda":
            gpu_decoded = _gpu_decode_jpeg(jpeg_bytes, self.device, self.fp16)

        if gpu_decoded is not None:
            src, frame_h, frame_w = gpu_decoded
        else:
            # CPU fallback: cv2.imdecode + to_torch_image
            buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("[frame] cv2.imdecode FAILED for %d bytes", len(jpeg_bytes))
                return None
            frame_h, frame_w = frame.shape[:2]
            src = to_torch_image(frame, self.device, self.fp16)

        t_decode = time.time()
        decode_ms = (t_decode - t0) * 1000
        self.timing.add_timing("data_prep", decode_ms)
        logger.debug("[frame] Decoded %dx%d frame in %.1fms", frame_w, frame_h, decode_ms)

        # CUDA Event timing for precise GPU kernel measurement
        if self.profile and self.device.type == "cuda":
            evt_start = torch.cuda.Event(enable_timing=True)
            evt_end = torch.cuda.Event(enable_timing=True)
            evt_start.record()

        with torch.inference_mode():
            fgr, pha, *self.rec = self.model(
                src, *self.rec, self.dsr
            )

        if self.profile and self.device.type == "cuda":
            evt_end.record()

        # Synchronize so wall-clock timing reflects actual GPU completion
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t_infer = time.time()
        infer_ms = (t_infer - t_decode) * 1000
        self.timing.add_timing("model_inference", infer_ms)

        # Record GPU kernel time from CUDA Events
        if self.profile and self.device.type == "cuda":
            gpu_kernel_ms = evt_start.elapsed_time(evt_end)
            self._gpu_kernel_times.append(gpu_kernel_ms)
            self.timing.add_timing("gpu_kernel", gpu_kernel_ms)

        # Alpha matte to polygon
        pha_np = pha[0, 0].float().cpu().numpy()  # ensure float32 [0,1] even under fp16

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
        poly_ms = (t_poly - t_infer) * 1000
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
            base_msg = (
                "FPS=%.1f | frames=%d polygons=%d nulls=%d | "
                "decode=%.1fms infer=%.1fms poly=%.1fms total=%.1fms"
            )
            base_args = (
                fps, self.frame_count, self.polygon_count, self.null_polygon_count,
                avgs.get("data_prep", 0),
                avgs.get("model_inference", 0),
                avgs.get("generate_polygon", 0),
                avgs.get("total_frame", 0),
            )

            if self.profile and self.device.type == "cuda":
                # GPU profiling extras
                gpu_kernel_avg = avgs.get("gpu_kernel", 0)
                infer_avg = avgs.get("model_inference", 0)
                cpu_overhead = max(0, infer_avg - gpu_kernel_avg)
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                total_avg = avgs.get("total_frame", 0)
                gpu_util_pct = (gpu_kernel_avg / total_avg * 100) if total_avg > 0 else 0

                logger.info(
                    base_msg + " | GPU: kernel=%.1fms overhead=%.1fms util=%.0f%% mem=%.0fMB/%.0fMB",
                    *base_args,
                    gpu_kernel_avg, cpu_overhead, gpu_util_pct,
                    gpu_mem_mb, gpu_reserved_mb,
                )
                self._gpu_kernel_times.clear()
            else:
                logger.info(base_msg, *base_args)

            self.frame_count = 0
            self.polygon_count = 0
            self.null_polygon_count = 0
            self.last_stats_time = now

        if polygon is None:
            return None

        return {
            "polygon": polygon.tolist(),
            "timestamp": time.time(),
            "original_image_size": [frame_h, frame_w],
            "frame_timestamp": int(frame_timestamp),
            "server_timings": {
                "decode_ms": round(decode_ms, 2),
                "inference_ms": round(infer_ms, 2),
                "polygon_ms": round(poly_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_client(websocket, model, device, fp16, dsr,
                        polygon_threshold, polygon_min_area,
                        polygon_epsilon, executor, profile=False):
    """Handle a single WebSocket client connection."""
    addr = websocket.remote_address
    logger.info("Client connected: %s", addr)

    session = SegmentationSession(
        model, device, fp16, dsr,
        polygon_threshold, polygon_min_area, polygon_epsilon,
        profile=profile,
    )

    frames_received = 0
    frames_dropped = 0

    loop = asyncio.get_running_loop()

    # Single-slot queue: putting a new frame evicts any unconsumed one.
    # Each entry is (frame_timestamp, jpeg_bytes).
    frame_queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=1)

    async def worker():
        """Persistent worker — consumes from the queue and sends results."""
        while True:
            frame_ts, jpeg_bytes = await frame_queue.get()
            try:
                result = await loop.run_in_executor(
                    executor, session.process_frame, jpeg_bytes, frame_ts
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
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                logger.exception("Error processing frame")

    worker_task = asyncio.create_task(worker())

    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                frames_received += 1
                raw = bytes(message)

                # Extract 8-byte Float64 big-endian timestamp header
                if len(raw) > 8:
                    frame_ts = struct.unpack('>d', raw[:8])[0]
                    jpeg_bytes = raw[8:]
                else:
                    frame_ts = 0.0
                    jpeg_bytes = raw

                # Single-slot: evict stale frame if queue is full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                        frames_dropped += 1
                        logger.debug(
                            "[ws] Frame evicted (dropped) — received=%d dropped=%d from %s",
                            frames_received, frames_dropped, addr,
                        )
                    except asyncio.QueueEmpty:
                        pass

                frame_queue.put_nowait((frame_ts, jpeg_bytes))
                logger.debug(
                    "[ws] Frame queued (%d bytes, ts=%.0f) — received=%d from %s",
                    len(jpeg_bytes), frame_ts, frames_received, addr,
                )
            elif isinstance(message, str):
                logger.debug("Text message from client: %s", message[:120])
            else:
                logger.warning("Unknown message type: %s", type(message))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
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

    # Probe GPU JPEG decode availability (torchvision decode_jpeg + nvJPEG)
    if device.type == "cuda":
        try:
            _probe = torch.zeros(100, dtype=torch.uint8)
            # Intentionally invalid — we just check the code path doesn't
            # error on missing nvJPEG; the RuntimeError from bad data is fine.
            decode_jpeg(_probe, device=device)
        except RuntimeError:
            # Expected: bad JPEG data, but nvJPEG call succeeded → available
            logger.info("GPU JPEG decode available (torchvision decode_jpeg + nvJPEG)")
        except Exception as e:
            logger.warning("GPU JPEG decode NOT available, will use CPU: %s", e)

    # Thread pool for blocking model inference (1 thread per client is enough)
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    # Start periodic memory logging
    asyncio.create_task(_log_memory_periodically(30.0))

    async def handler(websocket, *_handler_args):
        await handle_client(
            websocket, model, device, args.fp16, args.dsr,
            args.polygon_threshold, args.polygon_min_area,
            args.polygon_epsilon, executor,
            profile=args.profile,
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
    p.add_argument("--polygon_epsilon", type=float, default=0.001,
                    help="Douglas-Peucker epsilon ratio")
    p.add_argument("--max_workers", type=int, default=1,
                    help="Thread pool size for inference (1 avoids GIL contention)")
    p.add_argument("--tracemalloc", action="store_true",
                    help="Enable tracemalloc for memory debugging")
    p.add_argument("--debug", action="store_true",
                    help="Enable DEBUG-level logging (very verbose, per-frame)")
    p.add_argument("--profile", action="store_true",
                    help="Enable GPU profiling (CUDA Event timing + memory logging)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger("segmentation_server").setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled (per-frame logs)")
    if args.tracemalloc:
        tracemalloc.start()
        logger.info("tracemalloc enabled")
    if args.profile:
        logger.info("GPU profiling enabled (CUDA Event timing + memory)")
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
