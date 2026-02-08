#!/usr/bin/env python3
"""
Debug Polygon Sender — synthetic polygon WebSocket server for testing
the browser-side physics pipeline.

Replaces segmentation_server.py on the same port.  Accepts connections
from the **unmodified** browser game, ignores incoming JPEG frames, and
sends pre-programmed polygon sequences at a configurable rate.

Usage:
    python debug_polygon_sender.py                          # default: static rect
    python debug_polygon_sender.py --shape person           # person silhouette
    python debug_polygon_sender.py --shape moving_rect      # translating rectangle
    python debug_polygon_sender.py --shape growing_circle   # growing/shrinking circle
    python debug_polygon_sender.py --shape all --fps 10     # cycle through all shapes
"""

import argparse
import asyncio
import json
import logging
import math
import signal
import time
from typing import List, Tuple

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug_polygon_sender")

# ---------------------------------------------------------------------------
# Image size the browser expects (must match what process_frame normally
# puts in original_image_size).  360x640 is the default capture size.
# ---------------------------------------------------------------------------
IMAGE_H = 360
IMAGE_W = 640


# ---------------------------------------------------------------------------
# Shape generators — each returns a list of [x, y] pairs in *image* coords.
# ---------------------------------------------------------------------------

def shape_static_rect() -> List[List[float]]:
    """A static rectangle centred in the image."""
    cx, cy = IMAGE_W / 2, IMAGE_H / 2
    hw, hh = 120, 80
    return [
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ]


def shape_person_silhouette() -> List[List[float]]:
    """A rough person-like silhouette (~20 vertices)."""
    cx = IMAGE_W / 2
    # Define a person outline (head, shoulders, torso, legs)
    # All coordinates are absolute in image space
    return [
        # Head (top)
        [cx - 25, 60],
        [cx - 15, 40],
        [cx + 15, 40],
        [cx + 25, 60],
        # Neck
        [cx + 15, 75],
        # Right shoulder + arm
        [cx + 80, 90],
        [cx + 85, 130],
        [cx + 55, 135],
        # Right torso
        [cx + 50, 200],
        # Right leg
        [cx + 55, 280],
        [cx + 60, 340],
        [cx + 30, 345],
        [cx + 25, 285],
        # Crotch
        [cx, 230],
        # Left leg
        [cx - 25, 285],
        [cx - 30, 345],
        [cx - 60, 340],
        [cx - 55, 280],
        # Left torso
        [cx - 50, 200],
        # Left shoulder + arm
        [cx - 55, 135],
        [cx - 85, 130],
        [cx - 80, 90],
        # Left neck
        [cx - 15, 75],
    ]


def shape_moving_rect(t: float) -> List[List[float]]:
    """A rectangle that moves horizontally across the image."""
    period = 4.0  # seconds for one full sweep
    phase = (t % period) / period  # 0..1
    # Ping-pong
    if phase > 0.5:
        phase = 1.0 - phase
    phase *= 2.0  # 0..1

    margin = 100
    cx = margin + phase * (IMAGE_W - 2 * margin)
    cy = IMAGE_H / 2
    hw, hh = 60, 100
    return [
        [cx - hw, cy - hh],
        [cx + hw, cy - hh],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ]


def shape_growing_circle(t: float) -> List[List[float]]:
    """A circle that grows and shrinks over time."""
    period = 3.0
    phase = (t % period) / period
    # Sine wave: radius oscillates between 40 and 140
    radius = 40 + 100 * (0.5 + 0.5 * math.sin(phase * 2 * math.pi))

    cx, cy = IMAGE_W / 2, IMAGE_H / 2
    n_points = 24
    poly = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        poly.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle)])
    return poly


def shape_rotating_diamond(t: float) -> List[List[float]]:
    """A diamond that rotates in place."""
    cx, cy = IMAGE_W / 2, IMAGE_H / 2
    r = 100
    angle = t * 0.8  # radians per second
    pts = []
    for i in range(4):
        a = angle + i * math.pi / 2
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return pts


# Registry of all shapes
SHAPE_GENERATORS = {
    "static_rect": lambda t: shape_static_rect(),
    "person": lambda t: shape_person_silhouette(),
    "moving_rect": shape_moving_rect,
    "growing_circle": shape_growing_circle,
    "rotating_diamond": shape_rotating_diamond,
}

ALL_SHAPE_NAMES = list(SHAPE_GENERATORS.keys())


def get_polygon(shape_name: str, t: float, cycle_period: float = 5.0) -> Tuple[List[List[float]], str]:
    """
    Return (polygon, active_shape_name) for the given shape at time t.
    If shape_name == 'all', cycles through all shapes.
    """
    if shape_name == "all":
        idx = int(t / cycle_period) % len(ALL_SHAPE_NAMES)
        active = ALL_SHAPE_NAMES[idx]
        return SHAPE_GENERATORS[active](t), active
    else:
        return SHAPE_GENERATORS[shape_name](t), shape_name


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def handle_client(websocket, path, shape_name: str, fps: float, cycle_period: float):
    """Handle a single browser client."""
    addr = websocket.remote_address
    logger.info("Client connected: %s (shape=%s, fps=%.1f)", addr, shape_name, fps)

    interval = 1.0 / fps
    start_time = time.time()
    sent_count = 0
    frames_received = 0

    async def send_loop():
        nonlocal sent_count
        while True:
            t = time.time() - start_time
            polygon, active_shape = get_polygon(shape_name, t, cycle_period)

            msg = json.dumps({
                "polygon": polygon,
                "timestamp": time.time(),
                "original_image_size": [IMAGE_H, IMAGE_W],
            })

            try:
                await websocket.send(msg)
                sent_count += 1
                if sent_count % int(fps * 5) == 0:  # log every ~5 seconds
                    logger.info(
                        "Sent %d polygons to %s (shape=%s, %d verts)",
                        sent_count, addr, active_shape, len(polygon),
                    )
            except websockets.exceptions.ConnectionClosed:
                break

            await asyncio.sleep(interval)

    async def recv_loop():
        nonlocal frames_received
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    frames_received += 1
                    if frames_received % 100 == 1:
                        logger.debug(
                            "Received frame from browser (%d bytes, total=%d)",
                            len(message), frames_received,
                        )
                # Ignore all messages — we're a debug sender
        except websockets.exceptions.ConnectionClosed:
            pass

    # Run send and receive loops concurrently
    send_task = asyncio.create_task(send_loop())
    recv_task = asyncio.create_task(recv_loop())

    done, pending = await asyncio.wait(
        [send_task, recv_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

    logger.info(
        "Client disconnected: %s (sent=%d polygons, received=%d frames)",
        addr, sent_count, frames_received,
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def run_server(args):
    stop = asyncio.get_running_loop().create_future()

    def _shutdown(sig):
        logger.info("Received %s, shutting down...", sig.name)
        if not stop.done():
            stop.set_result(None)

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, _shutdown, sig)

    async def handler(websocket, path):
        await handle_client(websocket, path, args.shape, args.fps, args.cycle_period)

    server = await websockets.serve(
        handler, args.host, args.port,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    )

    shapes_info = (
        f"all (cycling every {args.cycle_period}s: {', '.join(ALL_SHAPE_NAMES)})"
        if args.shape == "all"
        else args.shape
    )
    logger.info(
        "Debug polygon sender on ws://%s:%d — shape=%s, fps=%.1f",
        args.host, args.port, shapes_info, args.fps,
    )
    logger.info("Available shapes: %s, all", ", ".join(ALL_SHAPE_NAMES))

    await stop
    server.close()
    await server.wait_closed()
    logger.info("Server stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Debug Polygon Sender")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--shape",
        default="static_rect",
        choices=list(SHAPE_GENERATORS.keys()) + ["all"],
        help="Which shape to send",
    )
    p.add_argument("--fps", type=float, default=10, help="Polygon send rate")
    p.add_argument(
        "--cycle_period", type=float, default=5.0,
        help="Seconds per shape when --shape=all",
    )
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
