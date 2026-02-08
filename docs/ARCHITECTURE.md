# Architecture

Real-time human segmentation system: a Python WebSocket server runs video matting inference and streams polygon outlines to a Phaser.js browser game where the silhouette interacts with physics objects.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (Phaser.js + Matter.js)                            │
│                                                             │
│  webcam → capture frames (JPEG) ──► WebSocket (binary) ──┐ │
│                                                           │ │
│  render polygon + physics ◄── WebSocket (JSON) ◄─────────┘ │
└──────────────────────────────────┬──────────────────────────┘
                                   │ ws://localhost:8765
┌──────────────────────────────────▼──────────────────────────┐
│  Python Segmentation Server (segmentation_server.py)        │
│                                                             │
│  JPEG decode → RVM inference (GPU) → contour → polygon JSON │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### Python Backend

| File | Purpose |
|---|---|
| `segmentation_server.py` | Async WebSocket server. Accepts binary JPEG frames, runs inference via `ThreadPoolExecutor`, returns polygon JSON. Uses a single-slot frame buffer (latest-frame-wins, drops stale frames). |
| `segmentation/segmentation.py` | Shared utilities: `matte_to_polygon()` converts a float32 alpha matte → binary mask → OpenCV contour → Douglas-Peucker simplified polygon. Also provides `TimingStats` and `draw_polygon_on_image()`. |
| `segmentation/rvm/` | Vendored [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (MattingNetwork, MobileNetV3 backbone). Recurrent model — maintains hidden state across frames for temporal coherence. |
| `segmentation/polygon_generator.py` | Standalone synthetic polygon broadcaster (sine-wave movement). Used for testing the frontend without a camera. |
| `segmentation/polygon_config/*.json` | Shape configs for the polygon generator (rectangle, triangle, octagon). |
| `debug_segmentation.py` | Flask-served browser debug tool with 4-panel MJPEG visualization (original, alpha heatmap, binary mask, polygon overlay). |
| `test_camera.py` | Minimal Flask camera test to verify webcam access. |

### TypeScript Frontend (`phaser-matter-game/`)

| File | Purpose |
|---|---|
| `src/main.ts` | `MainScene` — Phaser scene that sets up the webcam as background, spawns bouncing balls (Matter.js), receives polygon data and creates a static physics body from the silhouette vertices. Handles coordinate mapping from segmentation image space → game display space. |
| `src/GameWebSocket.ts` | Single WebSocket client: captures video frames to an `OffscreenCanvas`, encodes as JPEG, sends as binary messages, receives polygon JSON. Single-slot send buffer (frame-drop over latency). Auto-reconnect. Tracks FPS and round-trip metrics. |
| `src/types.ts` | Shared TypeScript types: `PolygonData`, `GameWebSocketConfig`, `GameWebSocketEvents`, `PerformanceMetrics`. |

### Tests

| File | Purpose |
|---|---|
| `tests/test_matte_to_polygon.py` | Unit tests for `matte_to_polygon()` with synthetic masks. |
| `tests/test_segmentation_server.py` | `SegmentationSession` tests + WebSocket round-trip integration test. |
| `tests/conftest.py` | Shared pytest fixtures (synthetic masks, sample JPEG bytes). |
| `phaser-matter-game/src/__tests__/GameWebSocket.test.ts` | Jest tests for the frontend WebSocket class. |

## Data Flow

1. **Browser** captures webcam at ~15 fps, downscales to 640×360, encodes JPEG (~0.7 quality).
2. **Binary WebSocket message** (JPEG bytes) → Python server.
3. **Server** decodes JPEG, converts to torch tensor, runs RVM inference (returns alpha matte).
4. **`matte_to_polygon()`**: Gaussian blur → threshold → morphological open/close → `findContours` → largest contour → `approxPolyDP` → Nx2 float32 polygon.
5. **JSON WebSocket message** back to browser: `{ polygon: [[x,y],...], timestamp, original_image_size: [h,w] }`.
6. **Phaser scene** scales polygon vertices from image coords to game display coords, creates a static Matter.js body, draws green overlay. Balls collide with this body.

## Concurrency Model

- **Server**: `asyncio` event loop + `ThreadPoolExecutor` (default 2 workers) for blocking torch inference. Each WebSocket connection gets a `SegmentationSession` holding RVM recurrent state.
- **Frame buffering**: Both client and server use single-slot buffers — if a new frame arrives before the previous one is processed, the old one is silently dropped. This prevents latency accumulation.
- **Client**: `setInterval` for frame capture; `isSending` flag prevents overlapping sends.

## Tech Stack

- **Python**: PyTorch (CUDA), OpenCV, numpy, websockets (asyncio), Flask (debug tools only)
- **Frontend**: Phaser 3 + Matter.js physics, TypeScript, Vite
- **Model**: RVM MobileNetV3 (`models/rvm_mobilenetv3.pth`, auto-downloaded on first run)
- **Testing**: pytest + pytest-asyncio (Python), Jest + ts-jest (TypeScript)

## Ports

| Service | Default Port |
|---|---|
| Segmentation WebSocket server | `ws://localhost:8765` |
| Phaser game (Vite dev) | `http://localhost:5173` |
| Debug segmentation viewer | `http://127.0.0.1:5001` |
| Camera test tool | `http://127.0.0.1:5000` |

## Running

```bash
# Terminal 1 — backend
source .venv/bin/activate
python segmentation_server.py            # add --device cuda --fp16 for GPU

# Terminal 2 — frontend
cd phaser-matter-game && npm run dev

# Tests
python -m pytest tests/ -v               # Python
cd phaser-matter-game && npm test         # TypeScript
```
