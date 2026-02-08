---
description: Project-wide context and guardrails for Segment Project (Python segmentation WS server + Phaser game client).
globs:
  - "**/*.py"
  - "phaser-matter-game/src/**/*.{ts,tsx}"
  - "phaser-matter-game/src/**/__tests__/**/*.{ts,tsx}"
  - "tests/**/*.py"
---

## What this repo is

- **Goal**: Real-time webcam human segmentation → polygon outline → physics interaction in a browser game.
- **Architecture** (read first): `docs/ARCHITECTURE.md`
- **Two processes**:
  - **Python**: `segmentation_server.py` (WebSocket server doing RVM inference + polygon extraction)
  - **Browser**: `phaser-matter-game/src/main.ts` (Phaser scene + Matter physics + webcam + polygon rendering)

## Critical invariants (don’t “optimize” these away)

- **Low latency beats completeness**: both sides intentionally use **single-slot buffers** and **drop frames** instead of queueing.
  - Server: `handle_client()` keeps `latest_frame` and overwrites when busy.
  - Client: `GameWebSocket` drops frames when `isSending` is true.
- **WebSocket message contract is stable**:
  - Client → server: **binary** message containing **JPEG bytes**.
  - Server → client: **text JSON**:
    - `polygon`: `[[x,y], ...]` in **original image coordinates**
    - `original_image_size`: `[h, w]`
    - `timestamp`: seconds since epoch
  - If you must change the schema, update **both** `segmentation_server.py` and `phaser-matter-game/src/types.ts` + tests.
- **RVM recurrent state is per-connection**: `SegmentationSession.rec` must not be shared across clients.

## Where to make changes

- **Segmentation / polygon extraction**:
  - `segmentation/segmentation.py` → `matte_to_polygon()` (thresholding, morphology, contours, simplification)
  - `segmentation_server.py` → decoding, inference, timing/logging, websocket handler
- **Networking and capture**:
  - `phaser-matter-game/src/GameWebSocket.ts` (capture rate/size/quality, reconnect, buffers, metrics)
- **Polygon → physics mapping & rendering**:
  - `phaser-matter-game/src/main.ts` → `applyLatestPolygon()` (scaling to video display, winding, area checks, `fromVertices`)

## Common sharp edges (check these before debugging “it’s broken”)

- **CUDA / torch**:
  - `requirements.txt` pins PyTorch from CUDA index; on non-CUDA machines use `--device cpu` at runtime.
  - Model weights auto-download to `models/rvm_mobilenetv3.pth` on first run (network required).
- **Coordinate transforms**:
  - Polygons arrive in segmentation image space (e.g. 640×360) and must be scaled to the displayed video size.
  - Matter.js can fail on self-intersecting/degenerate polygons; keep the client-side area and vertex limits.
- **Performance regressions**:
  - Avoid adding per-frame allocations, heavy logging, or buffering that can introduce latency.

## How to run (local dev)

```bash
# Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python segmentation_server.py                 # add: --device cuda --fp16

# Frontend
cd phaser-matter-game
npm install
npm run dev
```

## How to test

```bash
# Python
source .venv/bin/activate
python -m pytest tests/ -v

# TypeScript
cd phaser-matter-game
npm test
```

## Code style & PR hygiene (agent guidance)

- **Prefer small, localized edits** that preserve the real-time design (frame-drop policy).
- **Add/adjust tests** when you change polygon generation or the WS contract:
  - Python: `tests/test_matte_to_polygon.py`, `tests/test_segmentation_server.py`
  - TS: `phaser-matter-game/src/__tests__/GameWebSocket.test.ts`
- **Logging**:
  - Server has `--debug` for per-frame logs; avoid enabling verbose logs by default.
