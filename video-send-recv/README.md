# Video Send-Receive with WebRTC

A real-time video streaming application that captures camera video, streams it to a Python backend over WebRTC, and returns per-frame intensity analysis via DataChannel.

## Architecture

- **Frontend**: Vite + React + TypeScript with HTTPS dev server
- **Backend**: Python 3.11 with aiortc, aiohttp, and PyAV
- **Transport**: WebRTC (media for video + DataChannel for metrics)
- **Signaling**: HTTP POST /offer (SDP offer → answer)

## Quick Start

### Backend Setup

```bash
cd server
pip install -r requirements.txt
python server.py
```

Health check: `curl http://localhost:8080/health`

### Frontend Setup

```bash
cd web-app
npm install
npm run dev
```

Open the HTTPS URL shown in the terminal.

### E2E Testing

```bash
cd web-app
npx playwright install
npm run e2e
```

## Features

- Real-time camera capture at 640×360 @ 30fps
- WebRTC video streaming with H.264/VP8/VP9 codec preferences
- Per-frame intensity analysis (0-255 and normalized 0-1)
- DataChannel communication for metrics (~30 FPS)
- Live overlay showing intensity and metrics FPS
- HTTPS development server with SSL certificates
- E2E testing with fake camera input

## Configuration

### STUN/TURN Servers

Default STUN: `stun:stun.l.google.com:19302`

For production TURN servers, create `.env` file:
```
ICE_SERVERS=[{"urls":["turn:your-turn-server:3478"],"username":"user","credential":"pass"}]
```

### Video Constraints

- Resolution: 640×360 (target)
- Frame rate: 30 FPS (target)
- Bitrate: 600 kbps (max)
- Codec preference: H.264 > VP8 > VP9

## Development

### HTTPS Setup

The frontend uses HTTPS for WebRTC compatibility. The Vite config uses `vite-plugin-mkcert` for automatic SSL certificate generation.

### Testing Assets

Generate test video for E2E testing:
```bash
ffmpeg -f lavfi -i color=c=gray:rate=30:size=640x360 -pix_fmt yuv420p -t 5 web-app/test_assets/gray50.y4m
```

## API Endpoints

- `GET /health` - Health check
- `POST /offer` - WebRTC signaling (SDP offer → answer)

## DataChannel Protocol

- Name: `metrics`
- Protocol: `intensity-v1`
- Message format: `{"ts": timestamp, "intensity": 0-255, "intensity_norm": 0.0-1.0}`
- Settings: `ordered: false, maxRetransmits: 0`

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support  
- Safari: Works when H.264 is available
