# Segment Project

A dual-process architecture that combines real-time video segmentation with a Phaser.js game engine for interactive polygon visualization.

## Project Structure

```
segment_project/
├── segmentation_server.py  # WebSocket segmentation server (main server)
├── debug_segmentation.py   # Browser-based debug tool (4-panel visualization)
├── test_camera.py          # Simple camera test tool
├── segmentation/           # Segmentation modules
│   ├── segmentation.py    # Main segmentation module
│   ├── polygon_generator.py  # Polygon movement generator
│   ├── rvm/               # RVM (Robust Video Matting) model
│   ├── models/            # Pre-trained models
│   └── polygon_config/    # Polygon configuration files
├── tests/                 # Test suite
│   ├── test_segmentation_server.py  # Server tests
│   └── test_matte_to_polygon.py     # Polygon generation tests
├── phaser-matter-game/    # Phaser.js game frontend
│   └── src/main.ts        # Main game logic
└── models/                # Model weights (auto-downloaded)
```

## Quick Start

**1. Start the segmentation server:**
```bash
source .venv/bin/activate
python segmentation_server.py
```

**2. Start the Phaser.js game (in another terminal):**
```bash
cd phaser-matter-game
npm run dev
```

**3. Open browser:**
- Game: `http://localhost:5173`
- Debug tool: `http://127.0.0.1:5001` (run `python debug_segmentation.py` first)

## Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended for segmentation)
- Virtual environment (recommended)
- Webcam/camera access

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd segment_project
   ```

2. **Set up Python environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Node.js environment:**
   ```bash
   cd phaser-matter-game
   npm install
   cd ..
   ```

## Running the Application

### 1. Segmentation WebSocket Server

Start the segmentation server that processes video frames and sends polygons via WebSocket:

```bash
source .venv/bin/activate
python segmentation_server.py
```

**Options:**
- `--host HOST` - Server host (default: localhost)
- `--port PORT` - Server port (default: 8765)
- `--device DEVICE` - Device: cuda, cpu, or auto (default: auto)
- `--fp16` - Use FP16 precision on CUDA
- `--dsr FLOAT` - RVM downsample ratio (default: 0.25)
- `--polygon_threshold FLOAT` - Polygon threshold (default: 0.5)
- `--polygon_min_area INT` - Minimum polygon area (default: 2000)
- `--polygon_epsilon FLOAT` - Douglas-Peucker epsilon ratio (default: 0.002)
- `--max_workers INT` - Thread pool size (default: 2)
- `--debug` - Enable DEBUG-level logging
- `--tracemalloc` - Enable memory tracing

**Example:**
```bash
python segmentation_server.py --host 0.0.0.0 --port 8765 --device cuda --fp16
```

The server will listen on `ws://localhost:8765` and accept JPEG frames via WebSocket, returning polygon data as JSON.

### 2. Phaser.js Game Frontend

Start the game frontend that connects to the segmentation server:

```bash
cd phaser-matter-game
npm run dev
```

Then open your browser to: `http://localhost:5173`

The game will automatically connect to the segmentation server at `ws://localhost:8765` and display polygons in real-time.

### 3. Debug Tools

#### Debug Segmentation (Browser-based)

Visual debugger for the segmentation pipeline with 4-panel display:

```bash
source .venv/bin/activate
python debug_segmentation.py
```

**Options:**
- `--cam INT` - Camera index (default: 0)
- `--width INT` - Capture width (default: 640)
- `--height INT` - Capture height (default: 360)
- `--port INT` - Web server port (default: 5001)
- `--host HOST` - Web server host (default: 127.0.0.1)
- `--model_path PATH` - Model path (default: models/rvm_mobilenetv3.pth)
- `--device DEVICE` - Device: cuda, cpu, or auto (default: auto)
- `--fp16` - Use FP16 precision
- `--dsr FLOAT` - RVM downsample ratio (default: 0.25)
- `--polygon_threshold FLOAT` - Polygon threshold (default: 0.5)
- `--polygon_min_area INT` - Minimum polygon area (default: 2000)
- `--polygon_epsilon FLOAT` - Douglas-Peucker epsilon (default: 0.002)
- `--jpeg_quality INT` - JPEG quality 0-100 (default: 70)
- `--save_dir DIR` - Save debug frames to directory

**Example:**
```bash
python debug_segmentation.py --cam 0 --port 5001 --device cuda
```

Then open your browser to: `http://127.0.0.1:5001`

**Display Panels:**
1. **Top Left**: Original frame
2. **Top Right**: Alpha matte heatmap (JET colormap)
3. **Bottom Left**: Thresholded binary mask
4. **Bottom Right**: Original frame with polygon overlay

**Features:**
- Real-time FPS and frame statistics
- Timing breakdown (decode, inference, polygon generation)
- Alpha matte statistics (min/max/mean)
- Save frame button
- Dark theme interface

#### Test Camera (Browser-based)

Simple camera test tool to verify camera access:

```bash
source .venv/bin/activate
python test_camera.py
```

**Options:**
- `--cam INT` - Camera index (default: 0)
- `--width INT` - Capture width (default: 640)
- `--height INT` - Capture height (default: 480)
- `--port INT` - Web server port (default: 5000)
- `--host HOST` - Web server host (default: 127.0.0.1)
- `--force` - Try to open camera even if it appears to be in use

**Example:**
```bash
python test_camera.py --cam 0 --port 5000
```

Then open your browser to: `http://127.0.0.1:5000`

**Features:**
- Live camera feed
- FPS and frame count display
- Resolution information
- Save frame button
- Camera availability detection
- Permission diagnostics

### 4. Polygon Generator (Standalone)

Run the polygon generator to create moving polygons:

```bash
source .venv/bin/activate
python segmentation/polygon_generator.py segmentation/polygon_config/rectangle.json
```

## Testing

Run the complete test suite:

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

**Test Results:**
- ✅ 8 matte_to_polygon tests
- ✅ 4 SegmentationSession tests  
- ✅ 1 WebSocket round-trip integration test

**Individual Test Files:**
```bash
# All segmentation tests
python -m pytest tests/test_segmentation_server.py -v
python -m pytest tests/test_matte_to_polygon.py -v

# Run with coverage
python -m pytest tests/ --cov=segmentation --cov=segmentation_server
```

## Configuration

### Polygon Configuration

Edit polygon shapes in `segmentation/polygon_config/`:
- `rectangle.json` - Rectangle polygon
- `triangle.json` - Triangle polygon  
- `octagon.json` - Octagon polygon

### Game Configuration

The Phaser.js game runs on port 5173 by default. The Python WebSocket server runs on port 8765.

## Architecture

The system uses a multi-process architecture:

1. **Segmentation Server** (`segmentation_server.py`):
   - WebSocket server that accepts JPEG frames
   - Runs RVM (Robust Video Matting) model inference
   - Generates polygons from segmentation masks
   - Sends polygon data as JSON via WebSocket

2. **Phaser.js Game Frontend**:
   - Receives polygon data via WebSocket
   - Renders polygons in real-time using Phaser.js and Matter.js
   - Interactive physics-based visualization

3. **Debug Tools**:
   - `debug_segmentation.py`: Browser-based 4-panel visualization
   - `test_camera.py`: Simple camera test tool

**WebSocket Message Format:**
```json
{
  "polygon": [[x1, y1], [x2, y2], ...],
  "timestamp": 1234567890.123,
  "original_image_size": [height, width]
}
```

**Communication Flow:**
1. Browser captures video frames from webcam
2. Frames are encoded as JPEG and sent to segmentation server
3. Server processes frames and generates polygons
4. Polygons are sent back as JSON
5. Phaser.js game renders polygons in real-time

## Performance

**Segmentation Server:**
- Inference: ~7-15ms on CUDA (depending on resolution)
- Polygon generation: ~0-1ms
- Total frame processing: ~10-20ms
- Target: 30-60 FPS depending on hardware

**WebSocket:**
- Message latency: < 1ms (local network)
- Frame encoding: ~0.5-1ms (JPEG)
- JSON serialization: < 0.1ms

**Browser Display:**
- MJPEG streaming: ~30 FPS
- Real-time stats update: 500ms intervals
- Low latency visualization

**Hardware Recommendations:**
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CPU**: Multi-core CPU for thread pool execution
- **RAM**: 2GB+ for model and frame buffers
- **Camera**: USB webcam or built-in camera

## Troubleshooting

### Common Issues

1. **Port conflicts**: 
   - Segmentation server: port 8765
   - Debug segmentation: port 5001
   - Test camera: port 5000
   - Phaser game: port 5173
   - Ensure all ports are available

2. **Camera access issues**:
   ```bash
   # Check camera permissions
   ls -la /dev/video*
   
   # Fix permissions temporarily
   sudo chmod 666 /dev/video0
   
   # Fix permissions permanently
   sudo usermod -a -G video $USER
   # Then log out and back in
   
   # Check if camera is in use
   lsof /dev/video0
   ```

3. **CUDA errors**: 
   - Install CUDA-compatible PyTorch version
   - Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
   - Use `--device cpu` to fall back to CPU

4. **WebSocket connection failed**: 
   - Check if segmentation server is running
   - Verify WebSocket URL matches server host/port
   - Check firewall settings

5. **Browser display not working**:
   - Ensure Flask is installed: `pip install flask`
   - Check browser console for errors
   - Try accessing the URL directly in browser
   - Verify camera is accessible (use `test_camera.py` first)

6. **Model not found**:
   - Model will auto-download on first run
   - Or manually download: `models/rvm_mobilenetv3.pth`
   - Check model path in arguments

### Debug Mode

**Segmentation Server:**
```bash
python segmentation_server.py --debug --tracemalloc
```

**Debug Segmentation:**
```bash
python debug_segmentation.py --save_dir debug_output
```

**Test Camera:**
```bash
python test_camera.py --force  # Force camera open even if in use
```

## Development

### Adding New Polygon Shapes

1. Create a new JSON config in `segmentation/polygon_config/`
2. Run the polygon generator with the new config:
   ```bash
   python segmentation/polygon_generator.py segmentation/polygon_config/your_shape.json
   ```

### Modifying the Game

The Phaser.js game is in `phaser-matter-game/src/main.ts`. Key files:
- `main.ts` - Main game logic
- `style.css` - Game styling

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
