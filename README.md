# Segment Project

A dual-process architecture that combines real-time video segmentation with a Phaser.js game engine for interactive polygon visualization.

## Project Structure

```
segment_project/
├── segmentation/           # All segmentation and application code
│   ├── segmentation.py    # Main segmentation module
│   ├── segmentation_polygon_bridge.py  # Bridge to polygon system
│   ├── polygon_generator.py  # Polygon movement generator
│   ├── demo_segmentation_polygon_integration.py
│   ├── rvm/               # RVM (Robust Video Matting) model
│   ├── models/            # Pre-trained models
│   └── polygon_config/    # Polygon configuration files
├── tests/                 # Test suite
│   ├── run_tests.sh       # Test runner
│   └── test_*.py          # Individual test files
├── phaser-matter-game/    # Phaser.js game frontend
└── output/                # Generated output files
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended for segmentation)
- Virtual environment (recommended)

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

### 1. Polygon Generator (Standalone)

Run the polygon generator to create moving polygons:

```bash
source .venv/bin/activate
python segmentation/polygon_generator.py segmentation/polygon_config/rectangle.json
```

### 2. Phaser.js Game

Start the game frontend:

```bash
cd phaser-matter-game
npm run dev
```

Then open your browser to: `http://localhost:5173`

### 3. Real-time Segmentation

Run with real-time video segmentation:

```bash
source .venv/bin/activate
python segmentation/segmentation.py --web_display --show_alpha --polygon_bridge
```

## Testing

Run the complete test suite:

```bash
./tests/run_tests.sh
```

Or run individual tests:

```bash
# Python unit tests
python -m pytest tests/test_polygon_generator.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# FPS performance tests
python tests/test_pipeline_fps.py
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

The system uses a dual-process architecture:

1. **Python Process**: Generates polygon movement data and sends it via WebSocket
2. **TypeScript/Phaser Process**: Receives data and renders polygons in the browser

Communication happens through WebSocket messages containing:
- `position`: {x, y} coordinates
- `vertices`: Array of polygon vertices
- `rotation`: Rotation angle in radians

## Performance

- Target FPS: 60 FPS for smooth animation
- WebSocket latency: < 1ms
- Polygon generation: < 1ms per frame
- Total pipeline latency: < 16ms

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 5173 and 8765 are available
2. **CUDA errors**: Install CUDA-compatible PyTorch version
3. **WebSocket connection failed**: Check if Python generator is running
4. **Game not loading**: Verify Node.js server is running

### Debug Mode

Enable debug logging:
```bash
export DEBUG=1
python segmentation/polygon_generator.py segmentation/polygon_config/rectangle.json
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
