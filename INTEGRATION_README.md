# Video Communication + Segmentation Integration

This document explains how to use the integrated video communication and segmentation system.

## Overview

The integration connects the Python segmentation system to the video communication interface, allowing:

1. **Video frames** are received from the WebRTC frontend
2. **Segmentation processing** runs on these frames
3. **Polygon data** is sent back to the frontend via WebSocket
4. **Frontend displays** both the original video and the segmentation polygon

## Architecture

```
Frontend (React/TypeScript)
    ↓ WebRTC Video Stream
Backend (Python)
    ↓ Frame Processing
Segmentation System (RVM)
    ↓ Polygon Generation
WebSocket Server
    ↓ Polygon Data
Frontend (Polygon Display)
```

## Files Created/Modified

### New Files:
- `segmentation/video_segmentation_integration.py` - Integration module
- `web-app/src/components/PolygonDisplay.tsx` - Frontend polygon display
- `run_integrated_system.py` - Main runner script
- `test_integration.py` - Test script

### Modified Files:
- `segmentation/segmentation.py` - Added video communication option
- `video_send_recv/server/video_communication.py` - Added frame processing
- `video_send_recv/server/server.py` - Added WebSocket support
- `video_send_recv/web-app/src/App.tsx` - Added polygon display

## Usage

### 1. Start the Integrated System

```bash
# Start the integrated system
python run_integrated_system.py

# With custom options
python run_integrated_system.py \
  --host localhost \
  --port 8080 \
  --polygon_threshold 0.5 \
  --polygon_min_area 2000
```

### 2. Start the Frontend

```bash
cd video_send_recv/web-app
npm run dev
```

### 3. Open the Application

1. Open `https://localhost:3000`
2. Click "Start Camera" to enable video
3. Click "Connect" to start WebRTC connection
4. The segmentation system will automatically process frames
5. You'll see both the original video and the polygon overlay

## Configuration Options

### Segmentation Options:
- `--model`: Segmentation model file (default: rvm_mobilenetv3.pth)
- `--device`: Device for processing (cuda/cpu/auto)
- `--polygon_threshold`: Threshold for polygon extraction (0.0-1.0)
- `--polygon_min_area`: Minimum area for valid polygons
- `--polygon_epsilon`: Polygon simplification factor

### Server Options:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 8080)

## API Integration

### For Other Modules

The integration provides a clean API for other modules to use:

```python
from segmentation.video_segmentation_integration import VideoSegmentationProcessor

# Create processor
processor = VideoSegmentationProcessor(segmentation_args)
await processor.initialize()

# Start processing
await processor.start_processing(connection_id)

# Set up polygon callback
def handle_polygon(polygon_data):
    print(f"Polygon: {polygon_data['polygon']}")

processor.polygon_callback = handle_polygon
```

### Frontend Integration

```typescript
import PolygonDisplay from './components/PolygonDisplay';

// Use in React component
<PolygonDisplay
  width={640}
  height={360}
  style={{ border: '2px solid #333' }}
/>
```

## Data Flow

1. **Frontend** captures camera video via WebRTC
2. **Backend** receives video frames through WebRTC
3. **Segmentation** processes frames and generates polygons
4. **WebSocket** sends polygon data to frontend
5. **Frontend** displays polygon overlay on canvas

## Polygon Data Format

```json
{
  "connection_id": "conn_1234567890",
  "polygon": [[x1, y1], [x2, y2], ...],
  "timestamp": 1234567890.123,
  "frame_shape": [height, width]
}
```

## Troubleshooting

### Common Issues:

1. **WebSocket Connection Failed**
   - Check if the server is running on the correct port
   - Verify WebSocket endpoint is accessible

2. **No Polygon Data**
   - Check segmentation model is loaded correctly
   - Verify frame processing is working
   - Check polygon threshold settings

3. **Performance Issues**
   - Reduce polygon complexity with higher epsilon
   - Increase minimum area threshold
   - Use GPU acceleration if available

### Debug Mode:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_integrated_system.py
```

## Testing

Run the integration test:

```bash
python test_integration.py
```

This will verify that:
- Video system initializes correctly
- Segmentation processor works
- Frame processing functions properly

## Development

### Adding New Features:

1. **New Segmentation Models**: Modify `video_segmentation_integration.py`
2. **Custom Polygon Processing**: Extend `VideoSegmentationProcessor`
3. **Frontend Visualization**: Modify `PolygonDisplay.tsx`
4. **Data Channels**: Update WebSocket handlers in `server.py`

### Extending the API:

```python
# Add custom polygon processing
class CustomSegmentationProcessor(VideoSegmentationProcessor):
    def _process_frame(self, frame_data):
        # Custom processing logic
        super()._process_frame(frame_data)
        # Additional processing
```

## Performance Considerations

- **Frame Rate**: Segmentation runs at ~30 FPS
- **Memory Usage**: Frames are processed in real-time
- **GPU Usage**: CUDA acceleration recommended
- **Network**: WebSocket for polygon data, WebRTC for video

## Security Notes

- WebSocket connections are not authenticated
- Video streams are not encrypted (use HTTPS in production)
- Polygon data is sent in plain text JSON

For production use, consider:
- WebSocket authentication
- Video stream encryption
- Secure polygon data transmission
