# Integrated Phaser Game with Video Communication and Segmentation

This project combines a Phaser.js physics game with real-time video communication and AI-powered segmentation to create an interactive experience where the game responds to detected objects in the video stream.

## Architecture

The system consists of three main components:

1. **Phaser Game** (`phaser-matter-game/`) - Frontend game with physics simulation
2. **Video Communication System** (`video_send_recv/`) - WebRTC video streaming
3. **Segmentation System** (`segmentation/`) - AI-powered object detection and polygon generation

## Features

- 🎮 **Physics Game**: Bouncing balls with realistic physics using Phaser.js and Matter.js
- 📹 **Real-time Video**: WebRTC video streaming from webcam to Python backend
- 🤖 **AI Segmentation**: Real-time object detection and polygon generation
- 🔄 **Dynamic Platforms**: Game platforms that update based on detected objects
- 📊 **Live Monitoring**: FPS counters and connection status displays

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Webcam access

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Phaser game dependencies
cd phaser-matter-game
npm install
cd ..
```

### 2. Run the Integrated System

#### Option A: Using the Test Script (Recommended)

```bash
python test_integrated_phaser.py
```

This will start:
- WebRTC server on `http://localhost:8080`
- Phaser game on `http://localhost:5173`
- Segmentation system ready for processing

#### Option B: Manual Setup

1. **Start the Python Backend**:
```bash
python run_integrated_system.py --host localhost --port 8080
```

2. **Start the Phaser Game** (in a separate terminal):
```bash
cd phaser-matter-game
npm run dev
```

3. **Open the Game**: Navigate to `http://localhost:5173`

### 3. Using the System

1. **Allow Camera Access**: When prompted, allow the browser to access your webcam
2. **Wait for Connections**: The game will show connection status indicators
3. **Play the Game**: Balls will fall and bounce on the dynamic platform
4. **Move in Front of Camera**: The platform will update based on detected objects

## System Components

### Phaser Game (`phaser-matter-game/src/main.ts`)

- **MainScene**: Main game scene with physics simulation
- **VideoCommunicationManager**: Handles WebRTC video streaming
- **PolygonWebSocketManager**: Receives polygon data from Python backend
- **Dynamic Platform**: Updates based on segmentation results

### Video Communication (`video_send_recv/`)

- **WebRTC Server**: Handles video streaming and signaling
- **Video API**: RESTful API for connection management
- **WebSocket Support**: Real-time polygon data transmission

### Segmentation System (`segmentation/`)

- **AI Model**: RVM (Robust Video Matting) for object detection
- **Polygon Generation**: Converts segmentation masks to polygons
- **Real-time Processing**: Processes video frames as they arrive

## Configuration

### Video Settings

The video communication can be configured in `main.ts`:

```typescript
const videoConfig: VideoConfig = {
  serverUrl: 'http://localhost:8080',
  videoConstraints: {
    width: { ideal: GAME_WIDTH, max: 1920 },
    height: { ideal: GAME_HEIGHT, max: 1080 },
    frameRate: { ideal: 30, max: 60 },
    facingMode: 'user'
  },
  // ... other settings
};
```

### Segmentation Settings

Configure the segmentation system in `run_integrated_system.py`:

```bash
python run_integrated_system.py \
  --model rvm_mobilenetv3.pth \
  --device auto \
  --polygon_threshold 0.5 \
  --polygon_min_area 2000 \
  --polygon_epsilon 0.015
```

## Troubleshooting

### Common Issues

1. **Camera Access Denied**
   - Ensure browser has camera permissions
   - Check if another application is using the camera

2. **WebRTC Connection Failed**
   - Verify the Python backend is running on port 8080
   - Check firewall settings
   - Ensure STUN servers are accessible

3. **No Polygon Data**
   - Check if segmentation system is running
   - Verify WebSocket connection to `ws://localhost:8080/polygon`
   - Ensure video is being processed by the backend

4. **Game Performance Issues**
   - Reduce video resolution in configuration
   - Lower frame rate settings
   - Check browser performance

### Debug Information

The game displays several status indicators:

- **Rectangle FPS**: Rate of polygon updates
- **Polygon Status**: WebSocket connection status
- **Video Status**: WebRTC connection status

### Logs

Check the console for detailed logging:

```bash
# Python backend logs
python run_integrated_system.py

# Browser console (F12)
# Look for connection and data flow messages
```

## Development

### Adding New Features

1. **New Game Objects**: Add to `MainScene` class
2. **Video Processing**: Extend `VideoCommunicationManager`
3. **Segmentation**: Modify `video_segmentation_integration.py`

### Testing

```bash
# Run unit tests
cd phaser-matter-game
npm test

# Run integration tests
python test_integrated_phaser.py
```

## API Reference

### VideoCommunicationManager

```typescript
// Create connection
await videoManager.createConnection(connectionId);

// Start camera
const stream = await videoManager.startCamera(connectionId);

// Connect to server
await videoManager.connect(connectionId, stream);
```

### PolygonWebSocketManager

```typescript
// Set up event handlers
polygonManager.setEventHandlers({
  onPolygonData: (data) => {
    // Handle polygon data
  },
  onConnectionStateChange: (connected) => {
    // Handle connection state
  }
});

// Connect
await polygonManager.connect();
```

## Performance Optimization

### Video Settings

- Use appropriate resolution for your use case
- Adjust frame rate based on processing requirements
- Consider using hardware acceleration

### Game Settings

- Limit number of physics objects
- Use object pooling for frequently created/destroyed objects
- Optimize rendering with appropriate depth layers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Phaser.js**: Game framework
- **Matter.js**: Physics engine
- **RVM**: Robust Video Matting for segmentation
- **WebRTC**: Real-time communication
