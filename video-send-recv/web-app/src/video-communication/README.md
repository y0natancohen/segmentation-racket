# Video Communication API - TypeScript

A clean, reusable TypeScript API for video streaming and intensity analysis that can be used by other modules.

## Overview

The Video Communication API provides a simple interface for:
- WebRTC video connections
- Real-time intensity analysis
- Continuous video processing
- Connection management
- Custom intensity handlers
- Advanced intensity analysis

## Quick Start

```typescript
import { 
  initVideoSystem, 
  createConnection, 
  startCamera, 
  connectVideo, 
  setEventHandlers,
  getDefaultConfig 
} from './video-communication';

// Initialize the system
const config = getDefaultConfig();
initVideoSystem(config);

// Create a connection
const connectionId = 'my_connection';
await createConnection(connectionId);

// Set up event handlers
setEventHandlers({
  onIntensityUpdate: (id, metrics) => {
    console.log(`Intensity: ${metrics.avg_intensity.toFixed(1)}`);
  }
});

// Start camera and connect
const stream = await startCamera(connectionId);
await connectVideo(connectionId, stream);
```

## Core API Functions

### System Management

#### `initVideoSystem(config: VideoConfig): VideoCommunicationManager`
Initialize the video communication system.

**Parameters:**
- `config`: Configuration object

**Example:**
```typescript
import { getDefaultConfig } from './video-communication';

const config = getDefaultConfig();
config.serverUrl = 'http://localhost:8080';
initVideoSystem(config);
```

#### `cleanupVideoSystem(): Promise<void>`
Shutdown the entire video system and cleanup all connections.

### Connection Management

#### `createConnection(connectionId: string): Promise<VideoConnection>`
Create a new video connection.

**Parameters:**
- `connectionId`: Unique identifier for the connection

**Returns:**
- `VideoConnection` object with connection details

#### `startCamera(connectionId: string): Promise<MediaStream>`
Start camera and get media stream.

**Parameters:**
- `connectionId`: Connection identifier

**Returns:**
- `MediaStream` from the camera

#### `connectVideo(connectionId: string, stream: MediaStream): Promise<void>`
Connect to the video server.

**Parameters:**
- `connectionId`: Connection identifier
- `stream`: Media stream from camera

#### `disconnectVideo(connectionId: string): Promise<void>`
Disconnect a video connection and cleanup resources.

### Event Handling

#### `setEventHandlers(events: VideoConnectionEvents): void`
Set event handlers for video communication events.

**Parameters:**
- `events`: Object with event handler functions

**Example:**
```typescript
setEventHandlers({
  onIntensityUpdate: (connectionId, metrics) => {
    console.log(`Intensity: ${metrics.avg_intensity}`);
  },
  onConnectionStateChange: (connectionId, state) => {
    console.log(`Connection state: ${state}`);
  },
  onError: (connectionId, error) => {
    console.error(`Error: ${error.message}`);
  }
});
```

### Information and Statistics

#### `getConnectionInfo(connectionId: string): VideoConnection | undefined`
Get detailed information about a specific connection.

#### `getAllConnections(): VideoConnection[]`
Get information about all active connections.

#### `isConnectionActive(connectionId: string): boolean`
Check if a connection is currently active.

#### `getConnectionCount(): number`
Get the number of active connections.

## Advanced Features

### Intensity Analysis

```typescript
import { IntensityAnalyzer } from './video-communication';

const analyzer = new IntensityAnalyzer({
  lowThreshold: 15,
  highThreshold: 235,
  changeThreshold: 40,
  stabilityWindow: 10,
  maxHistory: 200
});

// Process intensity data
const stats = analyzer.processIntensity(connectionId, metrics);

// Get alerts
const alerts = analyzer.getAllAlerts();
```

### Video Application Class

```typescript
import { VideoApplication } from './video-communication';

const app = new VideoApplication();

// Initialize
await app.initialize();

// Add users
await app.addUser('user1');
await app.addUser('user2');

// Get statistics
const stats = app.getApplicationStats();

// Get alerts
const alerts = app.getAllAlerts();

// Shutdown
await app.shutdown();
```

### React Component Integration

```typescript
import { VideoComponent } from './video-communication';

class MyVideoComponent {
  private videoComponent: VideoComponent;

  constructor(userId: string) {
    this.videoComponent = new VideoComponent(userId);
  }

  async initialize() {
    await this.videoComponent.initialize();
  }

  async startVideo() {
    await this.videoComponent.startVideo();
  }

  getIntensityData() {
    return this.videoComponent.getIntensityData();
  }
}
```

## Data Structures

### `VideoConnection`
Contains information about a video connection.

```typescript
interface VideoConnection {
  id: string;
  isConnected: boolean;
  isStreaming: boolean;
  stats: ConnectionStats;
  lastIntensity: IntensityMetrics | null;
}
```

### `VideoConfig`
Configuration for video processing.

```typescript
interface VideoConfig {
  serverUrl: string;
  videoConstraints: VideoConstraints;
  dataChannelConfig: DataChannelConfig;
  reconnectInterval: number;
  maxReconnectAttempts: number;
}
```

### `IntensityMetrics`
Contains intensity analysis results.

```typescript
interface IntensityMetrics {
  ts: number;
  intensity: number; // 0-255
  intensity_norm: number; // 0.0-1.0
  avg_intensity: number; // 0-255 average
  avg_intensity_norm: number; // 0.0-1.0 average
  frame_count: number;
}
```

## Event Handlers

### `VideoConnectionEvents`
Interface for event handlers.

```typescript
interface VideoConnectionEvents {
  onConnectionStateChange?: (connectionId: string, state: RTCPeerConnectionState) => void;
  onIntensityUpdate?: (connectionId: string, metrics: IntensityMetrics) => void;
  onStatsUpdate?: (connectionId: string, stats: ConnectionStats) => void;
  onError?: (connectionId: string, error: Error) => void;
  onStreamReady?: (connectionId: string, stream: MediaStream) => void;
}
```

## Advanced Usage Examples

### Custom Intensity Analyzer

```typescript
class MyIntensityAnalyzer {
  private history: Map<string, number[]> = new Map();

  handleIntensity(connectionId: string, data: IntensityMetrics) {
    // Store history
    if (!this.history.has(connectionId)) {
      this.history.set(connectionId, []);
    }
    
    this.history.get(connectionId)!.push(data.avg_intensity);
    
    // Analyze patterns
    if (data.avg_intensity < 10) {
      console.log('ALERT: Very low intensity!');
    }
  }
}

// Usage
const analyzer = new MyIntensityAnalyzer();
setEventHandlers({
  onIntensityUpdate: analyzer.handleIntensity.bind(analyzer)
});
```

### Multi-User Management

```typescript
class VideoUserManager {
  private users: Map<string, string> = new Map(); // userId -> connectionId

  async addUser(userId: string): Promise<string> {
    const connectionId = `user_${userId}`;
    await createConnection(connectionId);
    this.users.set(userId, connectionId);
    return connectionId;
  }

  async removeUser(userId: string): Promise<void> {
    const connectionId = this.users.get(userId);
    if (connectionId) {
      await disconnectVideo(connectionId);
      this.users.delete(userId);
    }
  }
}
```

### Integration with React

```typescript
import React, { useEffect, useState } from 'react';
import { VideoComponent } from './video-communication';

const MyVideoComponent: React.FC<{ userId: string }> = ({ userId }) => {
  const [videoComponent] = useState(() => new VideoComponent(userId));
  const [intensity, setIntensity] = useState<number>(0);

  useEffect(() => {
    videoComponent.initialize();
    
    return () => {
      videoComponent.cleanup();
    };
  }, []);

  const startVideo = async () => {
    await videoComponent.startVideo();
  };

  const stopVideo = async () => {
    await videoComponent.stopVideo();
  };

  return (
    <div>
      <button onClick={startVideo}>Start Video</button>
      <button onClick={stopVideo}>Stop Video</button>
      <div>Intensity: {intensity.toFixed(1)}</div>
    </div>
  );
};
```

## Error Handling

The API includes robust error handling:

- **Connection failures**: Automatic cleanup and error reporting
- **Camera access errors**: Graceful handling of permission issues
- **Network errors**: Retry logic and fallback mechanisms
- **Resource cleanup**: Automatic cleanup of connections and resources

## Performance Considerations

- **Frame buffering**: Configurable number of frames to average
- **Memory management**: Automatic cleanup of old data
- **Connection pooling**: Efficient management of multiple connections
- **Event optimization**: Debounced event handling for performance

## Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Works with H.264 codec support
- **Mobile browsers**: Limited support (depends on WebRTC implementation)

## Dependencies

- **WebRTC APIs**: `RTCPeerConnection`, `MediaDevices`
- **TypeScript**: Type definitions and interfaces
- **React**: Optional integration (if using React components)

## Example Files

- `VideoCommunicationManager.ts`: Core implementation
- `VideoAPI.ts`: Simple API functions
- `IntensityAnalyzer.ts`: Advanced intensity analysis
- `ExampleUsage.ts`: Comprehensive usage examples
- `index.ts`: Main entry point with all exports
