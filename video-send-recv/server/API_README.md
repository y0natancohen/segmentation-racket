# Video Communication API

A clean, reusable API for video streaming and intensity analysis that can be used by other modules.

## Overview

The Video Communication API provides a simple interface for:
- WebRTC video connections
- Real-time intensity analysis
- Continuous video processing
- Connection management
- Custom intensity handlers

## Quick Start

```python
from video_api import init_video_system, connect_video, register_intensity_handler
from video_communication import VideoConfig, IntensityData

# Initialize the system
config = VideoConfig(max_samples=30, frame_timeout=5.0)
init_video_system(config)

# Create a connection
connection_id = "my_connection"
answer = await connect_video(connection_id, sdp, sdp_type)

# Register intensity handler
def my_handler(data: IntensityData):
    print(f"Intensity: {data.average_intensity:.1f}")

register_intensity_handler(connection_id, my_handler)
```

## Core API Functions

### System Management

#### `init_video_system(config: VideoConfig = None) -> VideoCommunicationManager`
Initialize the video communication system.

**Parameters:**
- `config`: Optional configuration object

**Example:**
```python
from video_communication import VideoConfig

config = VideoConfig(
    max_samples=30,      # Number of frames to average
    frame_timeout=5.0,   # Timeout for frame reception
    log_interval=1.0     # Log metrics every N seconds
)
init_video_system(config)
```

#### `shutdown_video_system()`
Shutdown the entire video system and cleanup all connections.

### Connection Management

#### `connect_video(connection_id: str, sdp: str, sdp_type: str) -> Dict[str, str]`
Handle WebRTC offer and create answer.

**Parameters:**
- `connection_id`: Unique identifier for the connection
- `sdp`: SDP offer string
- `sdp_type`: Type of SDP (usually "offer")

**Returns:**
- Dictionary with "sdp" and "type" keys for the answer

**Example:**
```python
answer = await connect_video("conn_123", sdp_offer, "offer")
# Returns: {"sdp": "...", "type": "answer"}
```

#### `disconnect_video(connection_id: str)`
Disconnect a video connection and cleanup resources.

### Intensity Handling

#### `register_intensity_handler(connection_id: str, handler: Callable[[IntensityData], None])`
Register a callback function to receive intensity data.

**Parameters:**
- `connection_id`: Connection to monitor
- `handler`: Function that receives `IntensityData` objects

**Example:**
```python
def intensity_handler(data: IntensityData):
    print(f"Current: {data.current_intensity:.1f}")
    print(f"Average: {data.average_intensity:.1f}")
    print(f"Frames: {data.frame_count}")

register_intensity_handler("conn_123", intensity_handler)
```

#### `unregister_intensity_handler(connection_id: str)`
Remove intensity handler for a connection.

### Information and Statistics

#### `get_connection_info(connection_id: str) -> Optional[Dict[str, Any]]`
Get detailed information about a specific connection.

**Returns:**
```python
{
    "connection_id": "conn_123",
    "uptime": 45.2,
    "frames_received": 1350,
    "messages_sent": 1350,
    "last_activity": 1640995200.0
}
```

#### `get_all_connections() -> Dict[str, Dict[str, Any]]`
Get information about all active connections.

#### `is_connection_active(connection_id: str) -> bool`
Check if a connection is currently active.

#### `get_connection_count() -> int`
Get the number of active connections.

## Data Structures

### `IntensityData`
Contains intensity analysis results for a frame.

```python
@dataclass
class IntensityData:
    timestamp: float              # Unix timestamp
    current_intensity: float     # Current frame intensity (0-255)
    current_normalized: float    # Current normalized (0.0-1.0)
    average_intensity: float    # Rolling average (0-255)
    average_normalized: float   # Rolling average normalized (0.0-1.0)
    frame_count: int            # Number of frames in average
    connection_id: str          # Connection identifier
```

### `VideoConfig`
Configuration for video processing.

```python
@dataclass
class VideoConfig:
    max_samples: int = 30       # Frames to average
    frame_timeout: float = 5.0  # Frame reception timeout
    log_interval: float = 1.0   # Logging interval
```

## Advanced Usage Examples

### Custom Intensity Analyzer

```python
class MyIntensityAnalyzer:
    def __init__(self):
        self.history = {}
    
    def handle_intensity(self, data: IntensityData):
        # Store history
        if data.connection_id not in self.history:
            self.history[data.connection_id] = []
        
        self.history[data.connection_id].append(data.average_intensity)
        
        # Analyze patterns
        if data.average_intensity < 10:
            print("ALERT: Very low intensity!")
        elif data.average_intensity > 240:
            print("ALERT: Very high intensity!")
    
    def get_summary(self, connection_id: str):
        if connection_id not in self.history:
            return None
        
        values = self.history[connection_id]
        return {
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

# Usage
analyzer = MyIntensityAnalyzer()
register_intensity_handler("conn_123", analyzer.handle_intensity)
```

### Multi-User Management

```python
class VideoUserManager:
    def __init__(self):
        self.users = {}  # user_id -> connection_id mapping
    
    async def add_user(self, user_id: str, sdp: str, sdp_type: str):
        connection_id = f"user_{user_id}"
        answer = await connect_video(connection_id, sdp, sdp_type)
        self.users[user_id] = connection_id
        return answer
    
    async def remove_user(self, user_id: str):
        if user_id in self.users:
            connection_id = self.users[user_id]
            await disconnect_video(connection_id)
            del self.users[user_id]
    
    def get_user_connection(self, user_id: str):
        return self.users.get(user_id)
```

### Integration with Web Server

```python
from aiohttp import web

async def handle_offer(request):
    data = await request.json()
    connection_id = f"conn_{int(time.time() * 1000)}"
    
    # Use the video API
    answer = await connect_video(connection_id, data['sdp'], data['type'])
    
    return web.json_response(answer)

# Register intensity handler
def log_intensity(data: IntensityData):
    logger.info(f"Intensity for {data.connection_id}: {data.average_intensity:.1f}")

# In your application startup
init_video_system()
register_intensity_handler("conn_123", log_intensity)
```

## Error Handling

The API includes robust error handling:

- **Connection timeouts**: Automatically handled with configurable timeouts
- **Frame processing errors**: Individual frame errors don't stop processing
- **Data channel errors**: Graceful handling of communication failures
- **Resource cleanup**: Automatic cleanup of connections and resources

## Performance Considerations

- **Frame buffering**: Configurable number of frames to average
- **Timeout handling**: Prevents hanging on slow connections
- **Memory management**: Automatic cleanup of old data
- **Logging control**: Configurable logging intervals

## Thread Safety

The API is designed to be thread-safe and can be used in multi-threaded applications. All operations are asynchronous and use proper locking mechanisms.

## Dependencies

- `aiortc`: WebRTC implementation
- `asyncio`: Asynchronous programming
- `dataclasses`: Data structure definitions
- `typing`: Type hints
- `logging`: Logging functionality

## Example Files

- `example_usage.py`: Comprehensive usage examples
- `video_communication.py`: Core implementation
- `video_api.py`: Simple API functions
- `server.py`: Web server integration example
