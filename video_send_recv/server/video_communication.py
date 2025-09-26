"""
Video Communication API - Reusable module for video streaming and intensity analysis.

This module provides a clean API for video communication that can be used by other modules.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

from aiortc import RTCPeerConnection, RTCSessionDescription
from media import mean_intensity
from metrics import MetricsCollector, ConnectionMetrics

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    max_samples: int = 2  # Number of frames to average
    frame_timeout: float = 0.1#5.0  # Timeout for frame reception
    log_interval: float = 1.0  # Log metrics every N seconds


@dataclass
class IntensityData:
    """Intensity analysis results."""
    timestamp: float
    current_intensity: float  # 0-255
    current_normalized: float  # 0.0-1.0
    average_intensity: float  # 0-255
    average_normalized: float  # 0.0-1.0
    frame_count: int
    connection_id: str


class VideoCommunicationManager:
    """Manages video communication and intensity analysis."""
    
    def __init__(self, config: VideoConfig = None):
        self.config = config or VideoConfig()
        self.connections: Dict[str, RTCPeerConnection] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.data_channels: Dict[str, Any] = {}
        self.intensity_callbacks: Dict[str, Callable[[IntensityData], None]] = {}
        self.frame_processors: Dict[str, Callable[[Dict[str, Any]], None]] = {}
        self.metrics_collector = MetricsCollector()
        
    async def init_connection(self, connection_id: str) -> RTCPeerConnection:
        """Initialize a new WebRTC connection."""
        logger.info(f"Initializing connection {connection_id}")
        
        pc = RTCPeerConnection()
        self.connections[connection_id] = pc
        self.connection_metrics[connection_id] = ConnectionMetrics(connection_id)
        
        # Set up event handlers
        self._setup_connection_handlers(pc, connection_id)
        
        return pc
    
    def _setup_connection_handlers(self, pc: RTCPeerConnection, connection_id: str):
        """Set up event handlers for a connection."""
        
        @pc.on("track")
        async def on_track(track):
            """Handle incoming video track."""
            logger.info(f"Received {track.kind} track from {connection_id}")
            
            if track.kind == "video":
                # Start processing in background
                asyncio.create_task(self._process_video_track(track, connection_id))
        
        @pc.on("datachannel")
        async def on_datachannel(channel):
            """Handle data channel."""
            logger.info(f"Received data channel: {channel.label} from {connection_id}")
            self.data_channels[connection_id] = channel
            
            @channel.on("open")
            async def on_open():
                logger.info(f"Data channel opened: {channel.label}")
            
            @channel.on("close")
            async def on_close():
                logger.info(f"Data channel closed: {channel.label}")
                if connection_id in self.data_channels:
                    del self.data_channels[connection_id]
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            """Handle connection state changes."""
            logger.info(f"Connection state changed to {pc.connectionState} for {connection_id}")
            
            if pc.connectionState in ["closed", "failed", "disconnected"]:
                await self.cleanup_connection(connection_id)
    
    async def _process_video_track(self, track, connection_id: str):
        """Process video track and calculate intensity."""
        logger.info(f"Starting video processing for {connection_id}")
        
        intensity_values = []
        
        try:
            while True:
                try:
                    # Receive frame with timeout
                    frame = await asyncio.wait_for(track.recv(), timeout=self.config.frame_timeout)
                    self.metrics_collector.record_frame_processed()
                    self.connection_metrics[connection_id].record_frame()
                    
                    # Calculate intensity
                    intensity_255, intensity_norm = mean_intensity(frame)
                    
                    # Add to intensity values for averaging
                    intensity_values.append(intensity_255)
                    if len(intensity_values) > self.config.max_samples:
                        intensity_values.pop(0)
                    
                    # Calculate average intensity
                    avg_intensity = sum(intensity_values) / len(intensity_values)
                    avg_intensity_norm = avg_intensity / 255.0
                    
                    # Create intensity data
                    intensity_data = IntensityData(
                        timestamp=time.time(),
                        current_intensity=intensity_255,
                        current_normalized=intensity_norm,
                        average_intensity=avg_intensity,
                        average_normalized=avg_intensity_norm,
                        frame_count=len(intensity_values),
                        connection_id=connection_id
                    )
                    
                    # Send data via data channel
                    await self._send_intensity_data(connection_id, intensity_data)
                    
                    # Call registered callback
                    if connection_id in self.intensity_callbacks:
                        try:
                            self.intensity_callbacks[connection_id](intensity_data)
                        except Exception as e:
                            logger.error(f"Error in intensity callback: {e}")
                    
                    # Call frame processor if registered
                    if connection_id in self.frame_processors:
                        try:
                            # Convert frame to numpy array for processing
                            frame_array = frame.to_ndarray(format="rgb24")
                            frame_data = {
                                'frame': frame_array,
                                'timestamp': time.time(),
                                'connection_id': connection_id,
                                'frame_shape': frame_array.shape
                            }
                            self.frame_processors[connection_id](frame_data)
                        except Exception as e:
                            logger.error(f"Error in frame processor: {e}")
                    
                    # Log metrics periodically
                    if self.metrics_collector.should_log():
                        self.metrics_collector.log_metrics()
                        logger.info(f"Connection {connection_id}: Avg intensity = {avg_intensity:.1f}")
                
                except asyncio.TimeoutError:
                    logger.warning(f"Frame timeout for connection {connection_id}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing frame for {connection_id}: {e}")
                    raise e
                    # continue
                
        except Exception as e:
            logger.error(f"Fatal error in video processing for {connection_id}: {e}")
    
    async def _send_intensity_data(self, connection_id: str, data: IntensityData):
        """Send intensity data via data channel."""
        data_channel = self.data_channels.get(connection_id)
        
        if data_channel and data_channel.readyState == "open":
            metrics_data = {
                "ts": data.timestamp,
                "intensity": data.current_intensity,
                "intensity_norm": data.current_normalized,
                "avg_intensity": data.average_intensity,
                "avg_intensity_norm": data.average_normalized,
                "frame_count": data.frame_count
            }
            
            try:
                data_channel.send(json.dumps(metrics_data))
                self.metrics_collector.record_message_sent()
                self.connection_metrics[connection_id].record_message_sent()
            except Exception as e:
                logger.warning(f"Failed to send intensity data: {e}")
    
    def register_intensity_callback(self, connection_id: str, callback: Callable[[IntensityData], None]):
        """Register a callback for intensity data."""
        self.intensity_callbacks[connection_id] = callback
        logger.info(f"Registered intensity callback for {connection_id}")
    
    def unregister_intensity_callback(self, connection_id: str):
        """Unregister intensity callback."""
        if connection_id in self.intensity_callbacks:
            del self.intensity_callbacks[connection_id]
            logger.info(f"Unregistered intensity callback for {connection_id}")
    
    async def cleanup_connection(self, connection_id: str):
        """Clean up connection resources."""
        try:
            if connection_id in self.connections:
                pc = self.connections[connection_id]
                await pc.close()
                del self.connections[connection_id]
                logger.info(f"Cleaned up connection {connection_id}")
            
            if connection_id in self.connection_metrics:
                del self.connection_metrics[connection_id]
                
            if connection_id in self.data_channels:
                del self.data_channels[connection_id]
                
            if connection_id in self.intensity_callbacks:
                del self.intensity_callbacks[connection_id]
                
        except Exception as e:
            logger.warning(f"Error during cleanup of connection {connection_id}: {e}")
    
    def get_connection_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a connection."""
        if connection_id not in self.connection_metrics:
            return None
            
        metrics = self.connection_metrics[connection_id]
        return {
            "connection_id": connection_id,
            "uptime": metrics.get_uptime(),
            "frames_received": metrics.frames_received,
            "messages_sent": metrics.messages_sent,
            "last_activity": metrics.last_activity
        }
    
    def get_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connections."""
        return {
            conn_id: self.get_connection_stats(conn_id)
            for conn_id in self.connections.keys()
        }
    
    async def cleanup_all_connections(self):
        """Clean up all connections."""
        for connection_id in list(self.connections.keys()):
            await self.cleanup_connection(connection_id)
        logger.info("Cleaned up all connections")
    
    def set_frame_processor(self, connection_id: str, processor: Callable[[Dict[str, Any]], None]):
        """Set a frame processor for a specific connection."""
        self.frame_processors[connection_id] = processor
        logger.info(f"Set frame processor for connection {connection_id}")
    
    def remove_frame_processor(self, connection_id: str):
        """Remove frame processor for a specific connection."""
        if connection_id in self.frame_processors:
            del self.frame_processors[connection_id]
            logger.info(f"Removed frame processor for connection {connection_id}")
    
    def get_data_channel(self, connection_id: str):
        """Get the data channel for a specific connection."""
        return self.data_channels.get(connection_id)


# Global instance for easy access
video_manager = VideoCommunicationManager()
