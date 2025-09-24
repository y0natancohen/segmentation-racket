"""
Simple Video API - Easy-to-use functions for video communication.

This module provides simple functions that other modules can import and use.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from aiortc import RTCSessionDescription
from video_communication import VideoCommunicationManager, IntensityData, VideoConfig

logger = logging.getLogger(__name__)

# Global video manager instance
_video_manager: Optional[VideoCommunicationManager] = None


def init_video_system(config: VideoConfig = None) -> VideoCommunicationManager:
    """Initialize the video communication system."""
    global _video_manager
    _video_manager = VideoCommunicationManager(config)
    logger.info("Video communication system initialized")
    return _video_manager


def get_video_manager() -> VideoCommunicationManager:
    """Get the global video manager instance."""
    if _video_manager is None:
        raise RuntimeError("Video system not initialized. Call init_video_system() first.")
    return _video_manager


async def create_connection(connection_id: str):
    """Create a new video connection."""
    manager = get_video_manager()
    return await manager.init_connection(connection_id)


async def connect_video(connection_id: str, sdp: str, sdp_type: str) -> Dict[str, str]:
    """Handle WebRTC offer and create answer."""
    manager = get_video_manager()
    
    # Create connection
    pc = await create_connection(connection_id)
    
    # Set remote description
    offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
    await pc.setRemoteDescription(offer)
    
    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }


def register_intensity_handler(connection_id: str, handler: Callable[[IntensityData], None]):
    """Register a handler for intensity data from a specific connection."""
    manager = get_video_manager()
    manager.register_intensity_callback(connection_id, handler)


def unregister_intensity_handler(connection_id: str):
    """Unregister intensity handler for a connection."""
    manager = get_video_manager()
    manager.unregister_intensity_callback(connection_id)


async def disconnect_video(connection_id: str):
    """Disconnect a video connection."""
    manager = get_video_manager()
    await manager.cleanup_connection(connection_id)


def get_connection_info(connection_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific connection."""
    manager = get_video_manager()
    return manager.get_connection_stats(connection_id)


def get_all_connections() -> Dict[str, Dict[str, Any]]:
    """Get information about all active connections."""
    manager = get_video_manager()
    return manager.get_all_connections()


async def shutdown_video_system():
    """Shutdown the entire video system."""
    manager = get_video_manager()
    await manager.cleanup_all_connections()
    logger.info("Video system shutdown complete")


# Convenience functions for common use cases
async def start_video_processing(connection_id: str, 
                                intensity_callback: Callable[[IntensityData], None] = None):
    """Start video processing for a connection with optional callback."""
    if intensity_callback:
        register_intensity_handler(connection_id, intensity_callback)
    logger.info(f"Started video processing for {connection_id}")


def is_connection_active(connection_id: str) -> bool:
    """Check if a connection is active."""
    manager = get_video_manager()
    return connection_id in manager.connections


def get_connection_count() -> int:
    """Get the number of active connections."""
    manager = get_video_manager()
    return len(manager.connections)


# Example usage functions
def example_intensity_handler(data: IntensityData):
    """Example intensity handler that logs the data."""
    logger.info(f"Intensity for {data.connection_id}: "
                f"Current={data.current_intensity:.1f}, "
                f"Average={data.average_intensity:.1f}")


async def example_usage():
    """Example of how to use the video API."""
    # Initialize the system
    init_video_system()
    
    # Create a connection
    connection_id = "example_conn"
    pc = await create_connection(connection_id)
    
    # Register intensity handler
    register_intensity_handler(connection_id, example_intensity_handler)
    
    # Start processing
    await start_video_processing(connection_id, example_intensity_handler)
    
    # Get connection info
    info = get_connection_info(connection_id)
    print(f"Connection info: {info}")
    
    # Cleanup
    await disconnect_video(connection_id)
