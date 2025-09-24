"""
Example usage of the Video Communication API.

This module demonstrates how other modules can use the video communication system.
"""

import asyncio
import logging
from typing import Dict, List
from video_api import (
    init_video_system, 
    connect_video, 
    register_intensity_handler,
    unregister_intensity_handler,
    disconnect_video,
    get_connection_info,
    get_all_connections,
    shutdown_video_system
)
from video_communication import IntensityData, VideoConfig

logger = logging.getLogger(__name__)


class IntensityAnalyzer:
    """Example class that analyzes video intensity data."""
    
    def __init__(self):
        self.intensity_history: Dict[str, List[float]] = {}
        self.alerts: List[str] = []
    
    def handle_intensity_data(self, data: IntensityData):
        """Handle incoming intensity data."""
        connection_id = data.connection_id
        
        # Store intensity history
        if connection_id not in self.intensity_history:
            self.intensity_history[connection_id] = []
        
        self.intensity_history[connection_id].append(data.average_intensity)
        
        # Keep only last 100 values
        if len(self.intensity_history[connection_id]) > 100:
            self.intensity_history[connection_id].pop(0)
        
        # Analyze intensity
        self._analyze_intensity(data)
        
        # Log the data
        logger.info(f"Connection {connection_id}: "
                   f"Current={data.current_intensity:.1f}, "
                   f"Average={data.average_intensity:.1f}, "
                   f"Frames={data.frame_count}")
    
    def _analyze_intensity(self, data: IntensityData):
        """Analyze intensity data for patterns or alerts."""
        connection_id = data.connection_id
        avg_intensity = data.average_intensity
        
        # Check for very low intensity (possible camera issue)
        if avg_intensity < 10:
            alert = f"ALERT: Very low intensity ({avg_intensity:.1f}) for {connection_id}"
            self.alerts.append(alert)
            logger.warning(alert)
        
        # Check for very high intensity (possible overexposure)
        elif avg_intensity > 240:
            alert = f"ALERT: Very high intensity ({avg_intensity:.1f}) for {connection_id}"
            self.alerts.append(alert)
            logger.warning(alert)
        
        # Check for sudden changes
        if len(self.intensity_history[connection_id]) >= 10:
            recent_avg = sum(self.intensity_history[connection_id][-10:]) / 10
            older_avg = sum(self.intensity_history[connection_id][-20:-10]) / 10
            
            if abs(recent_avg - older_avg) > 50:
                alert = f"ALERT: Sudden intensity change for {connection_id}: {older_avg:.1f} -> {recent_avg:.1f}"
                self.alerts.append(alert)
                logger.warning(alert)
    
    def get_intensity_summary(self, connection_id: str) -> Dict[str, float]:
        """Get intensity summary for a connection."""
        if connection_id not in self.intensity_history:
            return {}
        
        history = self.intensity_history[connection_id]
        if not history:
            return {}
        
        return {
            "current": history[-1],
            "average": sum(history) / len(history),
            "min": min(history),
            "max": max(history),
            "samples": len(history)
        }
    
    def get_all_alerts(self) -> List[str]:
        """Get all alerts."""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()


class VideoManager:
    """Example manager class that uses the video API."""
    
    def __init__(self):
        self.analyzer = IntensityAnalyzer()
        self.connections: Dict[str, str] = {}  # connection_id -> user_id mapping
    
    async def initialize(self):
        """Initialize the video system."""
        # Configure video processing
        config = VideoConfig(
            max_samples=30,
            frame_timeout=5.0,
            log_interval=1.0
        )
        
        # Initialize the video system
        init_video_system(config)
        logger.info("Video system initialized")
    
    async def handle_connection_request(self, user_id: str, sdp: str, sdp_type: str) -> Dict[str, str]:
        """Handle a new connection request."""
        connection_id = f"user_{user_id}_{int(asyncio.get_event_loop().time() * 1000)}"
        
        # Register intensity handler for this connection
        register_intensity_handler(connection_id, self.analyzer.handle_intensity_data)
        
        # Create the connection
        answer = await connect_video(connection_id, sdp, sdp_type)
        
        # Store the mapping
        self.connections[connection_id] = user_id
        
        logger.info(f"Created connection {connection_id} for user {user_id}")
        return answer
    
    async def disconnect_user(self, user_id: str):
        """Disconnect a user."""
        # Find connection for user
        connection_id = None
        for conn_id, uid in self.connections.items():
            if uid == user_id:
                connection_id = conn_id
                break
        
        if connection_id:
            # Unregister handler
            unregister_intensity_handler(connection_id)
            
            # Disconnect
            await disconnect_video(connection_id)
            
            # Remove from mapping
            del self.connections[connection_id]
            
            logger.info(f"Disconnected user {user_id} (connection {connection_id})")
    
    def get_user_intensity_summary(self, user_id: str) -> Dict[str, float]:
        """Get intensity summary for a user."""
        # Find connection for user
        connection_id = None
        for conn_id, uid in self.connections.items():
            if uid == user_id:
                connection_id = conn_id
                break
        
        if connection_id:
            return self.analyzer.get_intensity_summary(connection_id)
        return {}
    
    def get_all_alerts(self) -> List[str]:
        """Get all intensity alerts."""
        return self.analyzer.get_all_alerts()
    
    def get_connection_stats(self) -> Dict[str, Dict]:
        """Get statistics for all connections."""
        return get_all_connections()
    
    async def shutdown(self):
        """Shutdown the video system."""
        await shutdown_video_system()
        logger.info("Video system shutdown complete")


# Example usage functions
async def example_basic_usage():
    """Example of basic video API usage."""
    logger.info("=== Basic Video API Usage ===")
    
    # Initialize the system
    config = VideoConfig(max_samples=20, frame_timeout=3.0)
    init_video_system(config)
    
    # Create a connection
    connection_id = "example_conn"
    
    # Register a simple intensity handler
    def simple_handler(data: IntensityData):
        logger.info(f"Intensity: {data.average_intensity:.1f}")
    
    register_intensity_handler(connection_id, simple_handler)
    
    # Simulate connection (in real usage, this would be called by the server)
    # answer = await connect_video(connection_id, sdp, sdp_type)
    
    logger.info("Basic usage example complete")


async def example_advanced_usage():
    """Example of advanced video API usage with manager."""
    logger.info("=== Advanced Video API Usage ===")
    
    # Create manager
    manager = VideoManager()
    await manager.initialize()
    
    # Simulate handling multiple users
    users = ["user1", "user2", "user3"]
    
    for user_id in users:
        # In real usage, this would be called with actual SDP data
        logger.info(f"Would handle connection for {user_id}")
        # answer = await manager.handle_connection_request(user_id, sdp, sdp_type)
    
    # Get statistics
    stats = manager.get_connection_stats()
    logger.info(f"Connection stats: {stats}")
    
    # Get alerts
    alerts = manager.get_all_alerts()
    logger.info(f"Alerts: {alerts}")
    
    # Shutdown
    await manager.shutdown()
    logger.info("Advanced usage example complete")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    asyncio.run(example_basic_usage())
    asyncio.run(example_advanced_usage())
