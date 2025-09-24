"""
Metrics collection and monitoring utilities.
"""
import time
import logging
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and monitors processing metrics."""
    
    def __init__(self):
        self.frames_processed = 0
        self.messages_sent = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Rolling windows for rate calculation
        self.frame_times = deque(maxlen=100)
        self.message_times = deque(maxlen=100)
    
    def record_frame_processed(self):
        """Record that a frame was processed."""
        self.frames_processed += 1
        self.frame_times.append(time.time())
    
    def record_message_sent(self):
        """Record that a message was sent."""
        self.messages_sent += 1
        self.message_times.append(time.time())
    
    def get_frames_per_second(self) -> float:
        """Get current frames per second (rolling 1s window)."""
        now = time.time()
        one_second_ago = now - 1.0
        
        # Count frames in last second
        recent_frames = sum(1 for t in self.frame_times if t > one_second_ago)
        return float(recent_frames)
    
    def get_messages_per_second(self) -> float:
        """Get current messages per second (rolling 1s window)."""
        now = time.time()
        one_second_ago = now - 1.0
        
        # Count messages in last second
        recent_messages = sum(1 for t in self.message_times if t > one_second_ago)
        return float(recent_messages)
    
    def get_total_rates(self):
        """Get total processing rates since start."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0, 0.0
        
        frames_per_sec = self.frames_processed / elapsed
        messages_per_sec = self.messages_sent / elapsed
        
        return frames_per_sec, messages_per_sec
    
    def should_log(self) -> bool:
        """Check if it's time to log metrics (every 1 second)."""
        now = time.time()
        if now - self.last_log_time >= 1.0:
            self.last_log_time = now
            return True
        return False
    
    def log_metrics(self):
        """Log current metrics."""
        fps_current = self.get_frames_per_second()
        mps_current = self.get_messages_per_second()
        fps_total, mps_total = self.get_total_rates()
        
        logger.info(
            f"Metrics - Current: {fps_current:.1f} fps, {mps_current:.1f} msg/s | "
            f"Total: {fps_total:.1f} fps, {mps_total:.1f} msg/s | "
            f"Frames: {self.frames_processed}, Messages: {self.messages_sent}"
        )


class ConnectionMetrics:
    """Per-connection metrics tracking."""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.start_time = time.time()
        self.frames_received = 0
        self.messages_sent = 0
        self.last_activity = self.start_time
    
    def record_frame(self):
        """Record frame received."""
        self.frames_received += 1
        self.last_activity = time.time()
    
    def record_message_sent(self):
        """Record message sent."""
        self.messages_sent += 1
    
    def get_uptime(self) -> float:
        """Get connection uptime in seconds."""
        return time.time() - self.start_time
    
    def get_activity_age(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self.last_activity
    
    def is_stale(self, timeout: float = 30.0) -> bool:
        """Check if connection is stale (no activity for timeout seconds)."""
        return self.get_activity_age() > timeout
