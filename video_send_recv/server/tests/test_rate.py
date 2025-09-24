"""
Tests for rate monitoring and processing performance.
"""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from media import create_test_frame, mean_intensity
from metrics import MetricsCollector, ConnectionMetrics


class TestRateMonitoring:
    """Test rate monitoring and performance."""
    
    def test_metrics_collector_basic(self):
        """Test basic metrics collection."""
        collector = MetricsCollector()
        
        # Initially should be zero
        assert collector.frames_processed == 0
        assert collector.messages_sent == 0
        
        # Record some events
        collector.record_frame_processed()
        collector.record_frame_processed()
        collector.record_message_sent()
        
        assert collector.frames_processed == 2
        assert collector.messages_sent == 1
    
    def test_connection_metrics(self):
        """Test per-connection metrics."""
        conn_metrics = ConnectionMetrics("test_conn")
        
        assert conn_metrics.connection_id == "test_conn"
        assert conn_metrics.frames_received == 0
        assert conn_metrics.messages_sent == 0
        
        # Record activity
        conn_metrics.record_frame()
        conn_metrics.record_message_sent()
        conn_metrics.record_frame()
        
        assert conn_metrics.frames_received == 2
        assert conn_metrics.messages_sent == 1
        
        # Check uptime
        uptime = conn_metrics.get_uptime()
        assert uptime >= 0
        assert uptime < 1.0  # Should be very recent
    
    def test_stale_connection_detection(self):
        """Test stale connection detection."""
        conn_metrics = ConnectionMetrics("test_conn")
        
        # Should not be stale immediately
        assert not conn_metrics.is_stale(timeout=1.0)
        
        # Wait a bit and check again
        time.sleep(0.1)
        assert not conn_metrics.is_stale(timeout=1.0)
        
        # Should be stale after timeout
        assert conn_metrics.is_stale(timeout=0.05)  # Very short timeout
    
    @pytest.mark.asyncio
    async def test_sustained_processing_rate(self):
        """Test sustained processing at 30 FPS for 5 seconds."""
        collector = MetricsCollector()
        start_time = time.time()
        target_duration = 5.0  # 5 seconds
        target_fps = 30.0
        
        # Simulate processing frames at 30 FPS
        frame_interval = 1.0 / target_fps
        frames_processed = 0
        
        while time.time() - start_time < target_duration:
            # Create and process a test frame
            frame = create_test_frame(intensity=128)
            mean_intensity(frame)  # Process frame
            collector.record_frame_processed()
            frames_processed += 1
            
            # Wait for next frame
            await asyncio.sleep(frame_interval)
        
        elapsed_time = time.time() - start_time
        actual_fps = frames_processed / elapsed_time
        
        # Should achieve at least 28 FPS (allowing for some variance)
        assert actual_fps >= 28.0, f"Expected >=28 FPS, got {actual_fps:.1f} FPS"
        
        # Should not exceed 32 FPS (allowing for some variance)
        assert actual_fps <= 32.0, f"Expected <=32 FPS, got {actual_fps:.1f} FPS"
        
        # Check total rates
        total_fps, total_mps = collector.get_total_rates()
        assert total_fps >= 28.0, f"Total FPS should be >=28, got {total_fps:.1f}"
    
    @pytest.mark.asyncio
    async def test_message_emission_rate(self):
        """Test that messages are emitted at expected rate."""
        collector = MetricsCollector()
        
        # Simulate 5 seconds of processing at 30 FPS
        target_duration = 5.0
        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        
        start_time = time.time()
        messages_sent = 0
        
        while time.time() - start_time < target_duration:
            # Process frame and send message
            frame = create_test_frame(intensity=128)
            mean_intensity(frame)
            collector.record_frame_processed()
            collector.record_message_sent()
            messages_sent += 1
            
            await asyncio.sleep(frame_interval)
        
        elapsed_time = time.time() - start_time
        actual_rate = messages_sent / elapsed_time
        
        # Should achieve at least 28 messages/sec
        assert actual_rate >= 28.0, f"Expected >=28 msg/s, got {actual_rate:.1f} msg/s"
        
        # Check rolling window rate
        rolling_rate = collector.get_messages_per_second()
        assert rolling_rate >= 28.0, f"Rolling rate should be >=28, got {rolling_rate:.1f}"
    
    def test_rolling_window_calculation(self):
        """Test rolling window rate calculations."""
        collector = MetricsCollector()
        
        # Add some frame times
        base_time = time.time()
        for i in range(10):
            collector.frame_times.append(base_time + i * 0.1)  # 10 FPS
        
        # Add some message times
        for i in range(10):
            collector.message_times.append(base_time + i * 0.1)
        
        # Get rates
        fps = collector.get_frames_per_second()
        mps = collector.get_messages_per_second()
        
        # Should be around 10 FPS (allowing for timing variance)
        assert 8.0 <= fps <= 12.0, f"Expected ~10 FPS, got {fps:.1f}"
        assert 8.0 <= mps <= 12.0, f"Expected ~10 msg/s, got {mps:.1f}"
    
    def test_logging_frequency(self):
        """Test that logging happens at appropriate frequency."""
        collector = MetricsCollector()
        
        # Should not log immediately
        assert not collector.should_log()
        
        # Wait a bit and check again
        time.sleep(0.1)
        assert not collector.should_log()
        
        # Should log after 1 second
        time.sleep(1.0)
        assert collector.should_log()
        
        # Should not log again immediately after
        assert not collector.should_log()
