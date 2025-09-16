#!/usr/bin/env python3
"""
Unit tests for the Polygon Generator Process
Tests the core functionality of the polygon movement generation and WebSocket communication.
"""

import unittest
import asyncio
import json
import time
import math
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the current directory to the path so we can import rectangle_generator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polygon_generator import PolygonGenerator


class TestPolygonGenerator(unittest.TestCase):
    """Test cases for the PolygonGenerator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = PolygonGenerator(host="localhost", port=8766, fps=60)  # Use different port for testing
    
    def test_initialization(self):
        """Test that the generator initializes with correct parameters."""
        self.assertEqual(self.generator.host, "localhost")
        self.assertEqual(self.generator.port, 8766)
        self.assertEqual(self.generator.fps, 60)
        self.assertEqual(self.generator.frame_duration, 1.0 / 60)
        self.assertFalse(self.generator.running)
        self.assertEqual(len(self.generator.clients), 0)
        self.assertEqual(self.generator.size, 600)
        self.assertEqual(self.generator.amplitude, 90)  # 15% of 600
        self.assertEqual(self.generator.center_y, 330)  # 55% of 600
        self.assertEqual(self.generator.frequency, 0.5)
    
    def test_calculate_polygon_data(self):
        """Test polygon data calculation with sine wave movement."""
        # Test at start time (should be at center_y)
        start_time = time.time()
        self.generator.start_time = start_time
        
        polygon_data = self.generator.calculate_polygon_data(start_time)
        
        self.assertIn('position', polygon_data)
        self.assertIn('vertices', polygon_data)
        self.assertIn('rotation', polygon_data)
        
        # At start time, rotation should be 0
        self.assertEqual(polygon_data['rotation'], 0)
        
        # Position should be at center_y (330) at start
        self.assertEqual(polygon_data['position']['x'], 300)  # size/2
        self.assertEqual(polygon_data['position']['y'], 330)  # center_y
        
        # Should have vertices from the config
        self.assertIsInstance(polygon_data['vertices'], list)
        self.assertGreater(len(polygon_data['vertices']), 0)
    
    def test_sine_wave_movement(self):
        """Test that the movement follows a proper sine wave pattern."""
        start_time = time.time()
        self.generator.start_time = start_time
        
        # Test at different phases of the sine wave
        test_times = [
            (0, 0),  # Start: phase 0, should be at center_y
            (1, math.pi),  # 1 second: phase π, should be at center_y
            (0.5, math.pi/2),  # 0.5 seconds: phase π/2, should be at center_y + amplitude
            (1.5, 3*math.pi/2),  # 1.5 seconds: phase 3π/2, should be at center_y - amplitude
        ]
        
        for elapsed, expected_phase in test_times:
            test_time = start_time + elapsed
            polygon_data = self.generator.calculate_polygon_data(test_time)
            
            # Check position calculation
            expected_y = self.generator.center_y + self.generator.amplitude * math.sin(expected_phase)
            self.assertAlmostEqual(polygon_data['position']['y'], expected_y, places=5)
            
            # Check rotation is increasing
            expected_rotation = elapsed * 0.5  # 0.5 radians per second
            self.assertAlmostEqual(polygon_data['rotation'], expected_rotation, places=5)
    
    def test_polygon_vertices(self):
        """Test that polygon vertices are correctly transformed."""
        start_time = time.time()
        self.generator.start_time = start_time
        
        # Test vertices at different rotation angles
        test_times = [0, 1, 2, 3]
        
        for elapsed in test_times:
            test_time = start_time + elapsed
            polygon_data = self.generator.calculate_polygon_data(test_time)
            
            # Should have vertices from config
            self.assertIsInstance(polygon_data['vertices'], list)
            self.assertGreater(len(polygon_data['vertices']), 0)
            
            # Each vertex should have x and y coordinates
            for vertex in polygon_data['vertices']:
                self.assertIn('x', vertex)
                self.assertIn('y', vertex)
                self.assertIsInstance(vertex['x'], (int, float))
                self.assertIsInstance(vertex['y'], (int, float))
    
    def test_message_format(self):
        """Test that the message format is correct and serializable."""
        start_time = time.time()
        self.generator.start_time = start_time
        
        polygon_data = self.generator.calculate_polygon_data(start_time)
        
        # Test that the data can be serialized to JSON
        json_string = json.dumps(polygon_data)
        self.assertIsInstance(json_string, str)
        
        # Test that it can be deserialized back
        deserialized_data = json.loads(json_string)
        self.assertEqual(deserialized_data, polygon_data)
        
        # Test required fields
        required_fields = ['position', 'vertices', 'rotation']
        for field in required_fields:
            self.assertIn(field, polygon_data)
        
        # Test position structure
        self.assertIn('x', polygon_data['position'])
        self.assertIn('y', polygon_data['position'])
        
        # Test vertices structure
        self.assertIsInstance(polygon_data['vertices'], list)
        for vertex in polygon_data['vertices']:
            self.assertIn('x', vertex)
            self.assertIn('y', vertex)
    
    def test_fps_calculation(self):
        """Test that FPS is correctly calculated."""
        # Test different FPS values
        test_fps_values = [20, 30, 60, 120]
        
        for fps in test_fps_values:
            gen = PolygonGenerator(fps=fps)
            expected_frame_duration = 1.0 / fps
            self.assertEqual(gen.frame_duration, expected_frame_duration)
    
    def test_polygon_config_loading(self):
        """Test that polygon configuration is loaded correctly."""
        # Test with default config
        gen = PolygonGenerator()
        self.assertIn('name', gen.config)
        self.assertIn('vertices', gen.config)
        self.assertIn('center_offset', gen.config)
        self.assertIn('scale', gen.config)
        
        # Should have vertices
        self.assertIsInstance(gen.config['vertices'], list)
        self.assertGreater(len(gen.config['vertices']), 0)
    
    def test_movement_parameters(self):
        """Test that movement parameters are correctly set."""
        # Test with different size values
        test_size = 800
        gen = PolygonGenerator()
        
        # Update size and recalculate parameters
        gen.size = test_size
        gen.amplitude = gen.size * 0.15
        gen.center_y = gen.size * 0.55
        
        # Amplitude should be 15% of size
        expected_amplitude = test_size * 0.15
        self.assertEqual(gen.amplitude, expected_amplitude)
        
        # Center Y should be 55% of size
        expected_center_y = test_size * 0.55
        self.assertEqual(gen.center_y, expected_center_y)


class TestPolygonGeneratorAsync(unittest.IsolatedAsyncioTestCase):
    """Async test cases for the PolygonGenerator class."""
    
    async def asyncSetUp(self):
        """Set up async test fixtures."""
        self.generator = PolygonGenerator(host="localhost", port=8767, fps=60)  # Use different port
    
    async def test_websocket_server_startup(self):
        """Test that the WebSocket server can start up."""
        # Mock the websockets.serve function
        with patch('polygon_generator.websockets.serve') as mock_serve:
            # Create a proper async mock that can be awaited
            async def mock_serve_func(*args, **kwargs):
                return AsyncMock()
            
            mock_serve.side_effect = mock_serve_func
            
            # Start the server
            self.generator.running = True
            server_task = asyncio.create_task(self.generator.start_server())
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Verify that websockets.serve was called
            mock_serve.assert_called_once()
            
            # Clean up
            self.generator.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
    
    async def test_client_registration(self):
        """Test client registration and removal."""
        # Create mock WebSocket that never closes
        mock_websocket = AsyncMock()
        mock_websocket.wait_closed = AsyncMock()
        # Make wait_closed never complete (simulate active connection)
        mock_websocket.wait_closed.side_effect = asyncio.Event().wait
        
        # Start client registration in background
        registration_task = asyncio.create_task(
            self.generator.register_client(mock_websocket)
        )
        
        # Give it a moment to register
        await asyncio.sleep(0.01)
        
        # Verify client was added
        self.assertIn(mock_websocket, self.generator.clients)
        self.assertEqual(len(self.generator.clients), 1)
        
        # Cancel the registration task
        registration_task.cancel()
        try:
            await registration_task
        except asyncio.CancelledError:
            pass
        
        # Verify client was removed (due to finally block)
        self.assertNotIn(mock_websocket, self.generator.clients)
        self.assertEqual(len(self.generator.clients), 0)
    
    async def test_broadcast_rectangle_data(self):
        """Test that rectangle data is broadcast to clients."""
        # Create mock clients
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        self.generator.clients = {mock_client1, mock_client2}
        
        # Start broadcasting
        self.generator.running = True
        broadcast_task = asyncio.create_task(self.generator.broadcast_polygon_data())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop broadcasting
        self.generator.running = False
        broadcast_task.cancel()
        
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
        
        # Verify that both clients received messages
        self.assertTrue(mock_client1.send.called)
        self.assertTrue(mock_client2.send.called)
        
        # Verify that the sent data is valid JSON
        for call in mock_client1.send.call_args_list:
            message = call[0][0]  # First argument
            data = json.loads(message)
            self.assertIn('position', data)
            self.assertIn('vertices', data)
            self.assertIn('rotation', data)
    
    async def test_performance_monitoring(self):
        """Test that performance monitoring works correctly."""
        # Set up generator with performance monitoring
        self.generator.frame_count = 0
        self.generator.last_stats_time = time.time()
        
        # Create mock clients
        mock_client = AsyncMock()
        self.generator.clients = {mock_client}
        
        # Start broadcasting
        self.generator.running = True
        broadcast_task = asyncio.create_task(self.generator.broadcast_polygon_data())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop broadcasting
        self.generator.running = False
        broadcast_task.cancel()
        
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
        
        # Verify that frame count was incremented
        self.assertGreater(self.generator.frame_count, 0)


class TestPolygonGeneratorIntegration(unittest.TestCase):
    """Integration tests for the PolygonGenerator."""
    
    def test_message_consistency(self):
        """Test that messages are consistent over time."""
        generator = PolygonGenerator(fps=60)
        start_time = time.time()
        generator.start_time = start_time
        
        # Generate multiple messages
        messages = []
        for i in range(10):
            test_time = start_time + i * 0.016  # 60 FPS intervals
            message = generator.calculate_polygon_data(test_time)
            messages.append(message)
        
        # Verify that rotation is increasing
        for i in range(1, len(messages)):
            self.assertGreater(messages[i]['rotation'], messages[i-1]['rotation'])
        
        # Verify that position changes over time
        y_positions = [msg['position']['y'] for msg in messages]
        self.assertNotEqual(y_positions[0], y_positions[-1])  # Should move over time
        
        # Verify that vertices are consistent (same count)
        vertex_counts = [len(msg['vertices']) for msg in messages]
        self.assertTrue(all(count == vertex_counts[0] for count in vertex_counts))
    
    def test_movement_range(self):
        """Test that the rectangle stays within expected bounds."""
        generator = PolygonGenerator()
        start_time = time.time()
        generator.start_time = start_time
        
        # Test over a full cycle (2 seconds)
        min_y = float('inf')
        max_y = float('-inf')
        
        for i in range(120):  # 2 seconds at 60 FPS
            test_time = start_time + i * 0.016
            message = generator.calculate_polygon_data(test_time)
            y = message['position']['y']
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        
        # The rectangle should move within the expected range
        expected_min = generator.center_y - generator.amplitude
        expected_max = generator.center_y + generator.amplitude
        
        self.assertAlmostEqual(min_y, expected_min, places=1)
        self.assertAlmostEqual(max_y, expected_max, places=1)
    
    def test_frequency_accuracy(self):
        """Test that the movement frequency is accurate."""
        generator = PolygonGenerator()
        start_time = time.time()
        generator.start_time = start_time
        
        # Find the first peak (maximum Y position)
        peak_times = []
        last_y = 0
        increasing = True
        
        for i in range(240):  # 4 seconds at 60 FPS
            test_time = start_time + i * 0.016
            message = generator.calculate_polygon_data(test_time)
            y = message['position']['y']
            
            if increasing and y < last_y:
                peak_times.append(test_time - start_time)
                increasing = False
            elif not increasing and y > last_y:
                increasing = True
            
            last_y = y
        
        # Should have peaks every 2 seconds (0.5 Hz frequency)
        if len(peak_times) >= 2:
            period = peak_times[1] - peak_times[0]
            expected_period = 2.0  # 2 seconds for 0.5 Hz
            self.assertAlmostEqual(period, expected_period, places=1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
