#!/usr/bin/env python3
"""
Integration tests for the dual-process architecture
Tests the communication between Python rectangle generator and TypeScript game.
"""

import unittest
import asyncio
import websockets
import json
import time
import threading
import subprocess
import sys
import os
from unittest.mock import patch

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polygon_generator import PolygonGenerator


class TestWebSocketIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for WebSocket communication."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a random port for each test to avoid conflicts
        import random
        port = random.randint(9000, 9999)
        self.generator = PolygonGenerator(host="localhost", port=port, fps=60)
        self.received_messages = []
        self.client_connected = False
    
    async def test_client_connection(self):
        """Test that a client can connect to the WebSocket server."""
        # Start the server
        server_task = asyncio.create_task(self.generator.start_server())
        
        # Give the server time to start
        await asyncio.sleep(0.1)
        
        # Connect a client
        try:
            async with websockets.connect(f"ws://localhost:{self.generator.port}") as websocket:
                self.client_connected = True
                
                # Wait for a message
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(message)
                
                # Verify message format
                self.assertIn('position', data)
                self.assertIn('vertices', data)
                self.assertIn('rotation', data)
                
        except asyncio.TimeoutError:
            self.fail("No message received within timeout")
        finally:
            # Clean up
            self.generator.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
    
    async def test_message_frequency(self):
        """Test that messages are sent at the correct frequency."""
        # Start the server
        server_task = asyncio.create_task(self.generator.start_server())
        await asyncio.sleep(0.1)
        
        try:
            async with websockets.connect(f"ws://localhost:{self.generator.port}") as websocket:
                start_time = time.time()
                message_count = 0
                
                # Collect messages for 1 second
                while time.time() - start_time < 1.0:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        message_count += 1
                        self.received_messages.append(json.loads(message))
                    except asyncio.TimeoutError:
                        break
                
                # Should receive approximately 60 messages per second (60 FPS)
                # Allow some tolerance for timing variations
                self.assertGreaterEqual(message_count, 50)
                self.assertLessEqual(message_count, 70)
                
        finally:
            self.generator.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
    
    async def test_message_consistency(self):
        """Test that messages are consistent and properly formatted."""
        # Start the server
        server_task = asyncio.create_task(self.generator.start_server())
        await asyncio.sleep(0.1)
        
        try:
            async with websockets.connect(f"ws://localhost:{self.generator.port}") as websocket:
                # Collect several messages
                for _ in range(10):
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(message)
                    
                    # Verify all required fields are present
                    required_fields = ['position', 'vertices', 'rotation']
                    for field in required_fields:
                        self.assertIn(field, data)
                    
                    # Verify position structure
                    self.assertIn('x', data['position'])
                    self.assertIn('y', data['position'])
                    
                    # Verify vertices structure
                    self.assertIsInstance(data['vertices'], list)
                    for vertex in data['vertices']:
                        self.assertIn('x', vertex)
                        self.assertIn('y', vertex)
                    
                    # Verify data types
                    self.assertIsInstance(data['rotation'], (int, float))
                    self.assertIsInstance(data['position']['x'], (int, float))
                    self.assertIsInstance(data['position']['y'], (int, float))
                    
        finally:
            self.generator.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
    
    async def test_multiple_clients(self):
        """Test that multiple clients can connect simultaneously."""
        # Start the server
        server_task = asyncio.create_task(self.generator.start_server())
        await asyncio.sleep(0.1)
        
        try:
            # Connect multiple clients
            clients = []
            for i in range(3):
                websocket = await websockets.connect(f"ws://localhost:{self.generator.port}")
                clients.append(websocket)
            
            # Verify all clients are connected
            self.assertEqual(len(self.generator.clients), 3)
            
            # Each client should receive messages
            for i, websocket in enumerate(clients):
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(message)
                self.assertIn('position', data)
                self.assertIn('vertices', data)
                self.assertIn('rotation', data)
            
            # Close all clients
            for websocket in clients:
                await websocket.close()
            
            # Wait a moment for cleanup
            await asyncio.sleep(0.1)
            
            # All clients should be disconnected
            self.assertEqual(len(self.generator.clients), 0)
            
        finally:
            self.generator.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


class TestRectangleMovementIntegration(unittest.TestCase):
    """Integration tests for rectangle movement calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = PolygonGenerator(fps=60)
    
    def test_movement_consistency_over_time(self):
        """Test that movement is consistent over time."""
        start_time = time.time()
        self.generator.start_time = start_time
        
        # Generate messages over 2 seconds (one full cycle)
        messages = []
        for i in range(120):  # 2 seconds at 60 FPS
            test_time = start_time + i * 0.016
            message = self.generator.calculate_polygon_data(test_time)
            messages.append(message)
        
        # Verify rotation is increasing
        for i in range(1, len(messages)):
            self.assertGreater(messages[i]['rotation'], messages[i-1]['rotation'])
        
        # Verify position changes over time
        y_positions = [msg['position']['y'] for msg in messages]
        self.assertNotEqual(y_positions[0], y_positions[-1])  # Should move over time
        
        # Verify vertices are consistent
        vertex_counts = [len(msg['vertices']) for msg in messages]
        self.assertTrue(all(count == vertex_counts[0] for count in vertex_counts))
    
    def test_movement_range_validation(self):
        """Test that the rectangle stays within expected bounds."""
        start_time = time.time()
        self.generator.start_time = start_time
        
        min_y = float('inf')
        max_y = float('-inf')
        
        # Test over multiple cycles
        for i in range(240):  # 4 seconds at 60 FPS
            test_time = start_time + i * 0.016
            message = self.generator.calculate_polygon_data(test_time)
            y = message['position']['y']
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        
        # The rectangle should move within the expected range
        expected_min = self.generator.center_y - self.generator.amplitude
        expected_max = self.generator.center_y + self.generator.amplitude
        
        self.assertAlmostEqual(min_y, expected_min, places=1)
        self.assertAlmostEqual(max_y, expected_max, places=1)
    
    def test_frequency_accuracy(self):
        """Test that the movement frequency is accurate."""
        start_time = time.time()
        self.generator.start_time = start_time
        
        # Find peaks in the movement
        peak_times = []
        last_y = 0
        increasing = True
        
        for i in range(240):  # 4 seconds at 60 FPS
            test_time = start_time + i * 0.016
            message = self.generator.calculate_polygon_data(test_time)
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


class TestPerformanceIntegration(unittest.TestCase):
    """Performance integration tests."""
    
    def test_message_generation_performance(self):
        """Test that message generation is fast enough for real-time use."""
        generator = PolygonGenerator(fps=60)
        start_time = time.time()
        generator.start_time = start_time
        
        # Generate 1000 messages
        start_perf = time.time()
        for i in range(1000):
            test_time = start_time + i * 0.016
            message = generator.calculate_polygon_data(test_time)
        end_perf = time.time()
        
        duration = end_perf - start_perf
        
        # Should generate 1000 messages in less than 100ms
        self.assertLess(duration, 0.1)
        
        # Should be able to generate at least 60 messages per second
        messages_per_second = 1000 / duration
        self.assertGreater(messages_per_second, 60)
    
    def test_json_serialization_performance(self):
        """Test that JSON serialization is fast enough."""
        generator = PolygonGenerator(fps=60)
        start_time = time.time()
        generator.start_time = start_time
        
        # Generate and serialize 1000 messages
        start_perf = time.time()
        for i in range(1000):
            test_time = start_time + i * 0.016
            message = generator.calculate_polygon_data(test_time)
            json_string = json.dumps(message)
        end_perf = time.time()
        
        duration = end_perf - start_perf
        
        # Should serialize 1000 messages in less than 200ms
        self.assertLess(duration, 0.2)
        
        # Should be able to serialize at least 60 messages per second
        messages_per_second = 1000 / duration
        self.assertGreater(messages_per_second, 60)


class TestErrorHandlingIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for error handling."""
    
    async def test_server_startup_with_port_in_use(self):
        """Test that the server handles port conflicts gracefully."""
        # Start first server
        generator1 = PolygonGenerator(host="localhost", port=8769, fps=60)
        server1_task = asyncio.create_task(generator1.start_server())
        await asyncio.sleep(0.1)
        
        try:
            # Try to start second server on same port
            generator2 = PolygonGenerator(host="localhost", port=8769, fps=60)
            
            with self.assertRaises(OSError):
                await generator2.start_server()
                
        finally:
            # Clean up
            generator1.running = False
            server1_task.cancel()
            try:
                await server1_task
            except asyncio.CancelledError:
                pass
    
    async def test_client_disconnection_handling(self):
        """Test that the server handles client disconnections gracefully."""
        generator = PolygonGenerator(host="localhost", port=8770, fps=60)
        server_task = asyncio.create_task(generator.start_server())
        await asyncio.sleep(0.1)
        
        try:
            # Connect and immediately disconnect
            websocket = await websockets.connect("ws://localhost:8770")
            await websocket.close()
            
            # Wait for cleanup
            await asyncio.sleep(0.1)
            
            # Server should still be running
            self.assertTrue(generator.running)
            self.assertEqual(len(generator.clients), 0)
            
        finally:
            generator.running = False
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
