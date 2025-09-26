"""
Video Segmentation Integration - Connects segmentation to video communication system.

This module integrates the segmentation system with the video communication API
to receive frames from WebRTC and send polygon data back to the frontend.
"""

import asyncio
from collections import deque
import json
import logging
import time
import threading
from typing import Optional, Callable, Dict, Any
import numpy as np
import cv2
import torch

# Import the video communication API
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'video_send_recv', 'server'))
from video_api import init_video_system, get_video_manager, connect_video
from video_communication import VideoConfig

# Import segmentation components
from segmentation.segmentation import (
    setup_model, 
    process_frame, 
    build_view, 
    matte_to_polygon,
    draw_polygon_on_image,
    TimingStats
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VideoSegmentationProcessor:
    """Processes video frames from WebRTC and generates segmentation polygons."""
    
    def __init__(self, segmentation_args, polygon_callback: Optional[Callable] = None):
        """
        Initialize the video segmentation processor.
        
        Args:
            segmentation_args: Arguments for segmentation model setup
            polygon_callback: Optional callback function to handle generated polygons
        """
        self.segmentation_args = segmentation_args
        self.polygon_callback = polygon_callback
        self.model = None
        self.device = None
        self.timing_stats = TimingStats()
        self.is_processing = False
        self.frame_buffer = deque(maxlen=1)
        # self.max_buffer_size = 5
        
        # Initialize video system
        self.video_manager = None
        self.connection_id = None
        
        # Initialize recurrent states for RVM model
        self.rec = [None] * 4
        
        # Processing rate limiting
        self.last_process_time = 0
        self.min_process_interval = 1.0 / 15.0  # Max 15 FPS processing
        
        # Job pool for controlling concurrent processing
        self.max_concurrent_jobs = 1
        self.active_jobs = set()  # Track active processing tasks
        self.job_counter = 0  # Unique job IDs
        self.dropped_frames = 0  # Track dropped frames
        
    async def initialize(self):
        """Initialize the segmentation model and video system."""
        try:
            # Setup segmentation model
            self.model, self.device = setup_model(self.segmentation_args)
            logger.info("Segmentation model initialized")
            
            # Initialize video system
            config = VideoConfig(
                max_samples=30,
                frame_timeout=5.0,
                log_interval=1.0
            )
            self.video_manager = init_video_system(config)
            
            # Set up event handlers (using the correct API)
            # Note: The video manager handles events internally
            
            logger.info("Video segmentation processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize video segmentation processor: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def start_processing(self, connection_id: str):
        """Start processing frames for a specific connection."""
        self.connection_id = connection_id
        self.is_processing = True
        
        # Get the video manager and start processing
        video_manager = get_video_manager()
        
        # Set up frame processing callback
        video_manager.set_frame_processor(connection_id, self._process_frame)
        
        logger.info(f"Started processing frames for connection: {connection_id}")
    
    async def stop_processing(self):
        """Stop processing frames."""
        self.is_processing = False
        if self.connection_id:
            video_manager = get_video_manager()
            video_manager.remove_frame_processor(self.connection_id)
        logger.info("Stopped processing frames")
    
    def set_max_concurrent_jobs(self, max_jobs: int):
        """Set the maximum number of concurrent processing jobs."""
        self.max_concurrent_jobs = max_jobs
        logger.info(f"Set max concurrent jobs to {max_jobs}")
    
    def get_job_pool_status(self) -> Dict[str, Any]:
        """Get current job pool status."""
        return {
            'active_jobs': len(self.active_jobs),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'dropped_frames': self.dropped_frames,
            'job_counter': self.job_counter
        }
    
    def _process_frame(self, frame_data: Dict[str, Any]):
        """Process a single frame and generate segmentation polygon."""
        logger.debug(f"Processing frame: is_processing={self.is_processing}, model={self.model is not None}")
        
        if not self.is_processing or not self.model:
            logger.debug("Skipping frame processing: not processing or no model")
            return
        
        # Add frame to buffer (maxlen=1, so only keeps latest frame)
        # This ensures we always process the most recent frame and drop older ones
        # for minimal latency
        self.frame_buffer.append(frame_data)
        
        # Process only the latest frame from buffer
        if not self.frame_buffer:
            logger.debug("No frames in buffer")
            return
            
        latest_frame_data = self.frame_buffer[-1]
        
        # Check if we can start a new processing job
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            self.dropped_frames += 1
            logger.debug(f"Frame dropped: {len(self.active_jobs)}/{self.max_concurrent_jobs} jobs already running (total dropped: {self.dropped_frames})")
            return
        
        # Process frame asynchronously with job pool control
        asyncio.create_task(self._process_frame_with_job_pool(latest_frame_data))
    
    async def _process_frame_with_job_pool(self, frame_data: Dict[str, Any]):
        """Wrapper method that manages job pool for concurrent processing control."""
        # Generate unique job ID
        self.job_counter += 1
        job_id = self.job_counter
        
        # Add job to active jobs set
        self.active_jobs.add(job_id)
        logger.debug(f"Started job {job_id}, active jobs: {len(self.active_jobs)}/{self.max_concurrent_jobs}")
        
        try:
            # Process the frame
            await self._process_frame_async(frame_data)
        finally:
            # Remove job from active jobs set
            self.active_jobs.discard(job_id)
            logger.debug(f"Completed job {job_id}, active jobs: {len(self.active_jobs)}/{self.max_concurrent_jobs}")
    
    async def _process_frame_async(self, frame_data: Dict[str, Any]):
        """Process frame asynchronously to avoid blocking WebRTC pipeline."""
        # Rate limiting - skip processing if too frequent
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            logger.debug("Skipping frame processing: rate limited")
            return
        
        self.last_process_time = current_time
        
        try:
            # Extract frame data from latest frame
            frame_array = frame_data.get('frame')
            if frame_array is None:
                logger.debug("No frame data in latest frame")
                return
            
            logger.debug(f"Frame data shape: {frame_array.shape}, dtype: {frame_array.dtype}")
            
            # Validate frame array
            if not isinstance(frame_array, np.ndarray):
                logger.error(f"Frame data is not a numpy array: {type(frame_array)}")
                return
            
            if len(frame_array.shape) != 3 or frame_array.shape[2] != 3:
                logger.error(f"Invalid frame shape: {frame_array.shape}, expected (H, W, 3)")
                return
            
            if frame_array.dtype != np.uint8:
                logger.error(f"Invalid frame dtype: {frame_array.dtype}, expected uint8")
                return
            
            # Ensure frame is contiguous
            if not frame_array.flags.c_contiguous:
                logger.debug("Frame array is not contiguous, making it contiguous")
                frame_array = np.ascontiguousarray(frame_array)
            
            # Convert to OpenCV format (BGR)
            try:
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                logger.debug(f"Converted to BGR: {frame_bgr.shape}")
                
            except Exception as e:
                logger.error(f"OpenCV color conversion failed: {e}")
                logger.error(f"Frame array info: shape={frame_array.shape}, dtype={frame_array.dtype}, contiguous={frame_array.flags.c_contiguous}")
                return
            
            # Process frame with segmentation model
            with torch.inference_mode():
                # Convert to torch format (same as segmentation.py)
                frame_torch = self._to_torch_image(frame_bgr, self.device, self.segmentation_args.fp16)
                logger.debug(f"Torch tensor shape: {frame_torch.shape}")
                
                # Run segmentation (same as segmentation.py)
                logger.debug("Running segmentation model...")
                fgr, pha, self.rec[0], self.rec[1], self.rec[2], self.rec[3] = self.model(frame_torch, self.rec[0], self.rec[1], self.rec[2], self.rec[3], self.segmentation_args.dsr)
                logger.debug(f"Segmentation output: pha={pha.shape}, fgr={fgr.shape}")
                
                # Convert alpha matte to numpy (pha should be single channel)
                pha_np = pha.squeeze().cpu().numpy()
                logger.debug(f"Alpha matte numpy: {pha_np.shape}, min={pha_np.min():.3f}, max={pha_np.max():.3f}")
                
                # Ensure single channel (pha should already be single channel)
                if len(pha_np.shape) != 2:
                    logger.error(f"Invalid alpha matte shape: {pha_np.shape}, expected 2D")
                    return
                
                # Convert to uint8 if needed (matte_to_polygon expects 0-255 range)
                if pha_np.dtype != np.uint8:
                    # Scale from 0-1 range to 0-255 range
                    pha_np = (pha_np * 255).astype(np.uint8)
                    logger.debug(f"Converted to uint8: {pha_np.shape}, min={pha_np.min()}, max={pha_np.max()}")
                
                logger.debug(f"Final alpha matte: shape={pha_np.shape}, dtype={pha_np.dtype}, min={pha_np.min()}, max={pha_np.max()}")
                
                # Generate polygon from alpha matte
                print(f'self.segmentation_args.polygon_epsilon: {self.segmentation_args.polygon_epsilon}')
                polygon = matte_to_polygon(
                    pha_np,
                    threshold=self.segmentation_args.polygon_threshold,
                    min_area=self.segmentation_args.polygon_min_area,
                    epsilon_ratio=0.002,
                    timing_stats=self.timing_stats
                )
                
                # Debug logging
                logger.debug(f"Alpha matte shape: {pha_np.shape}, min: {pha_np.min():.3f}, max: {pha_np.max():.3f}")
                logger.debug(f"Polygon generated: {polygon is not None}")
                if polygon is not None:
                    logger.info(f"Polygon points: {len(polygon)}")
                
                # Send polygon data if callback is provided
                if self.polygon_callback and polygon is not None:
                    polygon_data = {
                        'connection_id': frame_data.get('connection_id', self.connection_id),
                        'polygon': polygon.tolist(),
                        'timestamp': frame_data.get('timestamp', time.time()),
                        'frame_shape': frame_bgr.shape[:2],  # (height, width) - processed frame dimensions
                        'original_image_size': frame_bgr.shape[:2]  # (height, width) - original image dimensions
                    }
                    logger.info(f"✅ Sending polygon data: {len(polygon)} points")
                    logger.debug(f"Polygon data: {polygon_data}")
                    self.polygon_callback(polygon_data)
                else:
                    logger.debug(f"No polygon data to send: callback={self.polygon_callback is not None}, polygon={polygon is not None}")
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def _to_torch_image(self, frame_bgr, device, half):
        """Convert OpenCV frame to torch tensor."""
        try:
            # Validate input
            if not isinstance(frame_bgr, np.ndarray):
                raise ValueError(f"frame_bgr must be numpy array, got {type(frame_bgr)}")
            
            if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                raise ValueError(f"frame_bgr must be (H, W, 3), got {frame_bgr.shape}")
            
            if frame_bgr.dtype != np.uint8:
                raise ValueError(f"frame_bgr must be uint8, got {frame_bgr.dtype}")
            
            # Ensure contiguous
            if not frame_bgr.flags.c_contiguous:
                frame_bgr = np.ascontiguousarray(frame_bgr)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor
            ten = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
            
            # Apply half precision if requested and on CUDA
            if half and device.type == "cuda":
                ten = ten.half()
            
            return ten.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error in _to_torch_image: {e}")
            logger.error(f"frame_bgr info: shape={frame_bgr.shape if hasattr(frame_bgr, 'shape') else 'no shape'}, dtype={frame_bgr.dtype if hasattr(frame_bgr, 'dtype') else 'no dtype'}")
            raise
    
    def _handle_intensity_update(self, connection_id: str, metrics: Dict[str, Any]):
        """Handle intensity updates from video communication."""
        logger.debug(f"Intensity update for {connection_id}: {metrics}")
    
    def _handle_connection_state_change(self, connection_id: str, state: str):
        """Handle connection state changes."""
        logger.info(f"Connection {connection_id} state changed to: {state}")
        if state in ['closed', 'failed', 'disconnected']:
            self.stop_processing()
    
    def _handle_error(self, connection_id: str, error: Exception):
        """Handle errors from video communication."""
        logger.error(f"Error for connection {connection_id}: {error}")


class PolygonDataChannel:
    """Handles sending polygon data back to the frontend via WebRTC data channel."""
    
    def __init__(self, video_manager, websocket_server=None):
        self.video_manager = video_manager
        self.websocket_server = websocket_server
        self.connection_id = None
    
    def set_connection(self, connection_id: str):
        """Set the connection ID for sending polygon data."""
        self.connection_id = connection_id
    
    def send_polygon(self, polygon_data: Dict[str, Any]):
        """Send polygon data to the frontend."""
        if not self.connection_id:
            return
        
        try:
            # Send via WebSocket if available
            if self.websocket_server:
                asyncio.create_task(self.websocket_server.broadcast_polygon_data(polygon_data))
            
            # Also send via WebRTC data channel if available
            data_channel = self.video_manager.get_data_channel(self.connection_id)
            if data_channel and data_channel.readyState == "open":
                # Send polygon data as JSON
                polygon_json = json.dumps(polygon_data)
                data_channel.send(polygon_json)
                logger.debug(f"Sent polygon data for connection {self.connection_id}")
        except Exception as e:
            logger.error(f"Failed to send polygon data: {e}")


async def run_video_segmentation(segmentation_args, connection_id: str = None):
    """
    Run video segmentation with WebRTC integration.
    
    Args:
        segmentation_args: Arguments for segmentation model
        connection_id: Optional connection ID to process
    """
    # Initialize processor
    processor = VideoSegmentationProcessor(segmentation_args)
    await processor.initialize()
    
    # Set up polygon data channel
    video_manager = get_video_manager()
    polygon_channel = PolygonDataChannel(video_manager)
    
    # Set up polygon callback
    def handle_polygon(polygon_data):
        polygon_channel.send_polygon(polygon_data)
    
    processor.polygon_callback = handle_polygon
    
    # Start processing
    if connection_id:
        await processor.start_processing(connection_id)
        polygon_channel.set_connection(connection_id)
    
    logger.info("Video segmentation started")
    
    # Keep running until stopped
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping video segmentation...")
    finally:
        await processor.stop_processing()


def create_segmentation_args():
    """Create default segmentation arguments."""
    class Args:
        def __init__(self):
            self.model = 'rvm_mobilenetv3.pth'
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.dsr = 0.25
            self.fp16 = False
            self.polygon_threshold = 0.5
            self.polygon_min_area = 2000
            self.polygon_epsilon = 0.015
    
    return Args()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create segmentation arguments
    args = create_segmentation_args()
    
    # Run video segmentation
    asyncio.run(run_video_segmentation(args))
