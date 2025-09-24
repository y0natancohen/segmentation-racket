"""
Media processing utilities for intensity calculation.
"""
import numpy as np
from av import VideoFrame


def mean_intensity(frame: VideoFrame):
    """
    Calculate mean intensity of a video frame.
    
    Args:
        frame: VideoFrame from aiortc
        
    Returns:
        Tuple of (intensity_0_255, intensity_normalized)
        - intensity_0_255: Mean intensity in range [0, 255]
        - intensity_normalized: Mean intensity normalized to [0.0, 1.0]
    """
    # Convert frame to RGB24 numpy array (HxWx3 uint8)
    arr = frame.to_ndarray(format="rgb24")
    
    # Convert RGB to luminance using standard weights
    # Y = 0.2126*R + 0.7152*G + 0.0722*B
    y = (0.2126 * arr[:, :, 0] + 
         0.7152 * arr[:, :, 1] + 
         0.0722 * arr[:, :, 2])
    
    # Calculate mean intensity
    mean_y = y.mean()
    
    # Return both 0-255 range and normalized 0-1 range
    return float(mean_y), float(mean_y / 255.0)


def create_test_frame(width: int = 640, height: int = 360, 
                     intensity: int = 127) -> VideoFrame:
    """
    Create a test frame with specified intensity.
    
    Args:
        width: Frame width
        height: Frame height  
        intensity: Gray intensity (0-255)
        
    Returns:
        VideoFrame with specified intensity
    """
    # Create RGB array with specified intensity
    arr = np.full((height, width, 3), intensity, dtype=np.uint8)
    
    # Create VideoFrame from array
    frame = VideoFrame.from_ndarray(arr, format="rgb24")
    return frame
