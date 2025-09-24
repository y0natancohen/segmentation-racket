"""
Tests for media processing functions.
"""
import pytest
import numpy as np
from media import mean_intensity, create_test_frame


class TestMediaProcessing:
    """Test media processing functions."""
    
    def test_all_black_frame(self):
        """Test intensity calculation for all-black frame."""
        frame = create_test_frame(intensity=0)
        intensity_255, intensity_norm = mean_intensity(frame)
        
        assert abs(intensity_255 - 0.0) < 1.0, f"Expected ~0, got {intensity_255}"
        assert abs(intensity_norm - 0.0) < 0.01, f"Expected ~0.0, got {intensity_norm}"
    
    def test_all_white_frame(self):
        """Test intensity calculation for all-white frame."""
        frame = create_test_frame(intensity=255)
        intensity_255, intensity_norm = mean_intensity(frame)
        
        assert abs(intensity_255 - 255.0) < 1.0, f"Expected ~255, got {intensity_255}"
        assert abs(intensity_norm - 1.0) < 0.01, f"Expected ~1.0, got {intensity_norm}"
    
    def test_gray_levels(self):
        """Test intensity calculation for various gray levels."""
        test_cases = [
            (64, 0.25),   # Dark gray
            (127, 0.5),   # Medium gray  
            (191, 0.75), # Light gray
        ]
        
        for intensity, expected_norm in test_cases:
            frame = create_test_frame(intensity=intensity)
            intensity_255, intensity_norm = mean_intensity(frame)
            
            # Allow ±5% tolerance
            expected_255 = intensity
            tolerance_255 = intensity * 0.05
            tolerance_norm = expected_norm * 0.05
            
            assert abs(intensity_255 - expected_255) < tolerance_255, \
                f"Intensity {intensity}: expected ~{expected_255}, got {intensity_255}"
            assert abs(intensity_norm - expected_norm) < tolerance_norm, \
                f"Intensity {intensity}: expected ~{expected_norm}, got {intensity_norm}"
    
    def test_random_frame_range(self):
        """Test that random frames produce results in valid range."""
        # Test with random intensity
        random_intensity = np.random.randint(0, 256)
        frame = create_test_frame(intensity=random_intensity)
        intensity_255, intensity_norm = mean_intensity(frame)
        
        assert 0 <= intensity_255 <= 255, f"Intensity out of range: {intensity_255}"
        assert 0.0 <= intensity_norm <= 1.0, f"Normalized intensity out of range: {intensity_norm}"
    
    def test_frame_dimensions(self):
        """Test with different frame dimensions."""
        dimensions = [(320, 240), (640, 360), (1280, 720)]
        
        for width, height in dimensions:
            frame = create_test_frame(width=width, height=height, intensity=128)
            intensity_255, intensity_norm = mean_intensity(frame)
            
            # Should be consistent regardless of dimensions
            assert abs(intensity_255 - 128.0) < 1.0, \
                f"Dimension {width}x{height}: expected ~128, got {intensity_255}"
            assert abs(intensity_norm - 0.5) < 0.01, \
                f"Dimension {width}x{height}: expected ~0.5, got {intensity_norm}"
    
    def test_luminance_calculation(self):
        """Test that luminance calculation is correct."""
        # Create frame with known RGB values
        # R=100, G=200, B=50 should give specific luminance
        frame = create_test_frame(intensity=0)  # Will be overwritten
        
        # Manually set RGB values
        arr = frame.to_ndarray(format="rgb24")
        arr[:, :, 0] = 100  # R
        arr[:, :, 1] = 200  # G  
        arr[:, :, 2] = 50   # B
        
        # Create new frame from modified array
        from av import VideoFrame
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        
        intensity_255, intensity_norm = mean_intensity(frame)
        
        # Expected luminance: 0.2126*100 + 0.7152*200 + 0.0722*50 = 21.26 + 143.04 + 3.61 = 167.91
        expected_luminance = 0.2126 * 100 + 0.7152 * 200 + 0.0722 * 50
        
        assert abs(intensity_255 - expected_luminance) < 1.0, \
            f"Expected luminance ~{expected_luminance}, got {intensity_255}"
        assert abs(intensity_norm - expected_luminance/255.0) < 0.01, \
            f"Expected normalized ~{expected_luminance/255.0}, got {intensity_norm}"
