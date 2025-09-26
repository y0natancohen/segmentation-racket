/**
 * Polygon Display Component - Shows segmentation polygons from the backend.
 */

import React, { useState, useEffect, useRef } from 'react';

export interface PolygonData {
  connection_id: string;
  polygon: number[][];
  timestamp: number;
  frame_shape: number[];
  original_image_size: number[];
}

interface PolygonDisplayProps {
  width?: number;
  height?: number;
  className?: string;
  style?: React.CSSProperties;
}

export const PolygonDisplay: React.FC<PolygonDisplayProps> = ({
  width = 640,
  height = 360,
  className,
  style
}) => {
  const [polygonData, setPolygonData] = useState<PolygonData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Set up WebSocket connection to receive polygon data
    console.log('🔌 Attempting to connect to WebSocket: ws://localhost:8080/polygon');
    const ws = new WebSocket('ws://localhost:8080/polygon');
    
    ws.onopen = () => {
      console.log('✅ Polygon WebSocket connected');
      setIsConnected(true);
    };
    
    ws.onmessage = (event) => {
      try {
        const data: PolygonData = JSON.parse(event.data);
        console.log('🎯 Polygon data received:', data);
        console.log('🎯 Polygon points:', data.polygon);
        console.log('🎯 Polygon string representation:', JSON.stringify(data.polygon));
        setPolygonData(data);
      } catch (error) {
        console.error('Failed to parse polygon data:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('❌ Polygon WebSocket disconnected');
      setIsConnected(false);
    };
    
    ws.onerror = (error) => {
      console.error('❌ Polygon WebSocket error:', error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (polygonData && polygonData.polygon) {
      // Set up drawing style
      ctx.strokeStyle = '#00ff00';
      ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
      ctx.lineWidth = 2;

      // Scale polygon to canvas size based on original image dimensions
      // This allows the polygon to be larger in the window
      const scaleX = width / polygonData.original_image_size[1];  // width / original_width
      const scaleY = height / polygonData.original_image_size[0];  // height / original_height
      
      console.log(`🎯 Polygon scaling: canvas=${width}x${height}, original=${polygonData.original_image_size[1]}x${polygonData.original_image_size[0]}, scale=${scaleX.toFixed(2)}x${scaleY.toFixed(2)}`);

      // Draw polygon
      ctx.beginPath();
      polygonData.polygon.forEach((point, index) => {
        const x = point[0] * scaleX;
        const y = point[1] * scaleY;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw polygon points
      ctx.fillStyle = '#ff0000';
      polygonData.polygon.forEach(point => {
        const x = point[0] * scaleX;
        const y = point[1] * scaleY;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, [polygonData, width, height]);

  return (
    <div className={className} style={style}>
      <div style={{ 
        position: 'relative', 
        width, 
        height, 
        border: '2px solid #333',
        backgroundColor: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            pointerEvents: 'none'
          }}
        />
        
        {/* Connection status */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          background: isConnected ? 'rgba(0, 255, 0, 0.8)' : 'rgba(255, 0, 0, 0.8)',
          color: 'white',
          padding: '5px 10px',
          borderRadius: '5px',
          fontSize: '12px',
          fontWeight: 'bold'
        }}>
          {isConnected ? 'Polygon Connected' : 'Polygon Disconnected'}
        </div>

        {/* Polygon info */}
        {polygonData && (
          <div style={{
            position: 'absolute',
            bottom: '10px',
            left: '10px',
            background: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '5px 10px',
            borderRadius: '5px',
            fontSize: '12px'
          }}>
            <div>Points: {polygonData.polygon.length}</div>
            <div>Original: {polygonData.original_image_size[1]}x{polygonData.original_image_size[0]}</div>
            <div>Canvas: {width}x{height}</div>
            <div>Scale: {(width / polygonData.original_image_size[1]).toFixed(2)}x{(height / polygonData.original_image_size[0]).toFixed(2)}</div>
            <div>Time: {new Date(polygonData.timestamp * 1000).toLocaleTimeString()}</div>
          </div>
        )}

        {/* No data message */}
        {!polygonData && (
          <div style={{
            color: '#666',
            fontSize: '16px',
            textAlign: 'center'
          }}>
            Waiting for polygon data...
          </div>
        )}
      </div>
    </div>
  );
};

export default PolygonDisplay;
