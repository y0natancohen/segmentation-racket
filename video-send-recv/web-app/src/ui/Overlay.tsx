import React from 'react';
import { IntensityMetrics, ConnectionStats } from '../types';

interface OverlayProps {
  metrics: IntensityMetrics | null;
  stats: ConnectionStats;
  isConnected: boolean;
  onReconnect: () => void;
}

export const Overlay: React.FC<OverlayProps> = ({ 
  metrics, 
  stats, 
  isConnected, 
  onReconnect 
}) => {
  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '16px',
      borderRadius: '8px',
      fontFamily: 'monospace',
      fontSize: '14px',
      minWidth: '300px',
      zIndex: 1000
    }}>
      <div style={{ marginBottom: '12px', fontWeight: 'bold' }}>
        Video Intensity Analysis
      </div>
      
      <div style={{ marginBottom: '8px' }}>
        <div>Connection: <span style={{ color: isConnected ? '#4CAF50' : '#F44336' }}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span></div>
      </div>

      {metrics && (
        <div style={{ marginBottom: '8px' }}>
          <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#4CAF50', marginBottom: '4px' }}>
            Average Intensity: {metrics.avg_intensity.toFixed(1)} / 255
          </div>
          <div style={{ fontSize: '14px', color: '#81C784' }}>
            Avg Normalized: {metrics.avg_intensity_norm.toFixed(3)}
          </div>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            Current: {metrics.intensity.toFixed(1)} | Frames: {metrics.frame_count}
          </div>
          <div style={{ fontSize: '10px', color: '#999' }}>
            {new Date(metrics.ts * 1000).toLocaleTimeString()}
          </div>
        </div>
      )}

      <div style={{ marginBottom: '8px' }}>
        <div>Outbound FPS: {stats.outboundFps.toFixed(1)}</div>
        <div>Outbound Bitrate: {(stats.outboundBitrate / 1000).toFixed(1)} kbps</div>
        <div>Inbound FPS: {stats.inboundFps.toFixed(1)}</div>
        <div>Messages/sec: {stats.messagesPerSecond.toFixed(1)}</div>
      </div>

      {!isConnected && (
        <button
          onClick={onReconnect}
          style={{
            background: '#2196F3',
            color: 'white',
            border: 'none',
            padding: '8px 16px',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          Reconnect
        </button>
      )}
    </div>
  );
};
