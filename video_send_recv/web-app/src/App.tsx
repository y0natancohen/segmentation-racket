import { useState, useEffect, useRef } from 'react';
import { 
  initVideoSystem, 
  createConnection, 
  connectVideo, 
  disconnectVideo,
  setEventHandlers,
  cleanupVideoSystem,
  getDefaultConfig
} from './video-communication';
import { Overlay } from './ui/Overlay';
import { IntensityMetrics, ConnectionStats } from './types';
import PolygonDisplay from './components/PolygonDisplay';

function App() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [metrics, setMetrics] = useState<IntensityMetrics | null>(null);
  const [stats, setStats] = useState<ConnectionStats>({
    outboundFps: 0,
    outboundBitrate: 0,
    inboundFps: 0,
    messagesPerSecond: 0
  });
  const [error, setError] = useState<string | null>(null);
  const [connectionId, setConnectionId] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);

  const startCamera = async () => {
    try {
      setError(null);
      console.log('Requesting camera access...');
      
      // Create connection
      const connId = `conn_${Date.now()}`;
      setConnectionId(connId);
      await createConnection(connId);
      
      // Start camera using the new API
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640, max: 640 },
          height: { ideal: 360, max: 360 },
          frameRate: { ideal: 30, max: 30 },
          facingMode: 'user'
        },
        audio: false
      });
      console.log('Camera stream obtained:', mediaStream);
      
      // Set the stream state - useEffect will handle video setup
      setStream(mediaStream);
      
    } catch (err) {
      console.error('Camera access error:', err);
      setError(`Camera access failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const connect = async () => {
    if (!stream || !connectionId) {
      setError('No camera stream or connection available');
      return;
    }

    try {
      setError(null);
      
      // Connect using the new API
      await connectVideo(connectionId, stream);
      setIsConnected(true);
      
    } catch (err) {
      setError(`Connection failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setIsConnected(false);
    }
  };

  const disconnect = async () => {
    if (connectionId) {
      await disconnectVideo(connectionId);
    }
    setIsConnected(false);
    setMetrics(null);
  };

  const reconnect = async () => {
    await disconnect();
    setTimeout(async () => {
      if (stream && connectionId) {
        await connect();
      }
    }, 1000);
  };

  // Initialize video system
  useEffect(() => {
    const config = getDefaultConfig();
    initVideoSystem(config);
    
    // Set up event handlers
    setEventHandlers({
      onIntensityUpdate: (_, intensityMetrics) => {
        setMetrics(intensityMetrics);
      },
      onConnectionStateChange: (_, state) => {
        setIsConnected(state === 'connected');
      },
      onStatsUpdate: (_, newStats) => {
        setStats(prev => ({ ...prev, ...newStats }));
      },
      onError: (_, error) => {
        setError(`Video error: ${error.message}`);
      }
    });
    
    return () => {
      cleanupVideoSystem();
    };
  }, []);

  // Handle video element setup when stream changes
  useEffect(() => {
    if (stream && videoRef.current) {
      console.log('Stream changed, setting up video element...');
      console.log('Video element:', videoRef.current);
      console.log('Stream:', stream);
      
      videoRef.current.srcObject = stream;
      console.log('Video srcObject set to:', videoRef.current.srcObject);
      
      // Ensure video plays
      videoRef.current.play().then(() => {
        console.log('Video play started successfully');
      }).catch(err => {
        console.error('Video play failed:', err);
      });
    } else if (stream && !videoRef.current) {
      console.error('Stream available but video element not found');
    }
  }, [stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (connectionId) {
        disconnectVideo(connectionId);
      }
    };
  }, []);

  return (
    <div style={{ 
      position: 'relative', 
      width: '100vw', 
      height: '100vh', 
      background: '#000',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      {/* Debug info */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '10px',
        fontSize: '12px',
        zIndex: 1000
      }}>
        <div>Stream: {stream ? 'Active' : 'None'}</div>
        <div>Connected: {isConnected ? 'Yes' : 'No'}</div>
        <div>Error: {error || 'None'}</div>
      </div>

      {/* Large Average Intensity Display */}
      {metrics && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(0, 0, 0, 0.9)',
          color: 'white',
          padding: '30px',
          borderRadius: '15px',
          textAlign: 'center',
          zIndex: 1000,
          border: '3px solid #4CAF50',
          minWidth: '300px'
        }}>
          <div style={{ fontSize: '14px', color: '#81C784', marginBottom: '10px' }}>
            AVERAGE INTENSITY
          </div>
          <div style={{ fontSize: '48px', fontWeight: 'bold', color: '#4CAF50', marginBottom: '10px' }}>
            {metrics.avg_intensity.toFixed(1)}
          </div>
          <div style={{ fontSize: '18px', color: '#81C784', marginBottom: '5px' }}>
            {metrics.avg_intensity_norm.toFixed(3)}
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            Based on {metrics.frame_count} frames
          </div>
        </div>
      )}
      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={startCamera}
          disabled={!!stream}
          style={{
            padding: '10px 20px',
            margin: '0 10px',
            fontSize: '16px',
            background: stream ? '#666' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: stream ? 'not-allowed' : 'pointer'
          }}
        >
          {stream ? 'Camera Active' : 'Start Camera'}
        </button>
        
        <button
          onClick={connect}
          disabled={!stream || isConnected}
          style={{
            padding: '10px 20px',
            margin: '0 10px',
            fontSize: '16px',
            background: (!stream || isConnected) ? '#666' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (!stream || isConnected) ? 'not-allowed' : 'pointer'
          }}
        >
          {isConnected ? 'Connected' : 'Connect'}
        </button>

        <button
          onClick={disconnect}
          disabled={!isConnected}
          style={{
            padding: '10px 20px',
            margin: '0 10px',
            fontSize: '16px',
            background: !isConnected ? '#666' : '#F44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: !isConnected ? 'not-allowed' : 'pointer'
          }}
        >
          Disconnect
        </button>
      </div>

      {stream && (
        <div style={{
          display: 'flex',
          gap: '20px',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={() => {
              console.log('Video metadata loaded');
              console.log('Video dimensions:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight);
            }}
            onCanPlay={() => {
              console.log('Video can play');
              console.log('Video readyState:', videoRef.current?.readyState);
            }}
            onPlay={() => {
              console.log('Video started playing');
              console.log('Video currentTime:', videoRef.current?.currentTime);
            }}
            onError={(e) => {
              console.error('Video error:', e);
              console.error('Video error details:', e.currentTarget?.error);
            }}
            onLoadStart={() => console.log('Video load started')}
            onLoadedData={() => console.log('Video data loaded')}
            style={{
              width: '640px',
              height: '360px',
              border: '2px solid #333',
              borderRadius: '8px',
              backgroundColor: '#222',
              transform: 'scaleX(-1)' // Horizontal flip
            }}
          />
          
          <PolygonDisplay
            width={1280}
            height={720}
            style={{
              border: '2px solid #333',
              borderRadius: '8px'
            }}
          />
        </div>
      )}

      {!stream && (
        <div style={{
          width: '640px',
          height: '360px',
          border: '2px solid #333',
          borderRadius: '8px',
          backgroundColor: '#222',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#666',
          fontSize: '18px'
        }}>
          Click "Start Camera" to begin
        </div>
      )}

      {error && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(244, 67, 54, 0.9)',
          color: 'white',
          padding: '20px',
          borderRadius: '8px',
          fontSize: '16px',
          zIndex: 1001
        }}>
          {error}
        </div>
      )}

      <Overlay
        metrics={metrics}
        stats={stats}
        isConnected={isConnected}
        onReconnect={reconnect}
      />
    </div>
  );
}

export default App;
