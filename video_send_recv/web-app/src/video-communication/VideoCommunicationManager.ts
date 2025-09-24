/**
 * Video Communication Manager - Core class for video streaming and intensity analysis.
 * 
 * This class provides a clean API for video communication that can be used by other modules.
 */

import { IntensityMetrics, ConnectionStats, VideoConstraints, DataChannelConfig } from '../types';

export interface VideoConfig {
  serverUrl: string;
  videoConstraints: VideoConstraints;
  dataChannelConfig: DataChannelConfig;
  reconnectInterval: number;
  maxReconnectAttempts: number;
}

export interface VideoConnection {
  id: string;
  isConnected: boolean;
  isStreaming: boolean;
  stats: ConnectionStats;
  lastIntensity: IntensityMetrics | null;
}

export interface VideoConnectionEvents {
  onConnectionStateChange?: (connectionId: string, state: RTCPeerConnectionState) => void;
  onIntensityUpdate?: (connectionId: string, metrics: IntensityMetrics) => void;
  onStatsUpdate?: (connectionId: string, stats: ConnectionStats) => void;
  onError?: (connectionId: string, error: Error) => void;
  onStreamReady?: (connectionId: string, stream: MediaStream) => void;
}

export class VideoCommunicationManager {
  private config: VideoConfig;
  private connections: Map<string, VideoConnection> = new Map();
  private peerConnections: Map<string, RTCPeerConnection> = new Map();
  private dataChannels: Map<string, RTCDataChannel> = new Map();
  private metricsCollectors: Map<string, any> = new Map();
  private events: VideoConnectionEvents = {};
  private reconnectTimeouts: Map<string, number> = new Map();

  constructor(config: VideoConfig) {
    this.config = config;
  }

  /**
   * Set event handlers for video communication events.
   */
  setEventHandlers(events: VideoConnectionEvents): void {
    this.events = { ...this.events, ...events };
  }

  /**
   * Create a new video connection.
   */
  async createConnection(connectionId: string): Promise<VideoConnection> {
    console.log(`Creating video connection: ${connectionId}`);

    const connection: VideoConnection = {
      id: connectionId,
      isConnected: false,
      isStreaming: false,
      stats: {
        outboundFps: 0,
        outboundBitrate: 0,
        inboundFps: 0,
        messagesPerSecond: 0
      },
      lastIntensity: null
    };

    this.connections.set(connectionId, connection);
    return connection;
  }

  /**
   * Start camera and get media stream.
   */
  async startCamera(connectionId: string): Promise<MediaStream> {
    console.log(`Starting camera for connection: ${connectionId}`);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: this.config.videoConstraints,
        audio: false
      });

      // Apply frame rate constraint
      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack) {
        await videoTrack.applyConstraints({ frameRate: 30 });
      }

      this.events.onStreamReady?.(connectionId, stream);
      return stream;
    } catch (error) {
      const err = new Error(`Camera access failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      this.events.onError?.(connectionId, err);
      throw err;
    }
  }

  /**
   * Connect to the video server.
   */
  async connect(connectionId: string, stream: MediaStream): Promise<void> {
    console.log(`Connecting to server for: ${connectionId}`);

    try {
      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: ['stun:stun.l.google.com:19302'] }],
        bundlePolicy: 'max-bundle',
      });

      this.peerConnections.set(connectionId, pc);

      // Set up event handlers
      this.setupPeerConnectionHandlers(pc, connectionId);

      // Add video track
      await this.addVideoTrack(pc, stream);

      // Create data channel
      const dataChannel = pc.createDataChannel('metrics', this.config.dataChannelConfig);
      this.dataChannels.set(connectionId, dataChannel);
      this.setupDataChannelHandlers(dataChannel, connectionId);

      // Create offer and signal to server
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const answer = await this.signalToServer(offer);
      await pc.setRemoteDescription(answer);

      // Update connection state
      const connection = this.connections.get(connectionId);
      if (connection) {
        connection.isConnected = true;
        connection.isStreaming = true;
      }

      console.log(`Connected successfully: ${connectionId}`);
    } catch (error) {
      const err = new Error(`Connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      this.events.onError?.(connectionId, err);
      throw err;
    }
  }

  /**
   * Disconnect a video connection.
   */
  async disconnect(connectionId: string): Promise<void> {
    console.log(`Disconnecting: ${connectionId}`);

    const pc = this.peerConnections.get(connectionId);
    if (pc) {
      pc.close();
      this.peerConnections.delete(connectionId);
    }

    const dataChannel = this.dataChannels.get(connectionId);
    if (dataChannel) {
      dataChannel.close();
      this.dataChannels.delete(connectionId);
    }

    const connection = this.connections.get(connectionId);
    if (connection) {
      connection.isConnected = false;
      connection.isStreaming = false;
    }

    // Clear reconnect timeout
    const timeout = this.reconnectTimeouts.get(connectionId);
    if (timeout) {
      clearTimeout(timeout);
      this.reconnectTimeouts.delete(connectionId);
    }
  }

  /**
   * Get connection information.
   */
  getConnection(connectionId: string): VideoConnection | undefined {
    return this.connections.get(connectionId);
  }

  /**
   * Get all connections.
   */
  getAllConnections(): VideoConnection[] {
    return Array.from(this.connections.values());
  }

  /**
   * Check if a connection is active.
   */
  isConnectionActive(connectionId: string): boolean {
    const connection = this.connections.get(connectionId);
    return connection?.isConnected ?? false;
  }

  /**
   * Get connection count.
   */
  getConnectionCount(): number {
    return this.connections.size;
  }

  /**
   * Cleanup all connections.
   */
  async cleanup(): Promise<void> {
    console.log('Cleaning up all connections...');
    
    for (const connectionId of this.connections.keys()) {
      await this.disconnect(connectionId);
    }
    
    this.connections.clear();
    this.peerConnections.clear();
    this.dataChannels.clear();
    this.metricsCollectors.clear();
  }

  private async addVideoTrack(pc: RTCPeerConnection, stream: MediaStream): Promise<void> {
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack) throw new Error('No video track found');

    const transceiver = pc.addTransceiver(videoTrack, { direction: 'sendonly' });
    
    // Set codec preferences
    await this.setCodecPreferences(transceiver);

    // Configure encoding parameters
    const sender = transceiver.sender;
    const params = sender.getParameters();
    params.encodings = [{
      maxBitrate: 600_000,
      maxFramerate: 30,
      scaleResolutionDownBy: 1
    }];
    await sender.setParameters(params);
  }

  private async setCodecPreferences(transceiver: RTCRtpTransceiver): Promise<void> {
    try {
      const capabilities = RTCRtpSender.getCapabilities('video');
      if (!capabilities) return;

      // Prefer H.264
      const h264Codecs = capabilities.codecs.filter(c => c.mimeType.includes('H264'));
      if (h264Codecs.length > 0) {
        await transceiver.setCodecPreferences(h264Codecs);
      }
    } catch (error) {
      console.warn('Failed to set codec preferences:', error);
    }
  }

  private setupPeerConnectionHandlers(pc: RTCPeerConnection, connectionId: string): void {
    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      console.log(`Connection state changed to ${state} for ${connectionId}`);
      
      const connection = this.connections.get(connectionId);
      if (connection) {
        connection.isConnected = state === 'connected';
      }
      
      this.events.onConnectionStateChange?.(connectionId, state);
      
      if (state === 'failed' || state === 'disconnected') {
        this.handleConnectionFailure(connectionId);
      }
    };

    pc.ondatachannel = (event) => {
      const channel = event.channel;
      console.log(`Data channel received: ${channel.label} for ${connectionId}`);
      this.setupDataChannelHandlers(channel, connectionId);
    };
  }

  private setupDataChannelHandlers(dataChannel: RTCDataChannel, connectionId: string): void {
    dataChannel.onmessage = (event) => {
      try {
        const metrics: IntensityMetrics = JSON.parse(event.data);
        
        const connection = this.connections.get(connectionId);
        if (connection) {
          connection.lastIntensity = metrics;
        }
        
        this.events.onIntensityUpdate?.(connectionId, metrics);
      } catch (error) {
        console.error('Failed to parse metrics message:', error);
      }
    };

    dataChannel.onopen = () => {
      console.log(`Data channel opened for ${connectionId}`);
    };

    dataChannel.onclose = () => {
      console.log(`Data channel closed for ${connectionId}`);
    };
  }

  private async signalToServer(offer: RTCSessionDescriptionInit): Promise<RTCSessionDescriptionInit> {
    const response = await fetch(`${this.config.serverUrl}/offer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type
      }),
    });

    if (!response.ok) {
      throw new Error(`Signaling failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }

  private handleConnectionFailure(connectionId: string): void {
    console.log(`Handling connection failure for ${connectionId}`);
    
    // Implement reconnection logic if needed
    // For now, just disconnect
    this.disconnect(connectionId);
  }
}
