export type IntensityMetrics = {
  ts: number;
  intensity: number; // 0-255
  intensity_norm: number; // 0.0-1.0
  avg_intensity: number; // 0-255 average
  avg_intensity_norm: number; // 0.0-1.0 average
  frame_count: number; // number of frames averaged
};

export type ConnectionStats = {
  outboundFps: number;
  outboundBitrate: number;
  inboundFps: number;
  messagesPerSecond: number;
};

export type VideoConstraints = {
  width: { ideal: number; max: number };
  height: { ideal: number; max: number };
  frameRate: { ideal: number; max: number };
  facingMode: 'user' | 'environment';
};

export type DataChannelConfig = {
  ordered: boolean;
  maxRetransmits: number;
  protocol: string;
};

export type SignalingRequest = {
  sdp: string;
  type: 'offer';
};

export type SignalingResponse = {
  sdp: string;
  type: 'answer';
};

export type PolygonData = {
  connection_id: string;
  polygon: number[][];
  timestamp: number;
  frame_shape: number[];
  original_image_size: number[];
};

export type VideoConfig = {
  serverUrl: string;
  videoConstraints: VideoConstraints;
  dataChannelConfig: DataChannelConfig;
  reconnectInterval: number;
  maxReconnectAttempts: number;
};

export type VideoConnection = {
  id: string;
  isConnected: boolean;
  isStreaming: boolean;
  stats: ConnectionStats;
  lastIntensity: IntensityMetrics | null;
};

export type VideoConnectionEvents = {
  onConnectionStateChange?: (connectionId: string, state: RTCPeerConnectionState) => void;
  onIntensityUpdate?: (connectionId: string, metrics: IntensityMetrics) => void;
  onStatsUpdate?: (connectionId: string, stats: ConnectionStats) => void;
  onError?: (connectionId: string, error: Error) => void;
  onStreamReady?: (connectionId: string, stream: MediaStream) => void;
};
