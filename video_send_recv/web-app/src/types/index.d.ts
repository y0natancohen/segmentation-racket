export interface IntensityMetrics {
  ts: number;
  intensity: number; // 0-255
  intensity_norm: number; // 0.0-1.0
  avg_intensity: number; // 0-255 average
  avg_intensity_norm: number; // 0.0-1.0 average
  frame_count: number; // number of frames averaged
}

export interface ConnectionStats {
  outboundFps: number;
  outboundBitrate: number;
  inboundFps: number;
  messagesPerSecond: number;
}

export interface VideoConstraints {
  width: { ideal: number; max: number };
  height: { ideal: number; max: number };
  frameRate: { ideal: number; max: number };
  facingMode: 'user' | 'environment';
}

export interface DataChannelConfig {
  ordered: boolean;
  maxRetransmits: number;
  protocol: string;
}

export interface SignalingRequest {
  sdp: string;
  type: 'offer';
}

export interface SignalingResponse {
  sdp: string;
  type: 'answer';
}
