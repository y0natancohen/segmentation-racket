/**
 * Polygon data received from the Python segmentation server.
 */
export type PolygonData = {
  /** Polygon vertices as [[x,y], [x,y], ...] in original image coordinates */
  polygon: number[][];
  /** Server-side timestamp (seconds since epoch) */
  timestamp: number;
  /** [height, width] of the image the polygon was extracted from */
  original_image_size: number[];
};

/**
 * Configuration for the GameWebSocket connection.
 */
export type GameWebSocketConfig = {
  /** WebSocket server URL, e.g. "ws://localhost:8765" */
  serverUrl: string;
  /** JPEG quality for frame capture (0.0 – 1.0) */
  jpegQuality: number;
  /** Frame capture/send rate in fps */
  captureRate: number;
  /** Width to downscale captured frames to before sending */
  captureWidth: number;
  /** Height to downscale captured frames to before sending */
  captureHeight: number;
  /** Reconnect delay in ms */
  reconnectDelay: number;
  /** Maximum reconnect attempts (0 = unlimited) */
  maxReconnectAttempts: number;
};

/**
 * Events emitted by GameWebSocket.
 */
export type GameWebSocketEvents = {
  onPolygonData?: (data: PolygonData) => void;
  onConnectionStateChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
  onFrameSent?: () => void;
};

/**
 * Performance metrics tracked by the system.
 */
export type PerformanceMetrics = {
  /** Browser render FPS (Phaser) */
  renderFps: number;
  /** Polygon messages received per second */
  polygonFps: number;
  /** Frames sent to server per second */
  frameSendFps: number;
  /** Estimated round-trip latency in ms */
  roundTripMs: number;
};
