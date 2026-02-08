/**
 * Per-frame timing breakdown from the Python server (milliseconds).
 */
export type ServerTimings = {
  /** JPEG decode time */
  decode_ms: number;
  /** RVM model inference time */
  inference_ms: number;
  /** Polygon generation (threshold + contour + simplification) */
  polygon_ms: number;
  /** Total server-side processing time */
  total_ms: number;
};

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
  /** Per-frame timing breakdown from the server */
  server_timings?: ServerTimings;
  /** Integer timestamp (ms since epoch) of the captured frame this polygon was
   *  computed from. Echoed by the server — used to correlate frame + polygon. */
  frame_timestamp?: number;
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
  /** Called when a polygon + its matching frame arrive.
   *  `frameImageData` is the exact ImageData of the captured frame (or null if
   *  the frame could not be correlated). */
  onPolygonData?: (data: PolygonData, frameImageData: ImageData | null) => void;
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

/**
 * Detailed per-component timing averages (milliseconds) for the full pipeline.
 */
export type DetailedTimingStats = {
  // Client-side
  /** Frame capture total (draw + encode) */
  capture_ms: number;
  /** Canvas drawImage time */
  capture_draw_ms: number;
  /** JPEG encode + arrayBuffer time */
  capture_encode_ms: number;

  // Server-side (from server_timings in response)
  /** JPEG decode on server */
  server_decode_ms: number;
  /** RVM inference on server */
  server_inference_ms: number;
  /** Polygon generation on server */
  server_polygon_ms: number;
  /** Total server processing */
  server_total_ms: number;

  // Round-trip
  /** Full round-trip time (send frame to receive polygon) */
  rtt_ms: number;
  /** Estimated network overhead (RTT minus server processing) */
  network_ms: number;

  // Client-side receive
  /** Polygon application time (scale, physics, draw) — tracked by MainScene */
  apply_ms: number;

  /** End-to-end estimated latency: capture + RTT + apply */
  overall_ms: number;
};
