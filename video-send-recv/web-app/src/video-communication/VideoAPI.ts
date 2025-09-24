/**
 * Simple Video API - Easy-to-use functions for video communication.
 * 
 * This module provides simple functions that other modules can import and use.
 */

import { 
  VideoCommunicationManager, 
  VideoConfig, 
  VideoConnection, 
  VideoConnectionEvents 
} from './VideoCommunicationManager';
import { IntensityMetrics } from '../types';

// Global video manager instance
let videoManager: VideoCommunicationManager | null = null;

/**
 * Initialize the video communication system.
 */
export function initVideoSystem(config: VideoConfig): VideoCommunicationManager {
  videoManager = new VideoCommunicationManager(config);
  console.log('Video communication system initialized');
  return videoManager;
}

/**
 * Get the global video manager instance.
 */
export function getVideoManager(): VideoCommunicationManager {
  if (!videoManager) {
    throw new Error('Video system not initialized. Call initVideoSystem() first.');
  }
  return videoManager;
}

/**
 * Create a new video connection.
 */
export async function createConnection(connectionId: string): Promise<VideoConnection> {
  const manager = getVideoManager();
  return await manager.createConnection(connectionId);
}

/**
 * Start camera and get media stream.
 */
export async function startCamera(connectionId: string): Promise<MediaStream> {
  const manager = getVideoManager();
  return await manager.startCamera(connectionId);
}

/**
 * Connect to the video server.
 */
export async function connectVideo(connectionId: string, stream: MediaStream): Promise<void> {
  const manager = getVideoManager();
  return await manager.connect(connectionId, stream);
}

/**
 * Disconnect a video connection.
 */
export async function disconnectVideo(connectionId: string): Promise<void> {
  const manager = getVideoManager();
  return await manager.disconnect(connectionId);
}

/**
 * Set event handlers for video communication.
 */
export function setEventHandlers(events: VideoConnectionEvents): void {
  const manager = getVideoManager();
  manager.setEventHandlers(events);
}

/**
 * Get connection information.
 */
export function getConnectionInfo(connectionId: string): VideoConnection | undefined {
  const manager = getVideoManager();
  return manager.getConnection(connectionId);
}

/**
 * Get all connections.
 */
export function getAllConnections(): VideoConnection[] {
  const manager = getVideoManager();
  return manager.getAllConnections();
}

/**
 * Check if a connection is active.
 */
export function isConnectionActive(connectionId: string): boolean {
  const manager = getVideoManager();
  return manager.isConnectionActive(connectionId);
}

/**
 * Get connection count.
 */
export function getConnectionCount(): number {
  const manager = getVideoManager();
  return manager.getConnectionCount();
}

/**
 * Cleanup all connections.
 */
export async function cleanupVideoSystem(): Promise<void> {
  const manager = getVideoManager();
  return await manager.cleanup();
}

/**
 * Convenience function to start video processing for a connection.
 */
export async function startVideoProcessing(
  connectionId: string,
  intensityCallback?: (connectionId: string, metrics: IntensityMetrics) => void
): Promise<void> {
  const manager = getVideoManager();
  
  if (intensityCallback) {
    manager.setEventHandlers({
      onIntensityUpdate: intensityCallback
    });
  }
  
  console.log(`Started video processing for ${connectionId}`);
}

/**
 * Get default video configuration.
 */
export function getDefaultConfig(): VideoConfig {
  return {
    serverUrl: 'http://localhost:8080',
    videoConstraints: {
      width: { ideal: 640, max: 640 },
      height: { ideal: 360, max: 360 },
      frameRate: { ideal: 30, max: 30 },
      facingMode: 'user'
    },
    dataChannelConfig: {
      ordered: false,
      maxRetransmits: 0,
      protocol: 'intensity-v1'
    },
    reconnectInterval: 5000,
    maxReconnectAttempts: 3
  };
}

/**
 * Example intensity handler that logs the data.
 */
export function exampleIntensityHandler(connectionId: string, data: IntensityMetrics): void {
  console.log(`Intensity for ${connectionId}: Current=${data.intensity.toFixed(1)}, Average=${data.avg_intensity.toFixed(1)}`);
}

/**
 * Example usage function.
 */
export async function exampleUsage(): Promise<void> {
  console.log('=== Video API Usage Example ===');
  
  // Initialize the system
  const config = getDefaultConfig();
  initVideoSystem(config);
  
  // Create a connection
  const connectionId = 'example_conn';
  await createConnection(connectionId);
  
  // Set up event handlers
  setEventHandlers({
    onIntensityUpdate: exampleIntensityHandler,
    onConnectionStateChange: (id, state) => {
      console.log(`Connection ${id} state: ${state}`);
    },
    onError: (id, error) => {
      console.error(`Error for ${id}:`, error);
    }
  });
  
  // Start camera
  const stream = await startCamera(connectionId);
  console.log('Camera started:', stream);
  
  // Connect to server
  await connectVideo(connectionId, stream);
  console.log('Connected to server');
  
  // Get connection info
  const info = getConnectionInfo(connectionId);
  console.log('Connection info:', info);
  
  // Cleanup
  await disconnectVideo(connectionId);
  console.log('Disconnected');
}
