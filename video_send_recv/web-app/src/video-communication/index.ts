/**
 * Video Communication API - Main entry point.
 * 
 * This module exports all the video communication functionality for easy use by other modules.
 */

// Core API
export {
  initVideoSystem,
  createConnection,
  startCamera,
  connectVideo,
  disconnectVideo,
  setEventHandlers,
  getConnectionInfo,
  getAllConnections,
  isConnectionActive,
  getConnectionCount,
  cleanupVideoSystem,
  getDefaultConfig,
  exampleIntensityHandler,
  exampleUsage
} from './VideoAPI';

// Core classes
export {
  VideoCommunicationManager
} from './VideoCommunicationManager';
export type {
  VideoConfig,
  VideoConnection,
  VideoConnectionEvents
} from './VideoCommunicationManager';

// Intensity analysis
export {
  IntensityAnalyzer
} from './IntensityAnalyzer';
export type {
  IntensityAlert,
  IntensityHistory,
  IntensityStats
} from './IntensityAnalyzer';

// Example usage
export {
  VideoApplication,
  VideoComponent,
  exampleBasicUsage,
  exampleAdvancedUsage
} from './ExampleUsage';

// Re-export types from main types file
export type {
  IntensityMetrics,
  ConnectionStats,
  VideoConstraints,
  DataChannelConfig
} from '../types';
