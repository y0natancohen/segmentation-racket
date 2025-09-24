/**
 * Example usage of the Video Communication API.
 * 
 * This module demonstrates how other modules can use the video communication system.
 */

import { 
  initVideoSystem, 
  createConnection, 
  connectVideo, 
  disconnectVideo,
  setEventHandlers,
  getConnectionInfo,
  getAllConnections,
  cleanupVideoSystem,
  getDefaultConfig
} from './VideoAPI';
import { VideoConfig } from './VideoCommunicationManager';
import { IntensityAnalyzer, IntensityAlert } from './IntensityAnalyzer';
// import { IntensityMetrics } from '../types';

/**
 * Example class that demonstrates advanced video communication usage.
 */
export class VideoApplication {
  private intensityAnalyzer: IntensityAnalyzer;
  private connections: Map<string, string> = new Map(); // connectionId -> userId

  constructor() {
    this.intensityAnalyzer = new IntensityAnalyzer({
      lowThreshold: 15,
      highThreshold: 235,
      changeThreshold: 40,
      stabilityWindow: 10,
      maxHistory: 200
    });
  }

  /**
   * Initialize the video application.
   */
  async initialize(): Promise<void> {
    console.log('Initializing video application...');
    
    // Initialize video system with custom config
    const config: VideoConfig = {
      ...getDefaultConfig(),
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
    
    initVideoSystem(config);
    
    // Set up event handlers
    this.setupEventHandlers();
    
    console.log('Video application initialized');
  }

  /**
   * Add a new user with video connection.
   */
  async addUser(userId: string): Promise<string> {
    console.log(`Adding user: ${userId}`);
    
    const connectionId = `user_${userId}_${Date.now()}`;
    
    try {
      // Create connection
      await createConnection(connectionId);
      
      // Start camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640, max: 640 },
          height: { ideal: 360, max: 360 },
          frameRate: { ideal: 30, max: 30 },
          facingMode: 'user'
        },
        audio: false
      });
      console.log(`Camera started for ${userId}`);
      
      // Connect to server
      await connectVideo(connectionId, stream);
      console.log(`Connected to server for ${userId}`);
      
      // Store mapping
      this.connections.set(connectionId, userId);
      
      return connectionId;
    } catch (error) {
      console.error(`Failed to add user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Remove a user and disconnect their video.
   */
  async removeUser(userId: string): Promise<void> {
    console.log(`Removing user: ${userId}`);
    
    // Find connection for user
    const connectionId = this.findConnectionForUser(userId);
    if (!connectionId) {
      console.warn(`No connection found for user: ${userId}`);
      return;
    }
    
    try {
      // Disconnect video
      await disconnectVideo(connectionId);
      
      // Remove from mapping
      this.connections.delete(connectionId);
      
      console.log(`User ${userId} removed successfully`);
    } catch (error) {
      console.error(`Failed to remove user ${userId}:`, error);
    }
  }

  /**
   * Get intensity analysis for a user.
   */
  getIntensityAnalysis(userId: string): any {
    const connectionId = this.findConnectionForUser(userId);
    if (!connectionId) {
      return null;
    }
    
    return this.intensityAnalyzer.getIntensityStats(connectionId);
  }

  /**
   * Get all alerts.
   */
  getAllAlerts(): IntensityAlert[] {
    return this.intensityAnalyzer.getAllAlerts();
  }

  /**
   * Get alerts for a specific user.
   */
  getUserAlerts(userId: string): IntensityAlert[] {
    const connectionId = this.findConnectionForUser(userId);
    if (!connectionId) {
      return [];
    }
    
    return this.intensityAnalyzer.getAlertsForConnection(connectionId);
  }

  /**
   * Get application statistics.
   */
  getApplicationStats(): {
    totalUsers: number;
    activeConnections: number;
    totalAlerts: number;
    connections: any[];
  } {
    const allConnections = getAllConnections();
    
    return {
      totalUsers: this.connections.size,
      activeConnections: allConnections.filter(c => c.isConnected).length,
      totalAlerts: this.intensityAnalyzer.getAllAlerts().length,
      connections: allConnections.map(conn => ({
        id: conn.id,
        isConnected: conn.isConnected,
        isStreaming: conn.isStreaming,
        lastIntensity: conn.lastIntensity?.avg_intensity || 0
      }))
    };
  }

  /**
   * Shutdown the application.
   */
  async shutdown(): Promise<void> {
    console.log('Shutting down video application...');
    
    // Disconnect all users
    for (const [, userId] of this.connections) {
      await this.removeUser(userId);
    }
    
    // Cleanup video system
    await cleanupVideoSystem();
    
    console.log('Video application shutdown complete');
  }

  private setupEventHandlers(): void {
    setEventHandlers({
      onConnectionStateChange: (connectionId, state) => {
        const userId = this.connections.get(connectionId);
        console.log(`Connection state changed for ${userId || connectionId}: ${state}`);
      },
      
      onIntensityUpdate: (connectionId, metrics) => {
        // Process intensity data
        const stats = this.intensityAnalyzer.processIntensity(connectionId, metrics);
        
        // Log significant changes
        if (stats.stability < 0.5) {
          console.warn(`Low stability detected for ${connectionId}: ${(stats.stability * 100).toFixed(1)}%`);
        }
        
        // Update alerts (stored in analyzer)
      },
      
      onStatsUpdate: (connectionId, stats) => {
        const userId = this.connections.get(connectionId);
        if (userId) {
          console.log(`Stats for ${userId}: FPS=${stats.outboundFps.toFixed(1)}, Bitrate=${(stats.outboundBitrate/1000).toFixed(1)}kbps`);
        }
      },
      
      onError: (connectionId, error) => {
        const userId = this.connections.get(connectionId);
        console.error(`Error for ${userId || connectionId}:`, error);
      },
      
      onStreamReady: (connectionId, stream) => {
        const userId = this.connections.get(connectionId);
        console.log(`Stream ready for ${userId || connectionId}:`, stream);
      }
    });
  }

  private findConnectionForUser(userId: string): string | null {
    for (const [connectionId, uid] of this.connections) {
      if (uid === userId) {
        return connectionId;
      }
    }
    return null;
  }
}

/**
 * Example usage functions.
 */
export async function exampleBasicUsage(): Promise<void> {
  console.log('=== Basic Video API Usage ===');
  
  // Initialize
  const config = getDefaultConfig();
  initVideoSystem(config);
  
  // Create connection
  const connectionId = 'example_conn';
  await createConnection(connectionId);
  
  // Set up simple event handlers
  setEventHandlers({
    onIntensityUpdate: (id, metrics) => {
      console.log(`Intensity for ${id}: ${metrics.avg_intensity.toFixed(1)}`);
    },
    onConnectionStateChange: (id, state) => {
      console.log(`Connection ${id} state: ${state}`);
    }
  });
  
  // Start camera and connect
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640, max: 640 },
      height: { ideal: 360, max: 360 },
      frameRate: { ideal: 30, max: 30 },
      facingMode: 'user'
    },
    audio: false
  });
  await connectVideo(connectionId, stream);
  
  // Get connection info
  const info = getConnectionInfo(connectionId);
  console.log('Connection info:', info);
  
  // Cleanup
  await disconnectVideo(connectionId);
  await cleanupVideoSystem();
}

export async function exampleAdvancedUsage(): Promise<void> {
  console.log('=== Advanced Video API Usage ===');
  
  // Create application
  const app = new VideoApplication();
  await app.initialize();
  
  // Add some users
  const users = ['user1', 'user2', 'user3'];
  
  for (const userId of users) {
    try {
      const connectionId = await app.addUser(userId);
      console.log(`Added user ${userId} with connection ${connectionId}`);
    } catch (error) {
      console.error(`Failed to add user ${userId}:`, error);
    }
  }
  
  // Wait a bit for data to accumulate
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  // Get statistics
  const stats = app.getApplicationStats();
  console.log('Application stats:', stats);
  
  // Get alerts
  const alerts = app.getAllAlerts();
  console.log('Total alerts:', alerts.length);
  
  // Get intensity analysis for first user
  const analysis = app.getIntensityAnalysis('user1');
  if (analysis) {
    console.log('Intensity analysis for user1:', analysis);
  }
  
  // Shutdown
  await app.shutdown();
}

/**
 * Example of integration with React components.
 */
export class VideoComponent {
  private app: VideoApplication;
  private userId: string;
  private connectionId: string | null = null;

  constructor(userId: string) {
    this.app = new VideoApplication();
    this.userId = userId;
  }

  async initialize(): Promise<void> {
    await this.app.initialize();
  }

  async startVideo(): Promise<void> {
    if (this.connectionId) {
      console.warn('Video already started');
      return;
    }

    this.connectionId = await this.app.addUser(this.userId);
  }

  async stopVideo(): Promise<void> {
    if (!this.connectionId) {
      console.warn('Video not started');
      return;
    }

    await this.app.removeUser(this.userId);
    this.connectionId = null;
  }

  getIntensityData(): any {
    return this.app.getIntensityAnalysis(this.userId);
  }

  getAlerts(): IntensityAlert[] {
    return this.app.getUserAlerts(this.userId);
  }

  async cleanup(): Promise<void> {
    if (this.connectionId) {
      await this.stopVideo();
    }
    await this.app.shutdown();
  }
}
