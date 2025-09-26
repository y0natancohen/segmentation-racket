/**
 * Polygon WebSocket Manager - Handles WebSocket connection for receiving polygon data.
 * 
 * This class manages the WebSocket connection to receive polygon data from the Python backend.
 */

import type { PolygonData } from './types';

export interface PolygonWebSocketEvents {
  onPolygonData?: (data: PolygonData) => void;
  onConnectionStateChange?: (connected: boolean) => void;
  onError?: (error: Error) => void;
}

export class PolygonWebSocketManager {
  private ws: WebSocket | null = null;
  private events: PolygonWebSocketEvents = {};
  private reconnectTimer: number = 0;
  private reconnectDelay: number = 2000; // 2 seconds
  private isConnecting: boolean = false;
  private serverUrl: string;

  constructor(serverUrl: string = 'ws://localhost:8080/polygon') {
    this.serverUrl = serverUrl;
  }

  /**
   * Set event handlers for polygon WebSocket events.
   */
  setEventHandlers(events: PolygonWebSocketEvents): void {
    this.events = { ...this.events, ...events };
  }

  /**
   * Connect to the polygon WebSocket server.
   */
  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('Polygon WebSocket already connected');
      return;
    }

    if (this.isConnecting) {
      console.log('Polygon WebSocket connection already in progress');
      return;
    }

    this.isConnecting = true;
    console.log(`Connecting to polygon WebSocket: ${this.serverUrl}`);

    try {
      this.ws = new WebSocket(this.serverUrl);
      
      this.ws.onopen = () => {
        console.log('✅ Polygon WebSocket connected');
        this.isConnecting = false;
        this.reconnectTimer = 0;
        this.events.onConnectionStateChange?.(true);
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data: PolygonData = JSON.parse(event.data);
          console.log('🎯 Polygon data received:', data);
          this.events.onPolygonData?.(data);
        } catch (error) {
          console.error('Failed to parse polygon data:', error);
          this.events.onError?.(new Error(`Failed to parse polygon data: ${error}`));
        }
      };
      
      this.ws.onclose = () => {
        console.log('❌ Polygon WebSocket disconnected');
        this.isConnecting = false;
        this.events.onConnectionStateChange?.(false);
        this.scheduleReconnect();
      };
      
      this.ws.onerror = (error) => {
        console.error('❌ Polygon WebSocket error:', error);
        this.isConnecting = false;
        this.events.onError?.(new Error(`WebSocket error: ${error}`));
        this.scheduleReconnect();
      };

    } catch (error) {
      this.isConnecting = false;
      console.error('Failed to create polygon WebSocket connection:', error);
      this.events.onError?.(new Error(`Failed to create WebSocket connection: ${error}`));
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the polygon WebSocket server.
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = 0;
    }
    
    this.isConnecting = false;
    console.log('Polygon WebSocket disconnected');
  }

  /**
   * Check if the WebSocket is connected.
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state.
   */
  getConnectionState(): string {
    if (!this.ws) return 'disconnected';
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'closed';
      default: return 'unknown';
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    
    this.reconnectTimer = window.setTimeout(() => {
      console.log('Attempting to reconnect polygon WebSocket...');
      this.connect();
    }, this.reconnectDelay);
  }
}
