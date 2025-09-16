/**
 * Unit tests for the Main Game Scene
 * Tests the core functionality of the Phaser game, WebSocket communication, and interpolation.
 */

import Phaser from "phaser";

// Mock WebSocket for testing
class MockWebSocket {
  public onopen: ((event: Event) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public readyState: number = WebSocket.CONNECTING;
  public url: string;
  
  constructor(url: string) {
    this.url = url;
    // Simulate connection after a short delay
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }
  
  send(_data: string): void {
    // Mock send implementation
  }
  
  close(): void {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

// Mock global WebSocket
(global as any).WebSocket = MockWebSocket;

// Import the main scene after mocking WebSocket
import { MainScene } from './main';

describe('MainScene', () => {
  let game: Phaser.Game;
  let scene: MainScene;
  
  beforeEach(() => {
    // Create a test game instance
    game = new Phaser.Game({
      type: Phaser.HEADLESS,
      width: 600,
      height: 600,
      physics: {
        default: 'matter',
        matter: {
          gravity: { x: 0, y: 1 },
          debug: false
        }
      },
      scene: MainScene
    });
    
    scene = game.scene.getScene('main') as MainScene;
  });
  
  afterEach(() => {
    game.destroy(true);
  });
  
  describe('Initialization', () => {
    test('should initialize with correct properties', () => {
      expect(scene).toBeDefined();
      expect(scene.balls).toBeDefined();
      expect(scene.platform).toBeDefined();
      expect(scene.messageBuffer).toBeDefined();
      expect(scene.maxBufferSize).toBe(3);
      expect(scene.interpolationDelay).toBe(50);
    });
    
    test('should create FPS and status text elements', () => {
      // The text elements should be created in the create method
      expect(scene.fpsText).toBeDefined();
      expect(scene.statusText).toBeDefined();
    });
  });
  
  describe('Ball Management', () => {
    test('should create balls with correct properties', () => {
      const initialBallCount = scene.balls.length;
      
      // Create a test ball
      const ball = scene.createBall(300, 100);
      
      expect(ball).toBeDefined();
      expect(scene.balls.length).toBe(initialBallCount + 1);
      expect(scene.balls).toContain(ball);
    });
    
    test('should clean up balls that fall off screen', () => {
      // Create a ball that's off screen
      const ball = scene.createBall(300, 700); // Below screen
      
      // Update the scene
      scene.update();
      
      // Ball should be removed
      expect(scene.balls).not.toContain(ball);
    });
  });
  
  describe('WebSocket Communication', () => {
    test('should initialize WebSocket connection', () => {
      expect(scene.ws).toBeDefined();
      expect(scene.ws?.url).toBe('ws://localhost:8765');
    });
    
    test('should handle WebSocket messages', () => {
      const testMessage = {
        timestamp: Date.now(),
        position: { x: 300, y: 200 },
        velocity: { x: 0, y: 10 },
        phase: 0.5,
        elapsed_time: 1.0
      };
      
      const initialBufferSize = scene.messageBuffer.length;
      
      // Simulate receiving a message
      if (scene.ws && scene.ws.onmessage) {
        const event = new MessageEvent('message', {
          data: JSON.stringify(testMessage)
        });
        scene.ws.onmessage(event);
      }
      
      // Message should be added to buffer
      expect(scene.messageBuffer.length).toBe(initialBufferSize + 1);
      expect(scene.messageBuffer[scene.messageBuffer.length - 1]).toEqual(testMessage);
    });
    
    test('should handle WebSocket connection open', () => {
      // Simulate connection open
      if (scene.ws && scene.ws.onopen) {
        scene.ws.onopen(new Event('open'));
      }
      
      // Retry timer should be reset
      expect(scene.connectionRetryTimer).toBe(0);
    });
    
    test('should handle WebSocket connection close', () => {
      // Simulate connection close
      if (scene.ws && scene.ws.onclose) {
        scene.ws.onclose(new CloseEvent('close'));
      }
      
      // WebSocket should be null and retry timer set
      expect(scene.ws).toBeNull();
      expect(scene.connectionRetryTimer).toBeGreaterThan(0);
    });
  });
  
  describe('Message Buffer Management', () => {
    test('should add messages to buffer', () => {
      const testMessage = {
        timestamp: Date.now(),
        position: { x: 300, y: 200 },
        velocity: { x: 0, y: 10 },
        phase: 0.5,
        elapsed_time: 1.0
      };
      
      scene.addMessageToBuffer(testMessage);
      
      expect(scene.messageBuffer).toContain(testMessage);
    });
    
    test('should limit buffer size', () => {
      // Add more messages than maxBufferSize
      for (let i = 0; i < 5; i++) {
        const message = {
          timestamp: Date.now() + i * 1000,
          position: { x: 300, y: 200 + i },
          velocity: { x: 0, y: 10 },
          phase: i * 0.1,
          elapsed_time: i
        };
        scene.addMessageToBuffer(message);
      }
      
      expect(scene.messageBuffer.length).toBeLessThanOrEqual(scene.maxBufferSize);
    });
  });
  
  describe('Interpolation', () => {
    test('should interpolate between messages', () => {
      const message1 = {
        timestamp: Date.now() - 100,
        position: { x: 300, y: 200 },
        velocity: { x: 0, y: 10 },
        phase: 0.0,
        elapsed_time: 0.0
      };
      
      const message2 = {
        timestamp: Date.now(),
        position: { x: 300, y: 300 },
        velocity: { x: 0, y: 10 },
        phase: 0.5,
        elapsed_time: 1.0
      };
      
      scene.messageBuffer = [message1, message2];
      
      const interpolated = scene.getInterpolatedPosition();
      
      expect(interpolated).toBeDefined();
      expect(interpolated?.x).toBe(300);
      expect(interpolated?.y).toBeGreaterThan(200);
      expect(interpolated?.y).toBeLessThan(300);
    });
    
    test('should return latest position when insufficient data', () => {
      const message = {
        timestamp: Date.now(),
        position: { x: 300, y: 250 },
        velocity: { x: 0, y: 10 },
        phase: 0.5,
        elapsed_time: 1.0
      };
      
      scene.messageBuffer = [message];
      
      const position = scene.getInterpolatedPosition();
      
      expect(position).toEqual(message.position);
    });
    
    test('should return null when no messages', () => {
      scene.messageBuffer = [];
      
      const position = scene.getInterpolatedPosition();
      
      expect(position).toBeNull();
    });
  });
  
  describe('FPS Monitoring', () => {
    test('should initialize FPS tracking variables', () => {
      expect(scene.messageCount).toBe(0);
      expect(scene.currentFps).toBe(0);
      expect(scene.lastFpsUpdateTime).toBeGreaterThan(0);
    });
    
    test('should update FPS display', () => {
      // Set up test data
      scene.messageCount = 60; // Simulate 60 messages received
      scene.lastFpsUpdateTime = Date.now() - 1000; // 1 second ago
      
      // Update FPS display
      scene.updateFpsDisplay();
      
      expect(scene.currentFps).toBe(60);
      expect(scene.messageCount).toBe(0); // Should be reset
    });
  });
  
  describe('Status Display', () => {
    test('should show connected status when WebSocket is open', () => {
      // Mock WebSocket as connected
      scene.ws = new MockWebSocket('ws://localhost:8765') as any;
      scene.connectionStartTime = Date.now() - 5000; // 5 seconds ago
      scene.messageBuffer = [{ timestamp: Date.now(), position: { x: 300, y: 200 }, velocity: { x: 0, y: 10 }, phase: 0, elapsed_time: 0 }];
      
      scene.updateStatusDisplay();
      
      // Status text should show connected
      expect(scene.statusText.text).toContain('Connected');
    });
    
    test('should show disconnected status when WebSocket is null', () => {
      scene.ws = null;
      
      scene.updateStatusDisplay();
      
      // Status text should show disconnected
      expect(scene.statusText.text).toContain('Disconnected');
    });
  });
  
  describe('Platform Movement', () => {
    test('should update platform position from WebSocket data', () => {
      const testMessage = {
        timestamp: Date.now(),
        position: { x: 300, y: 250 },
        velocity: { x: 0, y: 10 },
        phase: 0.5,
        elapsed_time: 1.0
      };
      
      scene.messageBuffer = [testMessage];
      
      // Mock getInterpolatedPosition to return the test position
      jest.spyOn(scene, 'getInterpolatedPosition').mockReturnValue(testMessage.position);
      
      scene.updatePlatformFromWebSocket();
      
      expect(scene.platform.x).toBe(300);
      expect(scene.platform.y).toBe(250);
    });
    
    test('should fallback to local calculation when no WebSocket data', () => {
      scene.messageBuffer = [];
      
      // Mock updatePlatformMovement
      const updatePlatformMovementSpy = jest.spyOn(scene, 'updatePlatformMovement');
      
      scene.updatePlatformFromWebSocket();
      
      expect(updatePlatformMovementSpy).toHaveBeenCalled();
    });
  });
});

// Performance tests
describe('Performance Tests', () => {
  let game: Phaser.Game;
  let scene: MainScene;
  
  beforeEach(() => {
    game = new Phaser.Game({
      type: Phaser.HEADLESS,
      width: 600,
      height: 600,
      physics: {
        default: 'matter',
        matter: {
          gravity: { x: 0, y: 1 },
          debug: false
        }
      },
      scene: MainScene
    });
    
    scene = game.scene.getScene('main') as MainScene;
  });
  
  afterEach(() => {
    game.destroy(true);
  });
  
  test('should handle high message frequency', () => {
    const startTime = performance.now();
    
    // Simulate receiving 100 messages quickly
    for (let i = 0; i < 100; i++) {
      const message = {
        timestamp: Date.now() + i * 16, // 60 FPS intervals
        position: { x: 300, y: 200 + Math.sin(i * 0.1) * 50 },
        velocity: { x: 0, y: Math.cos(i * 0.1) * 10 },
        phase: i * 0.1,
        elapsed_time: i * 0.016
      };
      scene.addMessageToBuffer(message);
    }
    
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    // Should handle 100 messages in less than 100ms
    expect(duration).toBeLessThan(100);
    expect(scene.messageBuffer.length).toBeLessThanOrEqual(scene.maxBufferSize);
  });
  
  test('should maintain smooth interpolation under load', () => {
    // Add multiple messages to buffer
    for (let i = 0; i < 10; i++) {
      const message = {
        timestamp: Date.now() - (10 - i) * 16,
        position: { x: 300, y: 200 + i * 10 },
        velocity: { x: 0, y: 10 },
        phase: i * 0.1,
        elapsed_time: i * 0.016
      };
      scene.addMessageToBuffer(message);
    }
    
    const startTime = performance.now();
    
    // Perform 1000 interpolation calculations
    for (let i = 0; i < 1000; i++) {
      scene.getInterpolatedPosition();
    }
    
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    // Should complete 1000 interpolations in less than 50ms
    expect(duration).toBeLessThan(50);
  });
});

// Integration tests
describe('Integration Tests', () => {
  test('should handle complete message flow', () => {
    const game = new Phaser.Game({
      type: Phaser.HEADLESS,
      width: 600,
      height: 600,
      physics: {
        default: 'matter',
        matter: {
          gravity: { x: 0, y: 1 },
          debug: false
        }
      },
      scene: MainScene
    });
    
    const scene = game.scene.getScene('main') as MainScene;
    
    // Simulate receiving a message
    const message = {
      timestamp: Date.now(),
      position: { x: 300, y: 250 },
      velocity: { x: 0, y: 10 },
      phase: 0.5,
      elapsed_time: 1.0
    };
    
    if (scene.ws && scene.ws.onmessage) {
      scene.ws.onmessage(new MessageEvent('message', {
        data: JSON.stringify(message)
      }));
    }
    
    // Update the scene
    scene.update();
    
    // Platform should be updated
    expect(scene.platform.x).toBe(300);
    expect(scene.platform.y).toBe(250);
    
    game.destroy(true);
  });
});
