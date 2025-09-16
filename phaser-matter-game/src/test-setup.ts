/**
 * Test setup file for Jest
 * Configures the testing environment for Phaser and WebSocket testing.
 */

// Mock Phaser for testing
jest.mock('phaser', () => {
  const mockPhaser = {
    Game: jest.fn().mockImplementation(() => ({
      scene: {
        getScene: jest.fn().mockReturnValue({
          balls: [],
          platform: { x: 300, y: 300, setPosition: jest.fn() },
          messageBuffer: [],
          maxBufferSize: 3,
          interpolationDelay: 50,
          fpsText: { text: '', setText: jest.fn(), setStyle: jest.fn() },
          statusText: { text: '', setText: jest.fn(), setStyle: jest.fn() },
          messageCount: 0,
          currentFps: 0,
          lastFpsUpdateTime: Date.now(),
          connectionStartTime: Date.now(),
          ws: null,
          connectionRetryTimer: 0,
          connectionRetryDelay: 2000,
          add: {
            text: jest.fn().mockReturnValue({
              setScrollFactor: jest.fn(),
              setDepth: jest.fn(),
              text: '',
              setText: jest.fn(),
              setStyle: jest.fn()
            }),
            graphics: jest.fn().mockReturnValue({
              fillStyle: jest.fn().mockReturnThis(),
              fillCircle: jest.fn().mockReturnThis(),
              generateTexture: jest.fn().mockReturnThis(),
              clear: jest.fn().mockReturnThis()
            })
          },
          matter: {
            add: {
              image: jest.fn().mockReturnValue({
                setCircle: jest.fn(),
                setBounce: jest.fn(),
                setFriction: jest.fn(),
                setFrictionStatic: jest.fn(),
                setFrictionAir: jest.fn(),
                setStatic: jest.fn(),
                setPosition: jest.fn(),
                x: 300,
                y: 300,
                body: { angularDamping: 0 }
              }),
              rectangle: jest.fn().mockReturnValue({
                setBounce: jest.fn(),
                setFriction: jest.fn(),
                setFrictionStatic: jest.fn()
              })
            },
            world: {
              setBounds: jest.fn(),
              setGravity: jest.fn(),
              walls: {
                top: { restitution: 0, friction: 0 },
                bottom: { restitution: 0, friction: 0 },
                left: { restitution: 0, friction: 0 },
                right: { restitution: 0, friction: 0 }
              },
              engine: {
                positionIterations: 8,
                velocityIterations: 6,
                constraintIterations: 4
              }
            }
          },
          cameras: {
            main: {
              setBackgroundColor: jest.fn()
            }
          },
          scale: {
            resize: jest.fn()
          },
          Math: {
            Between: jest.fn().mockReturnValue(0)
          },
          time: {
            now: Date.now(),
            addEvent: jest.fn().mockReturnValue({
              delay: 1000,
              callback: jest.fn(),
              loop: true
            })
          },
          filter: jest.fn().mockReturnValue([]),
          destroy: jest.fn(),
          update: jest.fn(),
          create: jest.fn(),
          preload: jest.fn(),
          createBall: jest.fn(),
          updatePlatformFromWebSocket: jest.fn(),
          handleConnectionRetry: jest.fn(),
          updateFpsDisplay: jest.fn(),
          updateStatusDisplay: jest.fn(),
          addMessageToBuffer: jest.fn(),
          getInterpolatedPosition: jest.fn(),
          updatePlatformMovement: jest.fn(),
          initializeWebSocketConnection: jest.fn(),
          shutdown: jest.fn()
        })
      },
      destroy: jest.fn()
    })),
    Scene: jest.fn(),
    AUTO: 'AUTO',
    HEADLESS: 'HEADLESS',
    Physics: {
      Matter: {
        Image: jest.fn(),
        World: jest.fn()
      }
    },
    GameObjects: {
      Graphics: jest.fn(),
      Text: jest.fn()
    },
    Math: {
      Between: jest.fn().mockReturnValue(0)
    },
    Time: {
      TimerEvent: jest.fn()
    }
  };
  
  return mockPhaser;
});

// Mock WebSocket
global.WebSocket = jest.fn().mockImplementation(() => ({
  onopen: null,
  onmessage: null,
  onclose: null,
  onerror: null,
  readyState: WebSocket.CONNECTING,
  send: jest.fn(),
  close: jest.fn(),
  url: 'ws://localhost:8765'
})) as any;

// Mock performance API
global.performance = {
  now: jest.fn().mockReturnValue(Date.now())
} as any;

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  log: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn()
};
