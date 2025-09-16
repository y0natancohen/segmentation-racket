#!/bin/bash

# Start Game Script
# Runs both the Python rectangle generator and the TypeScript Phaser game

echo "🎮 Starting Dual-Process Game Architecture"
echo "=========================================="

# Clean up any existing processes first
echo "🧹 Cleaning up existing processes..."
pkill -f "rectangle_generator.py" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed or not in PATH"
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed or not in PATH"
    exit 1
fi

echo "✅ Dependencies check passed"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down processes..."
    
    # Kill specific PIDs if they exist
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null
        echo "   - Python rectangle generator stopped (PID: $PYTHON_PID)"
    fi
    if [ ! -z "$NODE_PID" ]; then
        kill $NODE_PID 2>/dev/null
        echo "   - TypeScript game stopped (PID: $NODE_PID)"
    fi
    
    # Force kill any remaining processes
    echo "   - Cleaning up any remaining processes..."
    pkill -f "rectangle_generator.py" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    
    # Wait a moment for processes to terminate
    sleep 1
    
    echo "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo ""
echo "🚀 Starting Python rectangle generator..."

# Check if port 8765 is available
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8765 is still in use, force killing processes..."
    pkill -f "rectangle_generator.py" 2>/dev/null || true
    sleep 2
fi

cd /home/jonathan/segment_project
python3 rectangle_generator.py &
PYTHON_PID=$!

# Wait a moment for the Python server to start
sleep 2

echo "🎯 Starting TypeScript Phaser game..."

# Check if port 5173 is available
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 5173 is still in use, force killing processes..."
    pkill -f "vite" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    sleep 2
fi

cd /home/jonathan/segment_project/phaser-matter-game
npm run dev &
NODE_PID=$!

echo ""
echo "✅ Both processes started successfully!"
echo "   - Python rectangle generator (PID: $PYTHON_PID)"
echo "   - TypeScript game (PID: $NODE_PID)"
echo ""
echo "🌐 Game should be available at: http://localhost:5173"
echo "📡 Rectangle generator WebSocket: ws://localhost:8765"
echo ""
echo "Press Ctrl+C to stop both processes"

# Wait for either process to exit
wait $PYTHON_PID $NODE_PID
