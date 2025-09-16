#!/bin/bash

# Stop Game Script
# Kills all processes related to the dual-process game architecture

echo "🛑 Stopping Dual-Process Game Architecture"
echo "=========================================="

echo "🧹 Killing Python rectangle generator processes..."
pkill -f "rectangle_generator.py" 2>/dev/null || true

echo "🧹 Killing TypeScript/Vite processes..."
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

echo "🧹 Killing any remaining Node.js processes on ports 5173..."
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

echo "🧹 Killing any remaining Python processes on port 8765..."
lsof -ti:8765 | xargs kill -9 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 1

echo "✅ All processes stopped"
echo ""
echo "You can now run ./start_game.sh to restart the game"
