#!/bin/bash
# Demo script for different polygon configurations

echo "🎮 Polygon System Demo"
echo "====================="
echo ""
echo "This script demonstrates the polygon system with different shapes."
echo "Each demo will run for 10 seconds, then switch to the next shape."
echo ""

# Function to run a polygon demo
run_polygon_demo() {
    local config_file=$1
    local shape_name=$2
    
    echo "🔷 Starting $shape_name demo..."
    echo "   Config: $config_file"
    echo "   Game: http://localhost:5173"
    echo ""
    
    # Start the polygon generator
    python3 polygon_generator.py "$config_file" &
    local python_pid=$!
    
    # Start the game
    cd phaser-matter-game && npm run dev &
    local node_pid=$!
    cd ..
    
    # Wait for services to start
    sleep 3
    
    echo "✅ $shape_name demo is running!"
    echo "   Press Ctrl+C to stop and move to next shape"
    echo ""
    
    # Wait for 10 seconds or until interrupted
    sleep 10
    
    # Clean up
    kill $python_pid 2>/dev/null
    kill $node_pid 2>/dev/null
    pkill -f "polygon_generator.py" 2>/dev/null
    pkill -f "npm run dev" 2>/dev/null
    pkill -f "vite" 2>/dev/null
    
    sleep 2
    echo "🛑 $shape_name demo stopped"
    echo ""
}

# Run demos for each polygon type
run_polygon_demo "polygon_config/rectangle.json" "Rectangle"
run_polygon_demo "polygon_config/triangle.json" "Triangle" 
run_polygon_demo "polygon_config/octagon.json" "Octagon"

echo "🎉 All polygon demos completed!"
echo ""
echo "To run a specific polygon:"
echo "  python3 polygon_generator.py polygon_config/rectangle.json"
echo "  python3 polygon_generator.py polygon_config/triangle.json"
echo "  python3 polygon_generator.py polygon_config/octagon.json"
echo ""
echo "Then start the game with:"
echo "  cd phaser-matter-game && npm run dev"
