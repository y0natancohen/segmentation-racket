#!/bin/bash

# Run Python tests
echo "Running Python tests..."
cd server
python -m pytest tests/ -v

# Run E2E tests (requires both server and frontend running)
echo "To run E2E tests:"
echo "1. Start server: ./run_server.sh"
echo "2. Start frontend: cd web-app && npm run dev"
echo "3. Run E2E: cd web-app && npm run e2e"
