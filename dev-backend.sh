#!/bin/bash

# Development script for EyerisAI Backend
# This script runs just the backend with auto-reload for development

echo "🔧 Starting EyerisAI Backend Development Server..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing/updating Python dependencies..."
pip install -r backend/requirements-core.txt

# Start backend server with auto-reload
echo "🚀 Starting FastAPI server with auto-reload..."
echo "📍 API Server: http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo "📍 API Status: http://localhost:8000/status"
echo ""
echo "Press Ctrl+C to stop the server"

uvicorn backend.api_server:app --reload --host 0.0.0.0 --port 8000