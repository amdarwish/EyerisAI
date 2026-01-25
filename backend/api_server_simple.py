#!/usr/bin/env python3
"""
Simplified EyerisAI API Server
A minimal version that works without complex dependencies
"""

import os
import io
import json
import base64
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as e:
    print(f"Error importing FastAPI dependencies: {e}")
    print("Please install: pip install fastapi uvicorn python-multipart")
    exit(1)

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not available. Video processing will be limited.")
    OPENCV_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: Requests not available. Some features may be limited.")
    REQUESTS_AVAILABLE = False

app = FastAPI(
    title="EyerisAI API (Simplified)",
    description="Simplified AI-powered motion detection and analysis",
    version="1.0.0-simple"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple response models
class SimpleResponse(BaseModel):
    success: bool
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None

class SystemStatus(BaseModel):
    status: str
    opencv_available: bool
    requests_available: bool
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EyerisAI API Server (Simplified)", 
        "version": "1.0.0-simple",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get system status"""
    return SystemStatus(
        status="online",
        opencv_available=OPENCV_AVAILABLE,
        requests_available=REQUESTS_AVAILABLE,
        timestamp=datetime.now().isoformat()
    )

@app.post("/motion/start")
async def start_motion_detection():
    """Start motion detection (simplified)"""
    return SimpleResponse(
        success=True,
        message="Motion detection started (simplified mode)",
        timestamp=datetime.now().isoformat(),
        data={"mode": "simplified", "note": "This is a minimal implementation"}
    )

@app.post("/motion/stop")
async def stop_motion_detection():
    """Stop motion detection"""
    return SimpleResponse(
        success=True,
        message="Motion detection stopped",
        timestamp=datetime.now().isoformat()
    )

@app.post("/motion/detect")
async def detect_motion_single(file: UploadFile = File(...)):
    """Detect motion in uploaded image (simplified)"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        
        # Basic analysis without complex AI
        analysis = f"Received image: {file.filename}, size: {len(contents)} bytes"
        
        if OPENCV_AVAILABLE:
            # Try to process image with OpenCV
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                height, width = frame.shape[:2]
                analysis += f", dimensions: {width}x{height}"
            else:
                analysis += ", could not decode image"
        
        return SimpleResponse(
            success=True,
            message="Motion detection completed (simplified)",
            timestamp=datetime.now().isoformat(),
            data={
                "analysis": analysis,
                "filename": file.filename,
                "size": len(contents),
                "alert_sent": False
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Motion detection failed: {str(e)}")

@app.post("/video/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    use_case: str = Form(...),
    n_frames: int = Form(3),
    interval: float = Form(10.0),
    custom_prompt: Optional[str] = Form(None)
):
    """Analyze uploaded video (simplified)"""
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        if use_case not in ['trash', 'bottles']:
            raise HTTPException(status_code=400, detail="use_case must be 'trash' or 'bottles'")
        
        contents = await file.read()
        
        analysis = f"Video analysis for {use_case} use case"
        analysis += f"\nReceived: {file.filename}, size: {len(contents)} bytes"
        analysis += f"\nParameters: {n_frames} frames, {interval}s interval"
        
        if custom_prompt:
            analysis += f"\nCustom prompt: {custom_prompt}"
        
        if OPENCV_AVAILABLE:
            # Save and try to analyze video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(contents)
                video_path = tmp_file.name
            
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    
                    analysis += f"\nVideo info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration"
                    
                    cap.release()
                else:
                    analysis += "\nCould not open video file"
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
        
        return SimpleResponse(
            success=True,
            message=f"Video analysis completed for {use_case}",
            timestamp=datetime.now().isoformat(),
            data={
                "analysis": analysis,
                "use_case": use_case,
                "frames_analyzed": n_frames,
                "alert_sent": False
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

@app.post("/count/items")
async def count_items_endpoint(
    file: UploadFile = File(...),
    item_name: str = Form(...)
):
    """Count items in image (simplified)"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        
        # Simplified counting (random number for demo)
        import random
        count = random.randint(0, 10)
        
        return SimpleResponse(
            success=True,
            message=f"Item counting completed for '{item_name}'",
            timestamp=datetime.now().isoformat(),
            data={
                "count": count,
                "item_name": item_name,
                "note": "This is a simplified implementation"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Item counting failed: {str(e)}")

@app.get("/logs")
async def get_logs():
    """Get logs (simplified)"""
    return {
        "logs": [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "Simplified API server is running"
            }
        ],
        "count": 1
    }

if __name__ == "__main__":
    print("Starting EyerisAI Simplified API Server...")
    print("Available endpoints:")
    print("  - GET  / - Welcome message")
    print("  - GET  /status - System status")
    print("  - POST /motion/start - Start motion detection") 
    print("  - POST /motion/stop - Stop motion detection")
    print("  - POST /motion/detect - Analyze single image")
    print("  - POST /video/analyze - Analyze video")
    print("  - POST /count/items - Count items in image")
    print("  - GET  /logs - Get logs")
    print("\nServer will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server_simple:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )