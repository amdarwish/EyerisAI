#!/usr/bin/env python3
"""
FastAPI Backend Server for EyerisAI
Provides REST API endpoints for the three use cases:
1. Motion Detection (Restricted Area Access)
2. Trash Bin Status Analysis
3. Production Line Anomaly (Bottles)
"""

import os
import io
import base64
import tempfile
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import Body

import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Import your existing modules (support running from repo root or from within `backend/`)
try:
    from backend.EyerisAI import (  # type: ignore
        load_config,
        detect_motion,
        describe_frames,
        count_items,
        agent_decide_and_send,
        run_motion_detection,
        CONFIG,
    )
    from backend.my_agent import run_motion_agent, run_qa_agent  # type: ignore
except Exception:
    from EyerisAI import (  # type: ignore
        load_config,
        detect_motion,
        describe_frames,
        count_items,
        agent_decide_and_send,
        run_motion_detection,
        CONFIG,
    )
    from my_agent import run_motion_agent, run_qa_agent  # type: ignore

app = FastAPI(
    title="EyerisAI API",
    description="AI-powered motion detection, trash analysis, and production line monitoring",
    version="1.0.0",
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class MotionDetectionResponse(BaseModel):
    success: bool
    message: str
    alert_sent: bool = False  # Default to False
    description: Optional[str] = None
    timestamp: str
    image_path: Optional[str] = None


class VideoAnalysisRequest(BaseModel):
    use_case: str  # 'trash' or 'bottles'
    n_frames: int = 3
    interval: float = 10.0


class VideoAnalysisResponse(BaseModel):
    success: bool
    message: str
    analysis: str
    alert_sent: bool = False  # Default to False
    timestamp: str
    frames_analyzed: int


class ItemCountResponse(BaseModel):
    success: bool
    count: int
    item_name: str
    timestamp: str


class SystemStatusResponse(BaseModel):
    status: str
    config: Dict[str, Any]
    camera_available: bool


# Global variables for motion detection state
motion_detection_active = False
motion_detection_task = None
motion_stop_event: threading.Event | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("Starting EyerisAI API Server...")
    # Ensure save directory exists
    save_dir = Path(CONFIG["save_directory"])
    save_dir.mkdir(exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global motion_detection_active, motion_detection_task, motion_stop_event
    if motion_detection_active and motion_detection_task:
        motion_detection_active = False
        if motion_stop_event is not None:
            motion_stop_event.set()
        motion_detection_task.cancel()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "EyerisAI API Server", "version": "1.0.0", "status": "running"}


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "GET /status",
            "POST /motion/detect",
            "POST /video/analyze",
            "POST /count/items",
        ],
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and configuration"""
    try:
        # Test camera availability
        camera_available = False
        try:
            if CONFIG["camera"].get("use_ip", False):
                source = CONFIG["camera"]["ip_url"]
            else:
                source = CONFIG["camera"]["device_id"]

            cap = cv2.VideoCapture(source)
            camera_available = cap.isOpened()
            if cap.isOpened():
                cap.release()
        except Exception as e:
            print(f"Camera test failed: {e}")

        return SystemStatusResponse(
            status="online",
            config={
                "ai_model": CONFIG["ai"]["model"],
                "motion_detection_active": motion_detection_active,
                "email_enabled": CONFIG["email"]["enabled"],
                "tts_enabled": CONFIG["tts"]["enabled"],
            },
            camera_available=camera_available,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get system status: {str(e)}"
        )


@app.post("/motion/start")
async def start_motion_detection(background_tasks: BackgroundTasks):
    """Start continuous motion detection"""
    global motion_detection_active, motion_detection_task, motion_stop_event

    # Print the equivalent terminal command
    print("\n" + "="*60)
    print("🚀 STARTING MOTION DETECTION")
    print("="*60)
    print("📋 Equivalent Terminal Command:")
    print("   python EyerisAI.py --use-case motion")
    print("📍 Backend API Endpoint: POST /motion/start")
    print("="*60 + "\n")

    if motion_detection_active:
        return JSONResponse(
            {"success": False, "message": "Motion detection is already running"}
        )

    try:
        motion_detection_active = True
        motion_stop_event = threading.Event()

        # Start motion detection in background
        async def motion_detection_worker():
            try:
                # This would run your existing motion detection loop
                # For now, we'll use a simplified version
                await asyncio.to_thread(run_motion_detection, "", motion_stop_event)
            except Exception as e:
                print(f"Motion detection error: {e}")
                global motion_detection_active
                motion_detection_active = False

        motion_detection_task = asyncio.create_task(motion_detection_worker())

        return JSONResponse(
            {
                "success": True,
                "message": "Motion detection started",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to start motion detection: {str(e)}\n")
        motion_detection_active = False
        raise HTTPException(
            status_code=500, detail=f"Failed to start motion detection: {str(e)}"
        )


@app.post("/motion/stop")
async def stop_motion_detection():
    """Stop continuous motion detection"""
    global motion_detection_active, motion_detection_task, motion_stop_event

    # Print the equivalent terminal command
    print("\n" + "="*60)
    print("🛑 STOPPING MOTION DETECTION")
    print("="*60)
    print("📋 Equivalent Terminal Command:")
    print("   Ctrl+C (to stop python EyerisAI.py --use-case motion)")
    print("📍 Backend API Endpoint: POST /motion/stop")
    print("="*60 + "\n")

    if not motion_detection_active:
        return JSONResponse(
            {"success": False, "message": "Motion detection is not running"}
        )

    try:
        motion_detection_active = False
        if motion_stop_event is not None:
            motion_stop_event.set()
        if motion_detection_task:
            motion_detection_task.cancel()

        return JSONResponse(
            {
                "success": True,
                "message": "Motion detection stopped",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop motion detection: {str(e)}"
        )


@app.post("/motion/detect", response_model=MotionDetectionResponse)
async def detect_motion_single(file: UploadFile = File(...)):
    """Detect motion in a single uploaded image/frame"""
    try:
        # Print the equivalent terminal command
        print("\n" + "="*60)
        print("🔍 ANALYZING SINGLE IMAGE FOR MOTION")
        print("="*60)
        print(f"📋 Equivalent Terminal Command:")
        print(f"   python EyerisAI.py --use-case single-image --image '{file.filename}'")
        print(f"📍 Backend API Endpoint: POST /motion/detect")
        print("="*60 + "\n")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read uploaded file
        contents = await file.read()

        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # For single frame, we'll use AI description instead of motion detection
        # Convert frame to JPEG bytes for AI analysis
        success, img_bytes = cv2.imencode(".jpg", frame)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")

        # Generate description using AI
        description = ""
        alert_sent = False

        if CONFIG["ai_description"]:
            try:
                description = describe_frames(
                    [img_bytes.tobytes()], CONFIG["ai"]["prompt"]
                )

                # Use agent to decide if alert should be sent
                timestamp = datetime.now().isoformat()
                save_path = (
                    Path(CONFIG["save_directory"])
                    / f"motion_{timestamp.replace(':', '-')}.jpg"
                )
                cv2.imwrite(str(save_path), frame)

                try:
                    alert_sent = await asyncio.to_thread(
                        agent_decide_and_send,
                        description,
                        [img_bytes.tobytes()],
                        str(save_path),
                        timestamp,
                        "motion",
                    )
                except Exception as e:
                    print(f"Agent decision error (continuing without alert): {e}")
                    alert_sent = False

            except Exception as e:
                description = f"AI analysis failed: {str(e)}"

        return MotionDetectionResponse(
            success=True,
            message="Motion detection completed",
            alert_sent=bool(alert_sent),  # Ensure boolean
            description=description,
            timestamp=datetime.now().isoformat(),
            image_path=str(save_path) if "save_path" in locals() else None,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Motion detection failed: {str(e)}"
        )


@app.post("/video/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: Optional[UploadFile] = File(None),
    use_case: Optional[str] = Form(None),
    n_frames: int = Form(3),
    interval: float = Form(10.0),
    json_payload: Optional[VideoAnalysisRequest] = Body(None),
):
    """
    Analyze uploaded video for trash bin status or production line anomalies.
    Supports BOTH multipart/form-data and JSON payloads.
    """

    try:
        # ----------------------------
        # Normalize inputs
        # ----------------------------
        if json_payload:
            use_case = json_payload.use_case
            n_frames = json_payload.n_frames
            interval = json_payload.interval

        if not use_case:
            raise HTTPException(
                status_code=400,
                detail="use_case is required (trash or bottles)",
            )

        if use_case not in ["trash", "bottles"]:
            raise HTTPException(
                status_code=400,
                detail="use_case must be 'trash' or 'bottles'",
            )

        if file is None:
            raise HTTPException(
                status_code=400,
                detail="Video file is required for analysis",
            )

        if not file.content_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be a video",
            )

        # ----------------------------
        # Save uploaded video
        # ----------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            video_path = tmp_file.name

        print(
            f"Video analysis started | use_case={use_case}, "
            f"frames={n_frames}, interval={interval}s"
        )

        try:
            # ----------------------------
            # Extract frames
            # ----------------------------
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Could not open uploaded video",
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
            duration = total_frames / fps

            frames = []

            for i in range(n_frames):
                t = min(i * interval, duration)
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    ok, jpg = cv2.imencode(".jpg", frame)
                    if ok:
                        frames.append(jpg.tobytes())

            cap.release()

            if not frames:
                raise HTTPException(
                    status_code=400,
                    detail="No frames could be extracted from video",
                )

            # ----------------------------
            # Prompts
            # ----------------------------
            if use_case == "trash":
                prompt = (
                    "Analyze the frames and determine if pins are piling up "
                    "into a tall consolidated mass (issue) or remain scattered "
                    "at normal levels. Return JSON only."
                )
            else:
                prompt = (
                    "Detect whether any bottle appears in front of the horizontal "
                    "separator (anomaly) or all bottles remain behind it (normal). "
                    "Return JSON only."
                )

            # ----------------------------
            # AI Analysis
            # ----------------------------
            analysis = await asyncio.to_thread(
                describe_frames, frames, prompt
            )

            timestamp = datetime.now().isoformat()
            save_path = (
                Path(CONFIG["save_directory"])
                / f"{use_case}_{timestamp.replace(':', '-')}.jpg"
            )

            with open(save_path, "wb") as f:
                f.write(frames[0])

            alert_sent = False
            try:
                alert_sent = await asyncio.to_thread(
                    agent_decide_and_send,
                    analysis,
                    frames,
                    str(save_path),
                    timestamp,
                    "qa",
                )
            except Exception as e:
                print(f"Agent error (non-blocking): {e}")

            return VideoAnalysisResponse(
                success=True,
                message=f"Video analysis completed for {use_case}",
                analysis=analysis,
                alert_sent=bool(alert_sent),
                timestamp=timestamp,
                frames_analyzed=len(frames),
            )

        finally:
            try:
                os.unlink(video_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}",
        )

@app.post("/count/items", response_model=ItemCountResponse)
async def count_items_endpoint(
    file: UploadFile = File(...), item_name: str = Form(...)
):
    """Count items in an uploaded image"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        if not item_name.strip():
            raise HTTPException(status_code=400, detail="item_name is required")

        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            image_path = tmp_file.name

        try:
            # Count items using existing function
            result = await asyncio.to_thread(count_items, image_path, item_name.strip())

            # Extract count from result (assuming it returns a dict with 'count')
            if isinstance(result, dict) and "count" in result:
                count = result["count"]
            elif isinstance(result, int):
                count = result
            else:
                count = 0

            return ItemCountResponse(
                success=True,
                count=count,
                item_name=item_name.strip(),
                timestamp=datetime.now().isoformat(),
            )

        finally:
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Item counting failed: {str(e)}")


@app.get("/logs")
async def get_logs(limit: int = 100):
    """Get recent log entries"""
    try:
        log_file = Path(CONFIG["save_directory"]) / CONFIG["log_file"]
        if not log_file.exists():
            return {"logs": [], "message": "No log file found"}

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    import json

                    logs.append(json.loads(line.strip()))
                except:
                    continue

        return {"logs": logs, "count": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


if __name__ == "__main__":
    print("Starting EyerisAI API Server...")
    print("Available endpoints:")
    print("  - GET  /status - System status")
    print("  - POST /motion/start - Start motion detection")
    print("  - POST /motion/stop - Stop motion detection")
    print("  - POST /motion/detect - Analyze single image for motion")
    print("  - POST /video/analyze - Analyze video (trash/bottles)")
    print("  - POST /count/items - Count items in image")
    print("  - GET  /logs - Get recent logs")
    print("\nServer will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")

    uvicorn.run(
        "api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
