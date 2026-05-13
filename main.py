"""
Exam Proctoring API
===================
FastAPI backend that wraps proctoring_system.py and exposes:

  POST /register          – register a student with 1-3 photos
  POST /register/base64   – register with base64-encoded photos
  POST /session/start     – start a proctoring session
  POST /session/analyze   – send a webcam frame for analysis
  POST /session/end       – end session & get final report
  GET  /session/{id}      – get live session state
  GET  /registry          – list all registered students
  DELETE /registry/{name} – remove a student
  GET  /health            – system health check
"""

import base64
import io
import os
import sys
import time
import uuid
import warnings
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

# ── Silence noisy libs ───────────────────────────────────────────────────────
os.environ.update({
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "MEDIAPIPE_DISABLE_GPU": "1",
    "GLOG_minloglevel": "3",
})
warnings.filterwarnings("ignore")

# Configure logging for the API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExamProctoringAPI")

# Suppress only external library logging, not our own
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Import your proctoring logic ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from proctoring_system import (
    register_person,
    identify_person_from_frame,
    identify_person_from_frame_bytes,
    GazeTracker,
    RuleEngine,
    Thresholds,
    _registry_names,
    _registry_cache,
    NumpyEncoder,
    REGISTRY_PATH,
    _rebuild_registry_tensor,
)

import json

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Exam Proctoring API",
    description="Real-time exam integrity monitoring with face recognition, gaze tracking, and rule engine.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Only mount static files if directory exists and has files
if os.path.exists(static_dir) and os.listdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── In-memory session store ───────────────────────────────────────────────────
sessions: Dict[str, Dict[str, Any]] = {}

# Session cleanup configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Pydantic models ───────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    student_name: str = Field(..., min_length=1, max_length=100)
    session_id: Optional[str] = Field(None, max_length=100)


class AnalyzeFrameRequest(BaseModel):
    session_id: str
    frame_b64: str  # base64-encoded JPEG


class RegisterBase64Request(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    photos: List[str] = Field(..., min_items=1, max_items=3)


class EndSessionRequest(BaseModel):
    session_id: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def b64_to_frame(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string into a BGR numpy frame."""
    try:
        # Handle both with and without data URI prefix
        if "," in b64:
            header, _, data = b64.partition(",")
            raw = base64.b64decode(data)
        else:
            raw = base64.b64decode(b64)
        
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        return frame
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


def b64_to_pil(b64: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    try:
        if "," in b64:
            header, _, data = b64.partition(",")
            raw = base64.b64decode(data)
        else:
            raw = base64.b64decode(b64)
        
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


def pil_to_temp_path(img: Image.Image, name: str) -> str:
    """Save PIL image to temporary path."""
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    path = os.path.join(UPLOAD_DIR, f"{safe_name}_{uuid.uuid4().hex[:8]}.jpg")
    img.save(path, "JPEG", quality=95)
    return path


def cleanup_old_sessions():
    """Remove sessions that have timed out."""
    current_time = datetime.now()
    expired_sessions = []
    
    for sid, sess in sessions.items():
        started_at = datetime.fromisoformat(sess["started_at"])
        if (current_time - started_at).total_seconds() > SESSION_TIMEOUT:
            expired_sessions.append(sid)
    
    for sid in expired_sessions:
        try:
            if "rule_engine" in sessions[sid]:
                sessions[sid]["rule_engine"].close()
            del sessions[sid]
        except Exception:
            pass
    
    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect to static frontend."""
    return RedirectResponse("/static/index.html")


@app.get("/health")
async def health():
    """System health check."""
    cleanup_old_sessions()
    return {
        "status": "ok",
        "registered_students": len(_registry_cache),
        "active_sessions": len(sessions),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/registry")
async def list_registry():
    """Return all registered student names."""
    return {
        "students": sorted(list(_registry_cache.keys())),
        "count": len(_registry_cache)
    }


@app.delete("/registry/{name}")
async def delete_from_registry(name: str):
    """Remove a student from the registry."""
    if name not in _registry_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Student '{name}' not found in registry"
        )
    
    # Remove from cache
    del _registry_cache[name]
    
    # Persist to disk
    try:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(_registry_cache, f)
        _rebuild_registry_tensor()
        logger.info(f"Deleted student '{name}' from registry")
    except Exception as e:
        logger.error(f"Failed to persist registry after deletion: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update registry file"
        )
    
    return {
        "status": "deleted",
        "name": name,
        "remaining_students": len(_registry_cache)
    }


@app.post("/register")
async def register_student(
    name: str = Form(..., min_length=1, max_length=100),
    photos: List[UploadFile] = File(..., min_length=1, max_length=3),
):
    """
    Register a student with 1–3 face photos via multipart form upload.
    Returns success/failure and the number of usable faces found.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if len(photos) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 photos allowed")
    if len(photos) < 1:
        raise HTTPException(status_code=400, detail="Minimum 1 photo required")

    saved_paths = []
    try:
        for photo in photos:
            raw = await photo.read()
            if len(raw) == 0:
                raise HTTPException(status_code=400, detail=f"Empty file: {photo.filename}")
            
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            path = pil_to_temp_path(img, name)
            saved_paths.append(path)

        result = register_person(name, saved_paths)

        if result.get("status") != "success":
            raise HTTPException(
                status_code=422,
                detail=f"Registration failed: {result.get('reason', 'unknown')}"
            )

        logger.info(f"Registered student '{name}' with {len(saved_paths)} photos")
        
        return {
            "status": "registered",
            "name": name,
            "photos_processed": len(saved_paths),
            "message": f"✓ {name} registered successfully",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        # Cleanup temp files
        for p in saved_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


@app.post("/register/base64")
async def register_student_b64(payload: RegisterBase64Request):
    """Register a student from base64-encoded photos (for browser webcam capture)."""
    name = payload.name.strip()
    photos_b64 = payload.photos

    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    if not photos_b64:
        raise HTTPException(status_code=400, detail="At least one photo required")
    if len(photos_b64) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 photos allowed")

    saved_paths = []
    try:
        for b64 in photos_b64:
            img = b64_to_pil(b64)
            path = pil_to_temp_path(img, name)
            saved_paths.append(path)

        result = register_person(name, saved_paths)

        if result.get("status") != "success":
            raise HTTPException(
                status_code=422,
                detail=f"Registration failed: {result.get('reason', 'unknown')}"
            )

        logger.info(f"Registered student '{name}' via base64 with {len(saved_paths)} photos")
        
        return {
            "status": "registered",
            "name": name,
            "photos_processed": len(saved_paths),
            "message": f"✓ {name} registered successfully",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        # Cleanup temp files
        for p in saved_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


@app.post("/session/start")
async def start_session(req: StartSessionRequest):
    """
    Start a new proctoring session for a registered student.
    Returns a session_id used for subsequent /analyze calls.
    """
    # Clean up old sessions
    cleanup_old_sessions()
    
    if req.student_name not in _registry_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Student '{req.student_name}' not found in registry. Please register first.",
        )

    sid = req.session_id or str(uuid.uuid4())
    
    # Check if session already exists
    if sid in sessions:
        raise HTTPException(
            status_code=409,
            detail=f"Session '{sid}' already exists. Use a different session_id."
        )
    
    sessions[sid] = {
        "session_id": sid,
        "student_name": req.student_name,
        "started_at": datetime.now().isoformat(),
        "frame_count": 0,
        "rule_engine": RuleEngine(student_id=req.student_name),
        "gaze_tracker": GazeTracker(),
        "last_result": None,
        "alerts": [],
        "is_flagged": False,
        "violation_score": 0,
        "last_activity": datetime.now().isoformat(),
    }

    logger.info(f"Session started: {sid} for student {req.student_name}")
    
    return {
        "session_id": sid,
        "student_name": req.student_name,
        "started_at": sessions[sid]["started_at"],
        "message": f"Session started for {req.student_name}",
    }


@app.post("/session/analyze")
async def analyze_frame(req: AnalyzeFrameRequest):
    """
    Analyze a single webcam frame.
    Returns identity confidence, gaze status, and any rule violations.
    """
    sid = req.session_id
    if sid not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")

    sess = sessions[sid]
    
    # Update last activity
    sess["last_activity"] = datetime.now().isoformat()
    sess["frame_count"] += 1

    try:
        frame = b64_to_frame(req.frame_b64)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image data: {str(e)}")

    # Validate frame dimensions
    if frame.shape[0] < 60 or frame.shape[1] < 60:
        raise HTTPException(status_code=422, detail="Image too small. Minimum 60x60 pixels required.")

    try:
        # ── Identity check ────────────────────────────────────────────────────
        identity = identify_person_from_frame(frame)
        face_count = 0

        if "best_match" in identity:
            face_count = 1
            # Note: For production, implement proper multi-face detection
            # using MTCNN or YOLO face detection here
        elif "error" in identity:
            face_count = 0
            identity = {"best_match": "Unknown", "confidence": 0.0}

        # ── Gaze tracking ─────────────────────────────────────────────────────
        gaze = sess["gaze_tracker"].process(frame)

        # ── Build record for rule engine ──────────────────────────────────────
        record = {
            "occurrence": sess["frame_count"],
            "timestamp": datetime.now().isoformat(),
            "face_count": face_count,
            "identity": identity,
            "audio": {"detected": False, "level": 0.0},  # audio skipped in API mode
            "unauthorized_objects": {"detected": False, "count": 0, "items": []},
            "gaze": {
                "left_eye": {
                    "direction": gaze["left_eye"]["direction"],
                    "dx": gaze["left_eye"]["dx"],
                    "dy": gaze["left_eye"]["dy"]
                },
                "right_eye": {
                    "direction": gaze["right_eye"]["direction"],
                    "dx": gaze["right_eye"]["dx"],
                    "dy": gaze["right_eye"]["dy"]
                },
                "looking_away": gaze["looking_away"],
            },
        }

        # ── Evaluate rules ────────────────────────────────────────────────────
        rule_result = sess["rule_engine"].evaluate(record)
        sess["last_result"] = rule_result
        sess["is_flagged"] = rule_result["exam_flagged"]
        sess["violation_score"] = rule_result["total_score"]
        if rule_result["alerts_this_frame"]:
            sess["alerts"].extend(rule_result["alerts_this_frame"])

        # ── Determine overall status ──────────────────────────────────────────
        is_correct_person = (
            identity.get("best_match") == sess["student_name"]
            and identity.get("confidence", 0) >= Thresholds.IDENTITY_MIN_CONFIDENCE
        )

        status = "ok"
        if rule_result["exam_flagged"]:
            status = "flagged"
        elif rule_result["alerts_this_frame"]:
            status = "warning"

        return {
            "session_id": sid,
            "frame": sess["frame_count"],
            "status": status,
            "is_correct_person": is_correct_person,
            "identity": {
                "best_match": identity.get("best_match", "Unknown"),
                "confidence": round(identity.get("confidence", 0.0), 1),
                "expected": sess["student_name"],
            },
            "gaze": {
                "looking_away": gaze["looking_away"],
                "left": {
                    "h": gaze["left_eye"]["direction"]["horizontal"],
                    "v": gaze["left_eye"]["direction"]["vertical"]
                },
                "right": {
                    "h": gaze["right_eye"]["direction"]["horizontal"],
                    "v": gaze["right_eye"]["direction"]["vertical"]
                },
            },
            "face_count": face_count,
            "violation_score": rule_result["total_score"],
            "exam_flagged": rule_result["exam_flagged"],
            "alerts_this_frame": rule_result["alerts_this_frame"],
        }
        
    except Exception as e:
        logger.error(f"Frame analysis error for session {sid}: {e}")
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get current state of a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sess = sessions[session_id]
    
    # Check if session expired
    started_at = datetime.fromisoformat(sess["started_at"])
    if (datetime.now() - started_at).total_seconds() > SESSION_TIMEOUT:
        raise HTTPException(status_code=410, detail="Session expired")
    
    return {
        "session_id": session_id,
        "student_name": sess["student_name"],
        "started_at": sess["started_at"],
        "last_activity": sess.get("last_activity", sess["started_at"]),
        "frame_count": sess["frame_count"],
        "is_flagged": sess["is_flagged"],
        "violation_score": sess["violation_score"],
        "total_alerts": len(sess["alerts"]),
        "status": "active" if not sess["is_flagged"] else "flagged",
    }


@app.post("/session/end")
async def end_session(payload: EndSessionRequest):
    """End a session and return the final proctoring report."""
    sid = payload.session_id
    if not sid or sid not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        sess = sessions[sid]
        engine = sess["rule_engine"]
        summary = engine.summary()
        
        # Generate report
        report = {
            "session_id": sid,
            "student_name": sess["student_name"],
            "started_at": sess["started_at"],
            "ended_at": datetime.now().isoformat(),
            "frames_analyzed": sess["frame_count"],
            "total_alerts": summary["total_alerts"],
            "violation_score": summary["violation_score"],
            "exam_flagged": summary["exam_flagged"],
            "alert_breakdown": summary["breakdown"],
            "verdict": "SUSPICIOUS" if summary["exam_flagged"] else "CLEAN",
            "alerts": sess["alerts"][-20:],  # Last 20 alerts for context
        }
        
        # Clean up
        engine.close()
        
        # Clean up gaze tracker
        if "gaze_tracker" in sess:
            try:
                sess["gaze_tracker"].cleanup()
            except Exception:
                pass
        
        del sessions[sid]
        
        logger.info(f"Session ended: {sid} - Verdict: {report['verdict']}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error ending session {sid}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


# ── Error handlers ───────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500},
    )


# ── Startup/Shutdown events ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info(f"🚀 Exam Proctoring API started")
    # logger.info(f"📊 Device: {device}")
    logger.info(f"👥 Registered students: {len(_registry_cache)}")
    logger.info(f"📁 Registry path: {REGISTRY_PATH}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")
    
    # Close all active sessions
    for sid, sess in sessions.items():
        try:
            if "rule_engine" in sess:
                sess["rule_engine"].close()
            if "gaze_tracker" in sess:
                sess["gaze_tracker"].cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up session {sid}: {e}")
    
    sessions.clear()
    logger.info("API shutdown complete")