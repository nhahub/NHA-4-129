import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from ultralytics import YOLO

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"]      = "3"
os.environ["PYTHONWARNINGS"]        = "ignore"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


class _Devnull:
    def write(self, *_): pass
    def flush(self):     pass


#CONFIGURATION & THRESHOLDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VERIFY_THRESHOLD  = 0.70
IDENTITY_MIN_CONF = 45.0
REGISTRY_PATH     = "face_registry.json"
UPLOAD_DIR        = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

#Set up logging to file + console
LOG_DIR = "proctoring_logs"
os.makedirs(LOG_DIR, exist_ok=True)
_log_file = os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_sh = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
_fh.setFormatter(_fmt); _sh.setFormatter(_fmt)
_root_logger.addHandler(_fh); _root_logger.addHandler(_sh)
logging.disable(logging.NOTSET)   # re-enable only our logger
logger = logging.getLogger("ProctorSystem")


class Thresholds:
    IDENTITY_MIN_CONFIDENCE = 45.0
    MAX_FACES               = 1
    GAZE_DX_THRESHOLD       = 0.80
    GAZE_DY_THRESHOLD       = 0.55
    GAZE_AWAY_CONSECUTIVE   = 2      
    AUDIO_LEVEL_THRESHOLD   = 0.15
    OBJECT_MIN_CONFIDENCE   = 0.40
    SUSPICIOUS_SCORE_LIMIT  = 10   

    WEIGHTS = {
        "low_identity_confidence": 3,
        "multiple_faces":          5,
        "no_face":                 2,
        "gaze_away":               2,
        "unauthorized_object":     4,
        "suspicious_audio":        3,
    }


#RULE ENGINE
@dataclass
class Alert:
    rule_name  : str
    severity   : str    
    message    : str
    timestamp  : str
    student_id : str
    score      : int = 0

    def to_dict(self) -> dict:
        return {
            "rule"      : self.rule_name,
            "severity"  : self.severity,
            "message"   : self.message,
            "timestamp" : self.timestamp,
            "student_id": self.student_id,
            "score"     : self.score,
        }


@dataclass
class SessionState:
    student_id       : str
    violation_score  : int  = 0
    is_flagged       : bool = False
    gaze_away_streak : int  = 0
    alerts           : list = field(default_factory=list)


def rule_identity_confidence(record: dict, state: SessionState) -> Optional[Alert]:
    identity   = record.get("identity", {})
    best_match = identity.get("best_match", "Unknown")
    confidence = identity.get("confidence", 0.0)
    if record.get("face_count", 0) > 0 and best_match == "Unknown":
        msg = (
            f"Face detected but identity is UNKNOWN "
            f"(confidence {confidence:.1f}% < {Thresholds.IDENTITY_MIN_CONFIDENCE}%). "
            f"Possible impersonator or student not registered."
        )
        return Alert(
            rule_name  = "low_identity_confidence",
            severity   = "CRITICAL",
            message    = msg,
            timestamp  = record.get("timestamp", ""),
            student_id = state.student_id,
            score      = Thresholds.WEIGHTS["low_identity_confidence"],
        )
    return None


def rule_multiple_faces(record: dict, state: SessionState) -> Optional[Alert]:
    face_count = record.get("face_count", 0)
    if face_count > Thresholds.MAX_FACES:
        msg = (
            f"Multiple faces detected: {face_count} faces in frame "
            f"(allowed: {Thresholds.MAX_FACES}). Possible external assistance."
        )
        return Alert(
            rule_name  = "multiple_faces",
            severity   = "CRITICAL",
            message    = msg,
            timestamp  = record.get("timestamp", ""),
            student_id = state.student_id,
            score      = Thresholds.WEIGHTS["multiple_faces"],
        )
    return None


def rule_no_face(record: dict, state: SessionState) -> Optional[Alert]:
    if record.get("face_count", 1) == 0:
        msg = (
            "No face detected in frame (~5 s window). "
            "Student may have left the seat or covered the camera."
        )
        return Alert(
            rule_name  = "no_face",
            severity   = "WARNING",
            message    = msg,
            timestamp  = record.get("timestamp", ""),
            student_id = state.student_id,
            score      = Thresholds.WEIGHTS["no_face"],
        )
    return None


def rule_gaze_away(record: dict, state: SessionState) -> Optional[Alert]:
    gaze      = record.get("gaze", {})
    flag_away = gaze.get("looking_away", False)

    left_dx  = abs(gaze.get("left_eye",  {}).get("dx", 0.0))
    left_dy  = abs(gaze.get("left_eye",  {}).get("dy", 0.0))
    right_dx = abs(gaze.get("right_eye", {}).get("dx", 0.0))
    right_dy = abs(gaze.get("right_eye", {}).get("dy", 0.0))

    numeric_away = (
        (left_dx  >= Thresholds.GAZE_DX_THRESHOLD or
        left_dy  >= Thresholds.GAZE_DY_THRESHOLD)
        and
        (right_dx >= Thresholds.GAZE_DX_THRESHOLD or
        right_dy >= Thresholds.GAZE_DY_THRESHOLD)
    )

    if flag_away or numeric_away:
        state.gaze_away_streak += 1
    else:
        state.gaze_away_streak = 0

    if state.gaze_away_streak >= Thresholds.GAZE_AWAY_CONSECUTIVE:
        msg = (
            f"Sustained gaze away from screen: "
            f"{state.gaze_away_streak} consecutive occurrences "
            f"(approx. {state.gaze_away_streak * 5} s). "
            f"Left eye  dx={gaze.get('left_eye',{}).get('dx',0):.2f} "
            f"dy={gaze.get('left_eye',{}).get('dy',0):.2f} | "
            f"Right eye dx={gaze.get('right_eye',{}).get('dx',0):.2f} "
            f"dy={gaze.get('right_eye',{}).get('dy',0):.2f}."
        )
        return Alert(
            rule_name  = "gaze_away",
            severity   = "WARNING",
            message    = msg,
            timestamp  = record.get("timestamp", ""),
            student_id = state.student_id,
            score      = Thresholds.WEIGHTS["gaze_away"],
        )
    return None


def rule_unauthorized_object(record: dict, state: SessionState) -> Optional[Alert]:
    obj_data = record.get("unauthorized_objects", {})
    if obj_data.get("detected", False):
        items     = obj_data.get("items", [])
        item_strs = ", ".join(
            f"{i['type']} ({i['confidence']*100:.0f}%)" for i in items
        )
        msg = (
            f"Unauthorized object(s) detected: {item_strs}. "
            f"Total count in frame: {obj_data.get('count', 0)}."
        )
        return Alert(
            rule_name  = "unauthorized_object",
            severity   = "CRITICAL",
            message    = msg,
            timestamp  = record.get("timestamp", ""),
            student_id = state.student_id,
            score      = Thresholds.WEIGHTS["unauthorized_object"],
        )
    return None


def rule_suspicious_audio(record: dict, state: SessionState) -> Optional[Alert]:
    audio      = record.get("audio", {})
    is_talking = audio.get("detected", False)
    level      = audio.get("level", 0.0)
    if is_talking or level >= Thresholds.AUDIO_LEVEL_THRESHOLD:
        msg = (
            f"Suspicious audio detected: level={level:.3f} "
            f"(threshold={Thresholds.AUDIO_LEVEL_THRESHOLD}), "
            f"is_talking={is_talking}. "
            f"Student may be speaking or receiving verbal assistance."
        )
        return Alert(
            rule_name  = "suspicious_audio",
            severity   = "WARNING",
            message    = msg,
            timestamp  = record.get("timestamp", ""),
            student_id = state.student_id,
            score      = Thresholds.WEIGHTS["suspicious_audio"],
        )
    return None


RULES = [
    rule_identity_confidence,
    rule_multiple_faces,
    rule_no_face,
    rule_gaze_away,
    rule_unauthorized_object,
    rule_suspicious_audio,
]


class RuleEngine:
    """Evaluates proctoring records against all rules.
    Can be used in real-time (one record at a time) or offline (full session file)."""

    def __init__(self, student_id: str = "Unknown"):
        self.state = SessionState(student_id=student_id)
        logger.info(
            f"RuleEngine started | student={student_id} | "
            f"suspicious_limit={Thresholds.SUSPICIOUS_SCORE_LIMIT}"
        )

    def evaluate(self, record: dict) -> dict:
        """Run all rules against one record. Returns a per-frame summary."""
        triggered = []

        for rule_fn in RULES:
            alert = rule_fn(record, self.state)
            if alert is None:
                continue

            self.state.violation_score += alert.score
            self.state.alerts.append(alert)
            triggered.append(alert)

            log_fn = logger.critical if alert.severity == "CRITICAL" else logger.warning
            log_fn(
                f"[{alert.rule_name.upper()}] student={alert.student_id} | "
                f"+{alert.score}pts (total={self.state.violation_score}) | "
                f"{alert.message}"
            )

        if (not self.state.is_flagged and
                self.state.violation_score >= Thresholds.SUSPICIOUS_SCORE_LIMIT):
            self.state.is_flagged = True
            logger.critical(
                f"*** EXAM FLAGGED AS SUSPICIOUS *** | "
                f"student={self.state.student_id} | "
                f"score={self.state.violation_score} >= "
                f"limit={Thresholds.SUSPICIOUS_SCORE_LIMIT}"
            )

        return {
            "occurrence"       : record.get("occurrence"),
            "timestamp"        : record.get("timestamp"),
            "student_id"       : self.state.student_id,
            "alerts_this_frame": [a.to_dict() for a in triggered],
            "total_score"      : self.state.violation_score,
            "exam_flagged"     : self.state.is_flagged,
        }

    def evaluate_session_file(self, json_path: str) -> dict:
        """Load a saved proctoring_data_*.json and run all rules over it."""
        with open(json_path, "r") as f:
            data = json.load(f)

        detections  = data.get("detections", [])
        all_results = [self.evaluate(r) for r in detections]

        report = {
            "student_id"       : self.state.student_id,
            "session_file"     : json_path,
            "total_occurrences": len(detections),
            "total_alerts"     : len(self.state.alerts),
            "violation_score"  : self.state.violation_score,
            "exam_flagged"     : self.state.is_flagged,
            "alert_breakdown"  : self._alert_breakdown(),
            "per_occurrence"   : all_results,
        }

        report_path = json_path.replace(".json", "_rule_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Session report saved → {report_path}")
        return report

    def _alert_breakdown(self) -> dict:
        bd = {}
        for alert in self.state.alerts:
            bd[alert.rule_name] = bd.get(alert.rule_name, 0) + 1
        return bd

    def summary(self) -> dict:
        return {
            "student_id"     : self.state.student_id,
            "total_alerts"   : len(self.state.alerts),
            "violation_score": self.state.violation_score,
            "exam_flagged"   : self.state.is_flagged,
            "breakdown"      : self._alert_breakdown(),
        }

    def close(self):
        logger.info(f"Session ended | {self.summary()}")


#NUMPY JSON ENCODER

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


#ARCFACE LOSS & FACE MODEL

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=32.0, m=0.35):
        super().__init__()
        self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine  = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine    = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-9, 1.0))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return F.cross_entropy(output * self.s, labels)


class FaceModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone  = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.PReLU(), nn.Dropout(0.3),
        )
        self.arcface = ArcFaceLoss(256, num_classes)

    def get_embeddings(self, x):
        return F.normalize(self.projector(self.backbone(x)), dim=1)

    def forward(self, x, labels=None):
        emb = self.get_embeddings(x)
        if labels is not None:
            return self.arcface(emb, labels)
        return F.linear(emb, F.normalize(self.arcface.weight))


#Load model
import __main__
__main__.FaceModel   = FaceModel
__main__.ArcFaceLoss = ArcFaceLoss

_stderr_orig = sys.stderr
sys.stderr   = _Devnull()
try:
    face_recog_model = torch.load(
        "face_model_full.pth", map_location=device, weights_only=False
    )
except AttributeError:
    backbone         = InceptionResnetV1(pretrained="vggface2", classify=False)
    face_recog_model = FaceModel(backbone, num_classes=15)
    state_dict       = torch.load("face_model_full.pth", map_location=device)
    face_recog_model.load_state_dict(state_dict)
finally:
    sys.stderr = _stderr_orig

face_recog_model = face_recog_model.to(device)
face_recog_model.eval()

mtcnn = MTCNN(
    image_size=160, margin=28, min_face_size=40,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    keep_all=False, device=device, post_process=True,
)

_registry_cache: dict = {}


def _load_registry() -> dict:
    global _registry_cache
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            _registry_cache = json.load(f)
    return _registry_cache


_load_registry()
_registry_names : list                = []
_registry_tensor: torch.Tensor | None = None


def _rebuild_registry_tensor():
    global _registry_names, _registry_tensor
    if not _registry_cache:
        _registry_names = []; _registry_tensor = None; return
    _registry_names  = list(_registry_cache.keys())
    _registry_tensor = torch.stack(
        [torch.tensor(v) for v in _registry_cache.values()]
    )


_rebuild_registry_tensor()


#FACE RECOGNITION HELPERS

def _embed(pil_img: Image.Image):
    face = mtcnn(pil_img)
    if face is None:
        return None
    with torch.no_grad():
        emb = face_recog_model.get_embeddings(face.unsqueeze(0).to(device))
    return emb.squeeze(0).cpu()


def get_embedding_from_path(img_path: str):
    if not os.path.exists(img_path): return None
    return _embed(Image.open(img_path).convert("RGB"))


def get_embedding_from_frame(frame_bgr: np.ndarray):
    return _embed(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))


def register_person(name: str, img_paths: list) -> dict:
    embeddings = [e for p in img_paths if (e := get_embedding_from_path(p)) is not None]
    if not embeddings:
        return {"status": "failed", "reason": "no faces detected"}
    mean_emb = F.normalize(torch.stack(embeddings).mean(0), dim=0)
    _registry_cache[name] = mean_emb.tolist()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(_registry_cache, f)
    _rebuild_registry_tensor()
    return {"status": "success", "name": name}


def verify_faces(img1: str, img2: str) -> dict:
    e1 = get_embedding_from_path(img1)
    e2 = get_embedding_from_path(img2)
    if e1 is None or e2 is None:
        return {"error": "face not detected"}
    sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    return {"similarity": round(sim, 4), "verdict": "SAME" if sim >= VERIFY_THRESHOLD else "DIFFERENT"}


def _match_embedding(emb: torch.Tensor) -> dict:
    if _registry_tensor is None or not _registry_names:
        return {"error": "empty registry"}
    sims      = F.cosine_similarity(
        emb.unsqueeze(0).expand(_registry_tensor.size(0), -1), _registry_tensor,
    )
    best_idx  = int(sims.argmax())
    best_name = _registry_names[best_idx]
    conf      = round(float(sims[best_idx]) * 100, 2)
    return {
        "best_match": best_name if conf >= IDENTITY_MIN_CONF else "Unknown",
        "confidence": conf,
        "all": [
            {"name": _registry_names[i], "sim": round(float(sims[i]), 4)}
            for i in sims.argsort(descending=True).tolist()
        ],
    }


def identify_person_from_path(img_path: str) -> dict:
    emb = get_embedding_from_path(img_path)
    return {"error": "no face detected"} if emb is None else _match_embedding(emb)


def identify_person_from_frame(frame_bgr: np.ndarray) -> dict:
    emb = get_embedding_from_frame(frame_bgr)
    return {"error": "no face detected"} if emb is None else _match_embedding(emb)

#GAZE TRACKER

class GazeTracker:
    H_THRESH = 0.12
    V_THRESH = 0.10

    def __init__(self):
        _s = sys.stderr; sys.stderr = _Devnull()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
        )
        sys.stderr = _s
        self.LEFT_IRIS  = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE   = [33,  133, 160, 144]
        self.RIGHT_EYE  = [362, 263, 387, 373]
        _eye = {"direction": {"horizontal": "CENTER", "vertical": "CENTER"}, "dx": 0.0, "dy": 0.0}
        self.current_gaze   = {"left_eye": dict(_eye), "right_eye": dict(_eye), "looking_away": False}
        self.smooth_left    = {"dx": 0.0, "dy": 0.0}
        self.smooth_right   = {"dx": 0.0, "dy": 0.0}
        self.smoothing      = 0.5
        self.frame_counter  = 0
        self.process_every_n = 2

    def _gaze_dir(self, iris_center, eye_lms, fw, fh, lm):
        l = np.array([lm[eye_lms[0]].x * fw, lm[eye_lms[0]].y * fh])
        r = np.array([lm[eye_lms[1]].x * fw, lm[eye_lms[1]].y * fh])
        t = np.array([lm[eye_lms[2]].x * fw, lm[eye_lms[2]].y * fh])
        b = np.array([lm[eye_lms[3]].x * fw, lm[eye_lms[3]].y * fh])
        eye_w = np.linalg.norm(r - l)
        eye_h = np.linalg.norm(b - t)
        if eye_w < 1e-6: return "CENTER", "CENTER", 0.0, 0.0
        center = (l + r) / 2.0
        dx = float(np.clip((iris_center[0] - center[0]) / eye_w, -1.0, 1.0))
        dy = float(np.clip((iris_center[1] - center[1]) / eye_h, -1.0, 1.0))
        h = "LEFT" if dx < -self.H_THRESH else "RIGHT" if dx > self.H_THRESH else "CENTER"
        v = "UP"   if dy < -self.V_THRESH else "DOWN"  if dy > self.V_THRESH else "CENTER"
        return h, v, dx, dy

    def _process_eye(self, iris_ids, eye_ids, smooth, fw, fh, lm):
        pts    = np.array([(int(lm[i].x * fw), int(lm[i].y * fh)) for i in iris_ids])
        center = pts.mean(axis=0).astype(int)
        h, v, dx, dy = self._gaze_dir(center, eye_ids, fw, fh, lm)
        a = 1.0 - self.smoothing
        smooth["dx"] = smooth["dx"] * self.smoothing + dx * a
        smooth["dy"] = smooth["dy"] * self.smoothing + dy * a
        return {"direction": {"horizontal": h, "vertical": v},
                "dx": round(smooth["dx"], 3), "dy": round(smooth["dy"], 3)}

    def process(self, frame: np.ndarray) -> dict:
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n != 0:
            return self.current_gaze
        h, w = frame.shape[:2]
        res  = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try: self.current_gaze["left_eye"]  = self._process_eye(self.LEFT_IRIS,  self.LEFT_EYE,  self.smooth_left,  w, h, lm)
            except Exception: pass
            try: self.current_gaze["right_eye"] = self._process_eye(self.RIGHT_IRIS, self.RIGHT_EYE, self.smooth_right, w, h, lm)
            except Exception: pass
            lc = (self.current_gaze["left_eye"]["direction"]["horizontal"]  == "CENTER" and
                self.current_gaze["left_eye"]["direction"]["vertical"]    == "CENTER")
            rc = (self.current_gaze["right_eye"]["direction"]["horizontal"] == "CENTER" and
                self.current_gaze["right_eye"]["direction"]["vertical"]   == "CENTER")
            self.current_gaze["looking_away"] = not (lc and rc)
        else:
            self.current_gaze["looking_away"] = True
        return self.current_gaze

#AUDIO MONITOR

class AudioMonitor:
    def __init__(self, threshold=0.15):
        self.threshold   = threshold
        self.audio_level = 0.0
        self.is_talking  = False
        self.stream      = None

    def _callback(self, indata, frames, time_info, status):
        vol              = float(np.linalg.norm(indata) * 10)
        self.audio_level = vol
        self.is_talking  = vol > self.threshold

    def start(self):
        self.stream = sd.InputStream(callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream: self.stream.stop(); self.stream.close()


#OBJECT DETECTOR

class ObjectDetector:
    UNAUTHORIZED = {
        63: "Laptop", 64: "Laptop",
        67: "Cell Phone", 77: "Cell Phone",
        73: "Book", 74: "Book", 84: "Book",
    }
    CONF = 0.40

    def detect(self, model, frame: np.ndarray) -> list:
        results    = model(frame, classes=list(self.UNAUTHORIZED),
                        conf=self.CONF, iou=0.45, verbose=False, max_det=10)
        detections = []
        for res in results:
            if res.boxes is None: continue
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < self.CONF: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cid = int(box.cls[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2), "class_id": cid,
                    "confidence": conf,
                    "name": self.UNAUTHORIZED.get(cid, "Unauthorized Object"),
                })
        return detections

#  integrated RuleEngine

class ProctoringSystem:
    def __init__(self):
        self.object_model    = None
        self.yolo_face_model = None
        self.gaze_tracker    = None
        self.audio_monitor   = None
        self.object_detector = None
        self.cap             = None
        self.running         = True

        self.frame_width         = 640
        self.frame_height        = 480
        self.object_detect_every = 2
        self.save_every_n_frames = 30   # one record every N frames
        self.identify_every_n    = 45

        # Sensor state
        self.frame_count          = 0
        self.fps                  = 0
        self._fps_counter         = 0
        self._fps_time            = time.time()
        self.unauthorized_objects = []
        self.face_count           = 0
        self.current_identity     = {"best_match": "Unknown", "confidence": 0.0}

        # Output buffer
        self.saved_detections = []
        self.occurrence_count = 0

        # Rule engine
        self._rule_engine: Optional[RuleEngine] = None

    # Rule engine accessor 

    def _get_rule_engine(self) -> RuleEngine:
        """Return (or lazily create) the RuleEngine, syncing student id."""
        sid = self.current_identity.get("best_match", "Unknown")
        if self._rule_engine is None:
            self._rule_engine = RuleEngine(student_id=sid)
        elif self._rule_engine.state.student_id != sid and sid != "Unknown":
            # Identity became known mid-session — update without resetting score
            self._rule_engine.state.student_id = sid
        return self._rule_engine

    # Live camera overlay

    def _draw_rule_overlay(self, frame: np.ndarray, rule_result: dict) -> np.ndarray:
        """Render real-time rule feedback on the camera window."""
        score   = rule_result.get("total_score", 0)
        flagged = rule_result.get("exam_flagged", False)
        alerts  = rule_result.get("alerts_this_frame", [])
        h, w    = frame.shape[:2]

        # Colour-coded risk score bar along the bottom
        bar_color = (0, 200, 0)
        if score >= Thresholds.SUSPICIOUS_SCORE_LIMIT // 2:
            bar_color = (0, 165, 255)
        if flagged:
            bar_color = (0, 0, 255)

        bar_w = int(w * min(score / Thresholds.SUSPICIOUS_SCORE_LIMIT, 1.0))
        cv2.rectangle(frame, (0, h - 18), (bar_w, h), bar_color, -1)
        cv2.putText(frame, f"Risk: {score}/{Thresholds.SUSPICIOUS_SCORE_LIMIT}",
                    (6, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        #Large FLAGGED banner
        if flagged:
            cv2.putText(frame, "*** EXAM FLAGGED ***",
                        (w // 2 - 160, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        #Active alerts top-right
        for idx, a in enumerate(alerts[:4]):
            color = (0, 0, 255) if a["severity"] == "CRITICAL" else (0, 165, 255)
            cv2.putText(frame, f"[{a['severity']}] {a['rule']}",
                        (w - 340, 30 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        return frame

    # Initialise hardware & models

    def initialize(self):
        _s = sys.stderr; sys.stderr = _Devnull()
        try:
            self.object_model = YOLO("yolov8n.pt")
        except Exception as e:
            sys.stderr = _s; raise RuntimeError(f"Object model load failed: {e}")
        try:
            self.yolo_face_model = YOLO("yolov8n-face.pt")
        except Exception:
            self.yolo_face_model = None
        sys.stderr = _s

        self.object_detector = ObjectDetector()
        self.gaze_tracker    = GazeTracker()
        self.audio_monitor   = AudioMonitor(threshold=0.15)
        self.audio_monitor.start()

        for idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened(): break
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Could not open any webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print({
            "event":            "system_ready",
            "device":           str(device),
            "resolution":       f"{self.frame_width}x{self.frame_height}",
            "registry_entries": len(_registry_names),
            "controls":         {"q": "quit + auto-save", "s": "save now"},
        })

    #Per-frame detection pipeline

    def _run_detections(self, frame: np.ndarray) -> np.ndarray:
        self.face_count = 0
        if self.yolo_face_model:
            for res in self.yolo_face_model(frame, conf=0.5, verbose=False):
                if res.boxes is None: continue
                for box in res.boxes:
                    self.face_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if self.frame_count % self.object_detect_every == 0:
            self.unauthorized_objects = self.object_detector.detect(self.object_model, frame)
        for obj in self.unauthorized_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return frame

    def _run_recognition(self, frame: np.ndarray) -> np.ndarray:
        if self.frame_count % self.identify_every_n == 0:
            result = identify_person_from_frame(frame)
            if "best_match" in result:
                self.current_identity = result
        label = (f"{self.current_identity['best_match']} "
                f"({self.current_identity['confidence']:.1f}%)")
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    # Save helpers

    def save_detections(self, filename: str = None) -> str:
        if filename is None:
            filename = f"proctoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w") as f:
            json.dump(
                {"total_occurrences": self.occurrence_count,
                "detections":        self.saved_detections},
                f, indent=2, cls=NumpyEncoder,
            )
        logger.info(f"Detection data saved → {filename} ({self.occurrence_count} occurrences)")

        if self._rule_engine:
            report = {
                "student_id"       : self._rule_engine.state.student_id,
                "session_file"     : filename,
                "total_occurrences": self.occurrence_count,
                "total_alerts"     : len(self._rule_engine.state.alerts),
                "violation_score"  : self._rule_engine.state.violation_score,
                "exam_flagged"     : self._rule_engine.state.is_flagged,
                "alert_breakdown"  : self._rule_engine._alert_breakdown(),
            }
            report_path = filename.replace(".json", "_rule_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Rule report saved → {report_path}")
            self._print_summary(report)

        return filename

    def _print_summary(self, report: dict):
        print("\n" + "=" * 55)
        print("  SESSION SUMMARY")
        print("=" * 55)
        print(f"  Student         : {report['student_id']}")
        print(f"  Occurrences     : {report['total_occurrences']}")
        print(f"  Total Alerts    : {report['total_alerts']}")
        print(f"  Violation Score : {report['violation_score']}")
        print(f"  Exam Flagged    : {report['exam_flagged']}")
        print(f"  Breakdown       : {report['alert_breakdown']}")
        print("=" * 55)

    #Main loop

    def run(self):
        last_rule_result = {"total_score": 0, "exam_flagged": False, "alerts_this_frame": []}

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01); continue

            self.frame_count += 1
            frame     = self._run_detections(frame)
            frame     = self._run_recognition(frame)
            gaze_data = self.gaze_tracker.process(frame)

            # FPS counter
            now = time.time()
            if now - self._fps_time >= 1.0:
                self.fps          = self.frame_count - self._fps_counter
                self._fps_counter = self.frame_count
                self._fps_time    = now

            # Build detection record
            record = {
                "occurrence": self.occurrence_count + 1,
                "timestamp":  datetime.now().isoformat(),
                "fps":        self.fps,
                "face_count": self.face_count,
                "identity": {
                    "best_match": self.current_identity.get("best_match", "Unknown"),
                    "confidence": self.current_identity.get("confidence", 0.0),
                },
                "audio": {
                    "detected": self.audio_monitor.is_talking,
                    "level":    round(self.audio_monitor.audio_level, 3),
                },
                "unauthorized_objects": {
                    "detected": bool(self.unauthorized_objects),
                    "count":    len(self.unauthorized_objects),
                    "items": [
                        {"type": o["name"], "confidence": round(o["confidence"], 2)}
                        for o in self.unauthorized_objects
                    ],
                },
                "gaze": {
                    "left_eye": {
                        "direction": gaze_data["left_eye"]["direction"],
                        "dx":        gaze_data["left_eye"]["dx"],
                        "dy":        gaze_data["left_eye"]["dy"],
                    },
                    "right_eye": {
                        "direction": gaze_data["right_eye"]["direction"],
                        "dx":        gaze_data["right_eye"]["dx"],
                        "dy":        gaze_data["right_eye"]["dy"],
                    },
                    "looking_away": gaze_data["looking_away"],
                },
            }

            #Save record + evaluate rules every N frames
            if self.frame_count % self.save_every_n_frames == 0:
                self.saved_detections.append(record)
                self.occurrence_count += 1

                engine           = self._get_rule_engine()
                last_rule_result = engine.evaluate(record)

                # Print combined output
                print({**record, "rule_result": last_rule_result})

            # Always render the latest rule state as an overlay
            frame = self._draw_rule_overlay(frame, last_rule_result)

            cv2.imshow("Proctoring System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if self.saved_detections:
                    self.save_detections()
                self.running = False
            elif key == ord("s"):
                self.save_detections()

    def cleanup(self):
        if self._rule_engine:
            self._rule_engine.close()
        if self.audio_monitor:
            self.audio_monitor.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

#ENTRY POINT

if __name__ == "__main__":

    #REGISTRATION MODE
    if len(sys.argv) >= 4 and sys.argv[1].lower() == "register":
        name      = sys.argv[2]
        img_paths = sys.argv[3:]
        result    = register_person(name, img_paths)
        print(result)
        sys.exit(0)

    # OFFLINE EVALUATION MODE
    if len(sys.argv) == 3 and sys.argv[1].lower() == "evaluate":
        path = sys.argv[2]
        if not os.path.exists(path):
            print(f"File not found: {path}"); sys.exit(1)

        with open(path) as f:
            raw = json.load(f)

        sid    = (raw.get("detections", [{}])[0]
                    .get("identity", {})
                    .get("best_match", "Unknown"))
        engine = RuleEngine(student_id=sid)
        report = engine.evaluate_session_file(path)
        engine.close()

        print("\n" + "=" * 55)
        print("  SESSION SUMMARY")
        print("=" * 55)
        print(f"  Student         : {report['student_id']}")
        print(f"  Occurrences     : {report['total_occurrences']}")
        print(f"  Total Alerts    : {report['total_alerts']}")
        print(f"  Violation Score : {report['violation_score']}")
        print(f"  Exam Flagged    : {report['exam_flagged']}")
        print(f"  Breakdown       : {report['alert_breakdown']}")
        print("=" * 55)
        sys.exit(0)

    #LIVE MONITORING MODE
    system = None
    try:
        system = ProctoringSystem()
        system.initialize()
        system.run()
    except KeyboardInterrupt:
        if system and system.saved_detections:
            system.save_detections()
    except Exception as exc:
        print({"event": "fatal_error", "error": str(exc)})
        import traceback; traceback.print_exc()
    finally:
        if system:
            system.cleanup()
