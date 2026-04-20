# HOW TO USE:
#   STEP 1 — Register a person (run once per person, with 2-5 clear photos):
#     python proctoring_system.py register "Sherifa" photo1.jpg photo2.jpg photo3.jpg
#
#   STEP 2 — Start the proctoring system:
#     python proctoring_system.py
import os
import sys
import json
import math
import logging
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import sounddevice as sd
from datetime import datetime
import time
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1, MTCNN


os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"]      = "3"
os.environ["PYTHONWARNINGS"]        = "ignore"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


class _Devnull:
    def write(self, *_): pass
    def flush(self):     pass


#CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VERIFY_THRESHOLD  = 0.70   #cosine similarity threshold for verification
IDENTITY_MIN_CONF = 45.0   #% — below this confidence → report "Unknown"
REGISTRY_PATH     = "face_registry.json"
UPLOAD_DIR        = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


#JSON ENCODER
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


#ARCFACE LOSS
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=32.0, m=0.35):
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine  = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine    = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-9, 1.0))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)


# FACE MODEL
class FaceModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone  = backbone
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.3),
        )
        self.arcface = ArcFaceLoss(256, num_classes)

    def get_embeddings(self, x):
        emb = self.backbone(x)
        emb = self.projector(emb)
        return F.normalize(emb, dim=1)

    def forward(self, x, labels=None):
        emb = self.get_embeddings(x)
        if labels is not None:
            return self.arcface(emb, labels)
        return F.linear(emb, F.normalize(self.arcface.weight))


# LOAD FACE RECOGNITION MODEL
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

#tuned for better crop quality
mtcnn = MTCNN(
    image_size=160,
    margin=28,                        #wider margin → more face context
    min_face_size=40,                 #ignore tiny/distant detections
    thresholds=[0.6, 0.7, 0.7],      #P/R/O-net thresholds
    factor=0.709,
    keep_all=False,
    device=device,
    post_process=True,                #normalise pixel values for stability
)

#Registry cache + pre-computed tensor for fast batch similarity
_registry_cache: dict = {}


def _load_registry() -> dict:
    global _registry_cache
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            _registry_cache = json.load(f)
    return _registry_cache


_load_registry()

_registry_names:  list= []
_registry_tensor: torch.Tensor | None = None


def _rebuild_registry_tensor():
    global _registry_names, _registry_tensor
    if not _registry_cache:
        _registry_names  = []
        _registry_tensor = None
        return
    _registry_names  = list(_registry_cache.keys())
    _registry_tensor = torch.stack(
        [torch.tensor(v) for v in _registry_cache.values()]
    )  #shape (N, 256)


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
    if not os.path.exists(img_path):
        return None
    return _embed(Image.open(img_path).convert("RGB"))


def get_embedding_from_frame(frame_bgr: np.ndarray):
    return _embed(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))


def register_person(name: str, img_paths: list) -> dict:
    embeddings = [e for p in img_paths
                if (e := get_embedding_from_path(p)) is not None]
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
    return {
        "similarity": round(sim, 4),
        "verdict":    "SAME" if sim >= VERIFY_THRESHOLD else "DIFFERENT",
    }


def _match_embedding(emb: torch.Tensor) -> dict:
    if _registry_tensor is None or not _registry_names:
        return {"error": "empty registry"}

    sims      = F.cosine_similarity(
        emb.unsqueeze(0).expand(_registry_tensor.size(0), -1),
        _registry_tensor,
    )
    best_idx  = int(sims.argmax())
    best_sim  = float(sims[best_idx])
    best_name = _registry_names[best_idx]
    conf      = round(best_sim * 100, 2)

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
    if emb is None:
        return {"error": "no face detected"}
    return _match_embedding(emb)


def identify_person_from_frame(frame_bgr: np.ndarray) -> dict:
    emb = get_embedding_from_frame(frame_bgr)
    if emb is None:
        return {"error": "no face detected"}
    return _match_embedding(emb)


#GAZE TRACKER
class GazeTracker:
    H_THRESH = 0.12 
    V_THRESH = 0.10   

    def __init__(self):
        _s = sys.stderr; sys.stderr = _Devnull()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        sys.stderr = _s

        self.LEFT_IRIS  = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE   = [33,  133, 160, 144]
        self.RIGHT_EYE  = [362, 263, 387, 373]

        _eye = {"direction": {"horizontal": "CENTER", "vertical": "CENTER"},
                "dx": 0.0, "dy": 0.0}
        self.current_gaze = {
            "left_eye":    dict(_eye),
            "right_eye":   dict(_eye),
            "looking_away": False,
        }
        self.smooth_left     = {"dx": 0.0, "dy": 0.0}
        self.smooth_right    = {"dx": 0.0, "dy": 0.0}
        self.smoothing       = 0.5 
        self.frame_counter   = 0
        self.process_every_n = 2

    def _gaze_dir(self, iris_center, eye_lms, fw, fh, lm):
        l = np.array([lm[eye_lms[0]].x * fw, lm[eye_lms[0]].y * fh])
        r = np.array([lm[eye_lms[1]].x * fw, lm[eye_lms[1]].y * fh])
        t = np.array([lm[eye_lms[2]].x * fw, lm[eye_lms[2]].y * fh])
        b = np.array([lm[eye_lms[3]].x * fw, lm[eye_lms[3]].y * fh])

        eye_w = np.linalg.norm(r - l)
        eye_h = np.linalg.norm(b - t)
        if eye_w < 1e-6:
            return "CENTER", "CENTER", 0.0, 0.0

        center = (l + r) / 2.0
        dx = (iris_center[0] - center[0]) / eye_w
        dy = (iris_center[1] - center[1]) / eye_h

        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))

        h  = "LEFT"  if dx < -self.H_THRESH else "RIGHT" if dx >  self.H_THRESH else "CENTER"
        v  = "UP"    if dy < -self.V_THRESH else "DOWN"  if dy >  self.V_THRESH else "CENTER"
        return h, v, dx, dy

    def _process_eye(self, iris_ids, eye_ids, smooth, fw, fh, lm):
        pts    = np.array([(int(lm[i].x * fw), int(lm[i].y * fh)) for i in iris_ids])
        center = pts.mean(axis=0).astype(int)
        h, v, dx, dy = self._gaze_dir(center, eye_ids, fw, fh, lm)
        a = 1.0 - self.smoothing
        smooth["dx"] = smooth["dx"] * self.smoothing + dx * a
        smooth["dy"] = smooth["dy"] * self.smoothing + dy * a
        return {
            "direction": {"horizontal": h, "vertical": v},
            "dx": round(smooth["dx"], 3),
            "dy": round(smooth["dy"], 3),
        }

    def process(self, frame: np.ndarray) -> dict:
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n != 0:
            return self.current_gaze

        h, w = frame.shape[:2]
        res  = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try:
                self.current_gaze["left_eye"] = self._process_eye(
                    self.LEFT_IRIS, self.LEFT_EYE, self.smooth_left, w, h, lm)
            except Exception:
                pass
            try:
                self.current_gaze["right_eye"] = self._process_eye(
                    self.RIGHT_IRIS, self.RIGHT_EYE, self.smooth_right, w, h, lm)
            except Exception:
                pass

            lc = (self.current_gaze["left_eye"]["direction"]["horizontal"]  == "CENTER" and
                self.current_gaze["left_eye"]["direction"]["vertical"]    == "CENTER")
            rc = (self.current_gaze["right_eye"]["direction"]["horizontal"] == "CENTER" and
                self.current_gaze["right_eye"]["direction"]["vertical"]   == "CENTER")
            self.current_gaze["looking_away"] = not (lc and rc)
        else:
            self.current_gaze["looking_away"] = True  

        return self.current_gaze


# AUDIO MONITOR
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
        if self.stream:
            self.stream.stop()
            self.stream.close()


#OBJECT DETECTOR
class ObjectDetector:
    UNAUTHORIZED = {
        63: "Laptop",     64: "Laptop",
        67: "Cell Phone", 77: "Cell Phone",
        73: "Book",       74: "Book",       84: "Book",
    }
    CONF = 0.40

    def detect(self, model, frame: np.ndarray) -> list:
        results    = model(frame,
                        classes=list(self.UNAUTHORIZED),
                        conf=self.CONF,
                        iou=0.45,
                        verbose=False,
                        max_det=10)
        detections = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < self.CONF:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cid = int(box.cls[0])
                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "class_id":   cid,
                    "confidence": conf,
                    "name":       self.UNAUTHORIZED.get(cid, "Unauthorized Object"),
                })
        return detections


#SYSTEM
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
        self.object_detect_every = 2    #object detection every N frames
        self.save_every_n_frames = 30   #append a record every N frames
        self.identify_every_n    = 45   #ArcFace identification every N frames

        #State
        self.frame_count          = 0
        self.fps                  = 0
        self._fps_counter         = 0
        self._fps_time            = time.time()
        self.unauthorized_objects = []
        self.face_count           = 0
        self.current_identity     = {"best_match": "Unknown", "confidence": 0.0}

        #Output buffer
        self.saved_detections = []
        self.occurrence_count = 0

    #initialize
    def initialize(self):
        _s = sys.stderr; sys.stderr = _Devnull()
        try:
            self.object_model = YOLO("yolov8n.pt")
        except Exception as e:
            sys.stderr = _s
            raise RuntimeError(f"Object model load failed: {e}")

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
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Could not open any webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print({
            "event":             "system_ready",
            "device":            str(device),
            "resolution":        f"{self.frame_width}x{self.frame_height}",
            "registry_entries":  len(_registry_names),
            "controls":          {"q": "quit + auto-save", "s": "save now"},
        })

    #detection pipeline
    def _run_detections(self, frame: np.ndarray) -> np.ndarray:
        #YOLO face bounding boxes
        self.face_count = 0
        if self.yolo_face_model:
            for res in self.yolo_face_model(frame, conf=0.5, verbose=False):
                if res.boxes is None:
                    continue
                for box in res.boxes:
                    self.face_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        #Object detection
        if self.frame_count % self.object_detect_every == 0:
            self.unauthorized_objects = self.object_detector.detect(
                self.object_model, frame
            )
        for obj in self.unauthorized_objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return frame

    #face recognition overlay
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

    #save to JSON
    def save_detections(self, filename: str = None) -> str:
        if filename is None:
            filename = f"proctoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(
                {"total_occurrences": self.occurrence_count,
                "detections":        self.saved_detections},
                f, indent=2, cls=NumpyEncoder,
            )
        print({"event": "saved", "file": filename,
            "occurrences": self.occurrence_count})
        return filename

    #main loop
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self.frame_count += 1

            frame     = self._run_detections(frame)
            frame     = self._run_recognition(frame)
            gaze_data = self.gaze_tracker.process(frame)

            #FPS
            now = time.time()
            if now - self._fps_time >= 1.0:
                self.fps          = self.frame_count - self._fps_counter
                self._fps_counter = self.frame_count
                self._fps_time    = now

            #Build detection record
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


            if self.frame_count % self.save_every_n_frames == 0:
                self.saved_detections.append(record)
                self.occurrence_count += 1
                print(record)         

            cv2.imshow("Proctoring System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if self.saved_detections:
                    self.save_detections()
                self.running = False
            elif key == ord("s"):
                self.save_detections()

    def cleanup(self):
        if self.audio_monitor:
            self.audio_monitor.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


#ENTRY POINT
if __name__ == "__main__":
    import sys as _sys

    #REGISTRATION MODE
    if len(_sys.argv) >= 4 and _sys.argv[1].lower() == "register":
        name      = _sys.argv[2]
        img_paths = _sys.argv[3:]
        result    = register_person(name, img_paths)
        print(result)   
        _sys.exit(0)

    #MONITORING MODE
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
