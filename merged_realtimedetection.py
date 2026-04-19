import numpy as np
import cv2
from ultralytics import YOLO
import sounddevice as sd
import json
from datetime import datetime
import time
import os
from collections import defaultdict
import mediapipe as mp

#JSON ENCODER
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

#GAZE TRACKING WITH MEDIAPIPE
class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [33, 133, 160, 144]
        self.RIGHT_EYE = [362, 263, 387, 373]
        
        self.current_gaze = {
            "left_eye": {"direction": {"horizontal": "CENTER", "vertical": "CENTER"}, "dx": 0.0, "dy": 0.0},
            "right_eye": {"direction": {"horizontal": "CENTER", "vertical": "CENTER"}, "dx": 0.0, "dy": 0.0},
            "looking_away": False
        }
        
        self.smooth_left = {"dx": 0.0, "dy": 0.0}
        self.smooth_right = {"dx": 0.0, "dy": 0.0}
        self.smoothing = 0.6
        self.frame_counter = 0
        self.process_every_n = 2
    
    def get_gaze_direction(self, iris_center, eye_landmarks, frame_w, frame_h, landmarks):
        try:
            left_pt = np.array([landmarks[eye_landmarks[0]].x * frame_w,
                               landmarks[eye_landmarks[0]].y * frame_h])
            right_pt = np.array([landmarks[eye_landmarks[1]].x * frame_w,
                                landmarks[eye_landmarks[1]].y * frame_h])
            top_pt = np.array([landmarks[eye_landmarks[2]].x * frame_w,
                              landmarks[eye_landmarks[2]].y * frame_h])
            bot_pt = np.array([landmarks[eye_landmarks[3]].x * frame_w,
                              landmarks[eye_landmarks[3]].y * frame_h])

            eye_width = np.linalg.norm(right_pt - left_pt)
            eye_height = np.linalg.norm(bot_pt - top_pt)
            if eye_width == 0:
                return "CENTER", "CENTER", 0.0, 0.0

            eye_center = (left_pt + right_pt) / 2
            dx = (iris_center[0] - eye_center[0]) / eye_width
            dy = (iris_center[1] - eye_center[1]) / eye_height

            h_dir = "LEFT" if dx < -0.10 else "RIGHT" if dx > 0.10 else "CENTER"
            v_dir = "UP" if dy < -0.08 else "DOWN" if dy > 0.08 else "CENTER"
            
            return h_dir, v_dir, float(dx), float(dy)
        except Exception:
            return "CENTER", "CENTER", 0.0, 0.0
    
    def process(self, frame):
        self.frame_counter += 1
        
        if self.frame_counter % self.process_every_n != 0:
            return self.current_gaze
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            try:
                left_iris_pts = np.array([
                    (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in self.LEFT_IRIS
                ])
                left_center = left_iris_pts.mean(axis=0).astype(int)
                
                h_dir, v_dir, dx, dy = self.get_gaze_direction(
                    left_center, self.LEFT_EYE, w, h, face_landmarks.landmark
                )
                
                self.smooth_left["dx"] = self.smooth_left["dx"] * self.smoothing + dx * (1 - self.smoothing)
                self.smooth_left["dy"] = self.smooth_left["dy"] * self.smoothing + dy * (1 - self.smoothing)
                
                self.current_gaze["left_eye"] = {
                    "direction": {"horizontal": h_dir, "vertical": v_dir},
                    "dx": round(self.smooth_left["dx"], 3),
                    "dy": round(self.smooth_left["dy"], 3)
                }
            except Exception:
                pass
            
            try:
                right_iris_pts = np.array([
                    (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in self.RIGHT_IRIS
                ])
                right_center = right_iris_pts.mean(axis=0).astype(int)
                
                h_dir, v_dir, dx, dy = self.get_gaze_direction(
                    right_center, self.RIGHT_EYE, w, h, face_landmarks.landmark
                )
                
                self.smooth_right["dx"] = self.smooth_right["dx"] * self.smoothing + dx * (1 - self.smoothing)
                self.smooth_right["dy"] = self.smooth_right["dy"] * self.smoothing + dy * (1 - self.smoothing)
                
                self.current_gaze["right_eye"] = {
                    "direction": {"horizontal": h_dir, "vertical": v_dir},
                    "dx": round(self.smooth_right["dx"], 3),
                    "dy": round(self.smooth_right["dy"], 3)
                }
            except Exception:
                pass
            
            left_center = (self.current_gaze["left_eye"]["direction"]["horizontal"] == "CENTER" and 
                        self.current_gaze["left_eye"]["direction"]["vertical"] == "CENTER")
            right_center = (self.current_gaze["right_eye"]["direction"]["horizontal"] == "CENTER" and 
                        self.current_gaze["right_eye"]["direction"]["vertical"] == "CENTER")
            
            self.current_gaze["looking_away"] = not (left_center and right_center)
        else:
            self.current_gaze["looking_away"] = False
        
        return self.current_gaze

#AUDIO MONITOR
class AudioMonitor:
    def __init__(self, threshold=0.15):
        self.threshold = threshold
        self.audio_level = 0
        self.is_talking = False
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        volume_norm = np.linalg.norm(indata) * 10
        self.audio_level = volume_norm
        self.is_talking = volume_norm > self.threshold

    def start(self):
        self.stream = sd.InputStream(callback=self._audio_callback)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

#OBJECT DETECTION
class ObjectDetector:
    def __init__(self):
        self.unauthorized_classes = {
            63: "Laptop",
            64: "Laptop", 
            67: "Cell Phone",
            73: "Book",
            74: "Book",
            77: "Cell Phone",
            84: "Book"
        }
        
        self.conf_threshold = 0.35
        
    def get_class_name(self, class_id):
        return self.unauthorized_classes.get(class_id, "Unauthorized Object")
    
    def detect_objects(self, model, frame):
        results = model(frame, 
                    classes=list(self.unauthorized_classes.keys()),
                    conf=self.conf_threshold,
                    iou=0.45,
                    verbose=False,
                    max_det=10)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.conf_threshold:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class_id': class_id,
                            'confidence': confidence,
                            'name': self.get_class_name(class_id)
                        })
        
        return detections

#MAIN SYSTEM
class ProctoringSystem:
    def __init__(self):
        self.object_model = None
        self.face_model = None
        self.gaze_tracker = None
        self.audio_monitor = None
        self.object_detector = None
        self.cap = None
        self.running = True
        
        #Performance settings
        self.frame_width = 640
        self.frame_height = 480
        self.object_detect_every = 1
        self.save_every_n_frames = 30  #Save every 30 frames to avoid huge files
        
        self.frame_count = 0
        self.fps = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        
        self.unauthorized_objects = []
        self.face_count = 0
        
        #Store detections with occurrence counting
        self.saved_detections = []
        self.occurrence_count = 0
        
    def initialize(self):
        print("\n" + "="*50)
        print("PROCTORING SYSTEM INITIALIZATION")
        print("="*50)
        
        #Load YOLO models
        print("Loading YOLO models...")
        try:
            self.object_model = YOLO("yolov8n.pt")
            print("Object detection model loaded")
        except Exception as e:
            print(f"Failed to load object model: {e}")
            raise
        
        try:
            self.face_model = YOLO("yolov8n-face.pt")
            print("Face detection model loaded")
        except:
            self.face_model = None
            print("Face model not found")
        
        self.object_detector = ObjectDetector()
        
        print("Initializing MediaPipe gaze tracker...")
        self.gaze_tracker = GazeTracker()
        print("Gaze tracker ready")
        
        print("Initializing audio monitor...")
        self.audio_monitor = AudioMonitor(threshold=0.15)
        self.audio_monitor.start()
        print("Audio monitor ready")
        
        print("Opening webcam...")
        for cam_idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                print(f"Camera {cam_idx} opened")
                break
        
        if not self.cap or not self.cap.isOpened():
            raise Exception("Could not open any webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n✓ SYSTEM READY!\n")
        print("="*50)
        print("CONTROLS:")
        print("  'q' - Quit application (auto-saves with occurrences)")
        print("  's' - Save current detections to JSON")
        print("="*50 + "\n")
    
    def draw_bounding_boxes(self, frame):
        #Face detection
        self.face_count = 0
        if self.face_model:
            face_results = self.face_model(frame, conf=0.5, verbose=False)
            for result in face_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        self.face_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        #Object detection
        if self.frame_count % self.object_detect_every == 0:
            self.unauthorized_objects = self.object_detector.detect_objects(self.object_model, frame)
        
        for obj in self.unauthorized_objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        return frame
    
    def save_detections(self, filename=None):
        if filename is None:
            filename = f"proctoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        #Create output with occurrences
        output = {
            "total_occurrences": self.occurrence_count,
            "detections": self.saved_detections
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nSaved {self.occurrence_count} occurrences to: {filename}")
        return filename
    
    def run(self):
        print("Monitoring active. Press 'q' to quit, 's' to save.\n")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            self.frame_count += 1
            
            #Draw bounding boxes
            frame = self.draw_bounding_boxes(frame)
            
            #Gaze tracking
            gaze_data = self.gaze_tracker.process(frame)
            
            #Calculate FPS
            if time.time() - self.fps_time > 1.0:
                self.fps = self.frame_count - self.fps_counter
                self.fps_counter = self.frame_count
                self.fps_time = time.time()
            
            #Create detection record
            detection_record = {
                "occurrence": self.occurrence_count + 1,
                "timestamp": datetime.now().isoformat(),
                "fps": self.fps,
                "face_count": self.face_count,
                "audio": {
                    "detected": self.audio_monitor.is_talking,
                    "level": round(self.audio_monitor.audio_level, 3)
                },
                "unauthorized_objects": {
                    "detected": len(self.unauthorized_objects) > 0,
                    "count": len(self.unauthorized_objects),
                    "items": [
                        {
                            "type": obj['name'],
                            "confidence": round(obj['confidence'], 2)
                        }
                        for obj in self.unauthorized_objects
                    ]
                },
                "gaze": {
                    "left_eye": {
                        "direction": gaze_data["left_eye"]["direction"],
                        "dx": gaze_data["left_eye"]["dx"],
                        "dy": gaze_data["left_eye"]["dy"]
                    },
                    "right_eye": {
                        "direction": gaze_data["right_eye"]["direction"],
                        "dx": gaze_data["right_eye"]["dx"],
                        "dy": gaze_data["right_eye"]["dy"]
                    },
                    "looking_away": gaze_data["looking_away"]
                }
            }
            
            #Save every N frames
            if self.frame_count % self.save_every_n_frames == 0:
                self.saved_detections.append(detection_record)
                self.occurrence_count += 1
                
                print(f"\nOccurrence #{self.occurrence_count} saved at frame {self.frame_count}")
                print(json.dumps(detection_record, indent=2, cls=NumpyEncoder))
                print("-" * 50)
            
            cv2.imshow("Proctoring System", frame)
            
            #Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                #Save before quitting
                if self.saved_detections:
                    self.save_detections()
                print("\nQuitting...")
                self.running = False
                break
            elif key == ord('s'):
                self.save_detections()
    
    def cleanup(self):
        print("\nShutting down...")
        if self.audio_monitor:
            self.audio_monitor.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped!")

#MAIN
if __name__ == "__main__":
    system = None
    try:
        system = ProctoringSystem()
        system.initialize()
        system.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if system and system.saved_detections:
            print("Saving detections before exit...")
            system.save_detections()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            system.cleanup()
