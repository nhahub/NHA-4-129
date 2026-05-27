# new file: engine.py
from proctoring_system import ProctoringSystem
class ProctorEngineWrapper:
    def __init__(self):
        self.system = ProctoringSystem()
        self.running = False

    def start(self):
        self.system.initialize()
        self.running = True
        self.system.run()

    def stop(self):
        self.system.running = False
        self.system.cleanup()

    def get_status(self):
        return {
            "running": self.running,
            "detections": len(self.system.saved_detections),
        }