# Exam Proctoring System
**Subtitle:** A Cloud-Native Framework for Real-Time Student Monitoring and Integrity Verification

---

## 1. Project Overview
The **Exam Proctoring Intelligence** platform is an AI-driven solution designed to automate the monitoring and integrity verification of online exams. By leveraging a multi-stage deep learning and computer vision architecture, the system provides real-time detection of suspicious behavior (such as cheating or inattentiveness) and generates a comprehensive **Integrity Score** to assist educators and administrators in decision-making.

---

## 2. Problem Statement
Online exams face several challenges:

- **Subjective Oversight:** Manual proctoring is prone to human error and inconsistent observation standards.  
- **Invisible Misconduct:** Subtle forms of cheating (e.g., off-screen phone use, multiple people in frame) can go unnoticed.  
- **Lack of Quantitative Metrics:** Institutions need objective scores and logs to document exam integrity rather than relying solely on human judgment.  

This system addresses these challenges by providing automated, reliable, and objective monitoring.

---

## 3. Technical Pipeline (Core Logic)
The system follows a modular pipeline for high accuracy and reliability:

### Stage I: Face & Identity Verification
- **Face Detection & Recognition:** Verifies the registered student’s identity and detects any unregistered faces.  
- **Multi-Face Detection:** Flags additional people in the frame for potential collusion.

### Stage II: Behavioral & Object Analysis
- **Gaze Tracking:** Monitors the student’s line of sight for suspicious patterns, such as prolonged off-screen attention.  
- **Object Detection (Books & Phones):** Detects unauthorized items in the student’s workspace.  
- **Behavioral Analysis:** Combines gaze, head pose, and object presence to identify unusual behaviors.

### Stage III: Post-processing & Reporting
- Suspicious events (multiple faces, gaze away, phone detected) are timestamped and logged.  
- An **Integrity Score** is calculated dynamically based on detected anomalies, providing a quantitative measure of exam compliance.

---

## 4. System Architecture & Deployment
The project is designed as a **Cloud-First, Modular AI Architecture**:

### Deep Learning Engine
- Developed using **PyTorch/TensorFlow**.  
- Components include:
  - Face detection and identity verification  
  - Multi-face detection and continuous tracking  
  - Gaze and head-pose analysis  
  - Detection of unauthorized objects in the exam environment  

Models are fine-tuned on educational datasets to ensure robustness under varying lighting, camera quality, and backgrounds.

### Deployment
- Backend: **FastAPI** for real-time video stream processing  
- Frontend: **Streamlit** or **Gradio** for interactive exam monitoring  
- Cloud: Deployed on **Microsoft Azure** for scalability and reliability

---

## 5. Project Milestones
1. **Environment setup**: GitHub repository initialization and dataset acquisition  
2. **Model development**: Identity verification and behavioral analysis models  
3. **Backend development**: Real-time processing with FastAPI  
4. **Frontend development**: Interactive monitoring with Streamlit/Gradio  
5. **Deployment & validation**: Full system deployment on Azure and testing

---

## 6. Expected Outcomes (Proof of Concept)
- **Real-Time Monitoring:** Detect unauthorized behavior and verify student identity.  
- **Automated Integrity Scoring:** Instant calculation of an **Integrity Score** per student.  
- **Comprehensive Reports:** Timestamped events with visual overlays for audit and review.  

---

## 7. Getting Started

### Prerequisites
- Python 3.9+  
- PyTorch / TensorFlow  
- FastAPI  
- Streamlit or Gradio  
- OpenCV, dlib, and other CV libraries

##  How to Run

### Step 1 — Register Your Face
Train the system on your photos before starting a session. Provide 3–5 clear, well-lit photos of your face:

```bash
python proctoring_system.py register "YourName" photo1.jpg photo2.jpg photo3.jpg
```

**Example:**
```bash
python proctoring_system.py register "Sherifa" sherifa1.jpg sherifa2.jpg sherifa3.jpg
```

You should see: {'status': 'success', 'name': 'Sherifa'}

---

### Step 2 — Run the Live Proctoring System
Start the real-time monitoring session. This opens the webcam and begins detecting faces, gaze, audio, and unauthorized objects instantly:

```bash
python proctoring_system.py
```

**Controls during the session:**

| Key | Action |
|-----|--------|
| `s` | Save detection data + rule report now |
| `q` | Save everything and quit |

When the session ends, two files are saved automatically:
- `proctoring_data_YYYYMMDD_HHMMSS.json` — raw detection data
- `proctoring_data_YYYYMMDD_HHMMSS_rule_report.json` — rule engine report

---

### Step 3 — Evaluate a Saved Session (Offline)
Re-run the rule engine on any previously saved session file to get a full per-occurrence breakdown:

```bash
python proctoring_system.py evaluate proctoring_data_YYYYMMDD_HHMMSS.json
```

**Example:**
```bash
python proctoring_system.py evaluate proctoring_data_20260429_102846.json
```

This prints a full session summary and saves an updated rule report alongside the original file.
