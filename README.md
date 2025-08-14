# AthleteRise Cover Drive Analysis

## Overview
This project analyzes a cricket cover drive shot from a video using real-time human pose estimation.  
It overlays per-frame pose keypoints, calculates key biomechanical metrics, and generates:
- An annotated video with per-frame metrics and feedback cues
- A final summary frame with performance scores and comments
- A JSON report containing frame-by-frame metrics and final evaluation

Updated to match the revised AthleteRise internship brief:
- Full video analysis (no screenshots)
- Live metric overlays (≥3 metrics per frame)
- Annotated output video
- Final summary at the end of playback

---

## Features
1. **Real-time Pose Detection** using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose).
2. **Metrics Computed** (per frame):
   - `elbow_angle` — angle at the front elbow
   - `spine_lean` — lean of spine from vertical
   - `head_knee` — horizontal offset between head and front knee
   - `foot_dir` — direction of front foot relative to horizontal
3. **Live Feedback Cues** — on-screen OK/Off messages based on thresholds.
4. **Final Report Generation**:
   - Annotated video (`annotated_video.mp4`)
   - JSON report with per-frame metrics, scores, and comments
   - Final summary frame appended to video and shown at runtime

---

## Project Structure
.
├── cover_drive_analysis_realtime.py # Main script
├── form.py # Metric calculation functions
├── posetrack.py # Pose detection wrapper
├── Video_dowd.py # Utility to download video from YouTube
├── Data/
│ ├── Input/ # (Optional) raw frames (not used in final)
│ ├── Output/ # Output video + reports
│ ├── Video/ # Downloaded cricket video
├── requirements.txt
└── README.md
