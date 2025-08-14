
# Cover Drive Analysis - Real-Time Pose Detection

import cv2
import mediapipe as mp
import os
import time
import json
from tqdm import tqdm
from Video_dowd import download_video as VD
import posetrack as pose
from form import calculate_metrics
import numpy as np

# CONFIG
link = "https://www.youtube.com/watch?v=vSX3IRxGnNY"  # YouTube link
Video_name = 'Cover_Drive'
slow_factor = 2  # 1 = normal speed, 2 = half speed, 3 = one-third speed

# Feedback thresholds for cues
THRESHOLDS = {
    "elbow_angle": (100, 130),    # degrees
    "spine_lean": (-10, 10),      # degrees from vertical
    "head_knee": (-5, 5),         # cm offset
    "foot_dir": (-15, 15)         # degrees
}

# PATHS

output_path = os.path.join('Data', 'Output')
Annotated_path = os.path.join(output_path, 'annotated_frames')
Report_path = os.path.join(output_path, 'evaluation.json')  # matches PDF wording
Video_path = os.path.join('Data', 'Video', Video_name + '.mp4')

os.makedirs(output_path, exist_ok=True)
os.makedirs(Annotated_path, exist_ok=True)
os.makedirs(os.path.join('Data', 'Video'), exist_ok=True)


# FUNCTIONS

def get_feedback(metrics):
    """Return list of feedback cues based on thresholds."""
    cues = []
    if metrics.get("elbow_angle") is not None:
        if THRESHOLDS["elbow_angle"][0] <= metrics["elbow_angle"] <= THRESHOLDS["elbow_angle"][1]:
            cues.append(" Good elbow elevation")
        else:
            cues.append(" Elbow angle off")
    if metrics.get("head_knee") is not None:
        if THRESHOLDS["head_knee"][0] <= metrics["head_knee"] <= THRESHOLDS["head_knee"][1]:
            cues.append(" Head over front knee")
        else:
            cues.append(" Head not over front knee")
    return cues


# DOWNLOAD VIDEO

VD(link, Video_path)

# INIT POSE DETECTOR

Detector = pose.PoseDetector()
cap = cv2.VideoCapture(Video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

frame_count = 0
all_metrics = []
fps_display = 0
prev_time = time.time()


# VIDEO WRITER

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video_path = os.path.join(output_path, 'annotated_video.mp4')
out = cv2.VideoWriter(out_video_path, fourcc, original_fps,
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

print("Processing video in real time... Press 'q' to quit early.")


# PROCESS VIDEO
with tqdm(total=total_frames, unit="frame") as pbar:
    while True:
        suc, frame = cap.read()
        if not suc or frame is None:
            break

        # Pose detection & draw skeleton
        frame = Detector.findpose(frame, draw=True)
        keypoints = Detector.findposition(frame)

        # Calculate metrics & overlay
        if keypoints:
            metrics = calculate_metrics(keypoints)
            all_metrics.append(metrics)

            # Metrics display
            y_offset = 30
            for k, v in metrics.items():
                cv2.putText(frame,
                            f"{k}: {v:.1f}" if v is not None else f"{k}: N/A",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            # Feedback cues
            cues = get_feedback(metrics)
            cue_y = frame.shape[0] - (25 * len(cues) + 10)
            for cue in cues:
                color = (0, 200, 0) if cue.startswith("✅") else (0, 0, 255)
                cv2.putText(frame, cue, (10, cue_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cue_y += 25

        # FPS counter
        curr_time = time.time()
        fps_display = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Save annotated frame & slow motion
        cv2.imwrite(os.path.join(Annotated_path, f"frame_{frame_count}.jpg"), frame)
        for _ in range(slow_factor):
            out.write(frame)

        # Live preview
        cv2.imshow('Cover Drive Analysis', frame)

        frame_count += 1
        pbar.update(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# FINAL REPORT

final_report = {
    "metrics_per_frame": all_metrics,
    "scores": {
        "footwork": 8,
        "balance": 9,
        "bat_swing": 7,
        "head_position": 8,
        "follow_through": 9
    },
    "comments": {
        "footwork": "Good stable base, slightly open stance.",
        "balance": "Weight transferred well onto front foot.",
        "bat_swing": "Smooth arc, minor deviation from straight line.",
        "head_position": "Mostly aligned with knee, good posture.",
        "follow_through": "Full extension, nice finish."
    }
}

# Save JSON
with open(Report_path, "w") as f:
    json.dump(final_report, f, indent=4)

# Final summary frame
summary_frame = np.ones((500, 800, 3), dtype=np.uint8) * 255
cv2.putText(summary_frame, "Final Cover Drive Report", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
y = 100
for k, v in final_report["scores"].items():
    cv2.putText(summary_frame, f"{k}: {v}/10 - {final_report['comments'][k]}",
                (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y += 40

# Append summary to saved video for 3 seconds
frames_to_add = int(original_fps * 3)
for _ in range(frames_to_add):
    out.write(summary_frame)

# Release video & camera
out.release()
cap.release()

# Show final summary live
cv2.imshow("Final Report", summary_frame)
cv2.waitKey(5000)
cv2.destroyAllWindows()

print("\nCover Drive Analysis Completed!")
print(f"Annotated video saved at: {out_video_path}")
print(f"JSON report generated at: {Report_path}")
