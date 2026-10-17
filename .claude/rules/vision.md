---
paths:
  - "vision/**"
  - "config/objects.yaml"
  - "config/scene.yaml"
  - "config/cameras.yaml"
---
# Lane B rules
- Cameras run at 640x480, 30 fps, MJPG; on Windows use cv2.CAP_DSHOW.
- Detect markers with cv2.aruco.ArucoDetector and DICT_4X4_50.
- Correct marker positions for object height: p = c + (p_table - c)(1 - h/Hc).
- Every published scene dict validates against schemas/scene.json.
- Verification snapshots are the median of 10 frames, taken with the arm at home.
