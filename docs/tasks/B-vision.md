# Lane B tasks: cameras, scene state, verification, dashboard

Lane B gives everyone the scene in meters, ten times a second, accurate to 1.5 cm, then builds the dashboard judges watch. The camera check after every real run uses the same success test that scored the simulations.

**Progress** (in time order; B sleeps 3:30 to 7:30 AM Saturday)

- [ ] B1 · Cameras · Fri 11:00 PM
- [ ] B2 · OpenCV with ArUco · Fri 11:15 PM
- [ ] B3 · Markers · Sat 12:45 AM
- [ ] B4 · Mount the overhead camera · Sat 1:15 AM
- [ ] B5 · Table calibration · Sat 2:15 AM
- [ ] B6 · Scene state · Sat 3:00 AM (M1a)
- [ ] B7 · Real verification · Sat 3:30 AM
- [ ] B8 · Wrist camera · Sat 8:30 AM
- [ ] B9 · Dashboard · Sat 6:00 PM

## B1 · Cameras

**Needs:** nothing. **By:** Fri 11:00 PM. **Files:** `vision/cameras.py`, `config/cameras.yaml`.

Find indices with `lerobot-find-cameras opencv` (older releases: `python -m lerobot.find_cameras opencv`). Open both at 640x480, 30 fps, MJPG, which keeps two streams within USB bandwidth:

```python
cap = cv2.VideoCapture(index)        # Windows: cv2.VideoCapture(index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

Turn off auto exposure and autofocus if the camera allows it (the settings differ by OS), because exposure flicker moves detections. Plug the cameras into different USB ports from the arm.

**Done when:** `python -m vision.cameras --check 1800` runs both feeds for 30 minutes without dropping frames.

## B2 · OpenCV with ArUco

**Needs:** nothing. **By:** Fri 11:15 PM. Run `python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"`. OpenCV 4.7 and later include ArUco in the main package. If it prints False, uninstall every OpenCV package and install a single `opencv-contrib-python` of the same version; two OpenCV packages side by side break `cv2`.

**Done when:** the command prints a version of 4.7 or later and `True`.

## B3 · Markers

**Needs:** A11 (final object list). **By:** Sat 12:45 AM. **Files:** `scripts/make_markers.py`.

Use dictionary `DICT_4X4_50`. IDs 0 to 7 are table calibration markers (6 cm), 10 to 19 go on top of objects (3 cm), 30 to 34 on obstacles (4 cm), and 40 is the wristband (4 cm). Generate each with `cv2.aruco.generateImageMarker`, print at 100% scale, measure the printed size, and glue each flat with a white border. With no printer, use one color of tape per object with HSV thresholds and taped object spots; accuracy drops and yaw is lost.

**Done when:** every marker in `config/objects.yaml` plus the 8 calibration markers is printed, measured and detected from the camera's height.

## B4 · Mount the overhead camera

**Needs:** B1. **By:** Sat 1:15 AM. **Files:** `config/scene.yaml` (camera block).

Mount it about 65 cm above the table, behind the arm, angled down at the workspace so the arm doesn't hide objects while reaching. Clamp and tape it: it must not move after calibration. Measure the lens height above the table (`Hc`) and the table point directly under the lens (`c`) in the robot frame. With the arm at `home_q`, every calibration and object marker must stay visible; agree the home pose with Lane A (A10).

**Done when:** `Hc` and `c` are in `config/scene.yaml`, and a frame with the arm at home shows all markers.

## B5 · Table calibration (`vision/calibrate.py`)

**Needs:** B3, B4. **By:** Sat 2:15 AM. **Files:** `vision/calibrate.py`, `config/homography.npy`, `config/scene.yaml`.

Tape calibration markers 0 to 7 around the outside of the workspace, at least 5 cm from the tray, objects, cereal box and arm base: about (0.08, ±0.34), (0.27, ±0.34), (0.46, ±0.34) and (0.46, ±0.17) m, and tape-measure where each center landed into `config/scene.yaml`. Fit `H` with `cv2.findHomography(pixels, table_xy, cv2.RANSAC)` over all 8 centers and save it; a table point is `H` times `[u, v, 1]`, normalized. Eight spread points average out most lens distortion. If the edges still read over 1.5 cm off, calibrate the lens with a printed checkerboard (`cv2.calibrateCamera`, 15 images) and undistort before detecting.

A marker on top of an object of height `h` appears farther from the camera than the object really is, so correct it:

```latex
p = c + (p_{table} - c)(1 - h / H_c)
```

Apply this to objects and obstacles (`h` from `config/objects.yaml`) and to the wristband (`h` = 0.03).

**Done when:** markers at 5 tape-measured check spots read within 1.0 cm (1.5 cm worst case), and `POST /calibrate` reports that error.

## B6 · Scene state (`vision/scene.py`)

**Needs:** B5. **By:** Sat 3:00 AM. **Files:** `vision/scene.py`, `vision/markers.py`.

A thread inside the orchestrator process:

1. Grab a frame and detect markers with `cv2.aruco.ArucoDetector`.
2. Map IDs through `config/objects.yaml` into objects, obstacles and the hand.
3. Compute x and y with the height correction, and yaw from the marker's top edge (corner 0 to corner 1, both mapped through `H`).
4. Median-filter the last 5 readings; set `seen: false` after 0.5 s without a detection.
5. Publish the scene with `zones` from `config/scene.yaml`, validated against `schemas/scene.json`, plus an annotated MJPEG stream for the dashboard. Expose `latest()` and `snapshot(n=10)`, the median of the next `n` frames.

**Done when:** published positions match tape measurements within 1.5 cm and yaw within 10°, and `python -m vision.scene --print` prints valid scene JSON at 10 Hz.

## B7 · Real verification (`vision/verify.py`)

**Needs:** B6, S0 (`sim/success.py`). **By:** Sat 3:30 AM. **Files:** `vision/verify.py`, `tests/test_verify.py`, `tests/fixtures/verify/`.

- `before = snapshot(10)` when planning starts, so it is the scene the sandbox saw.
- After the replay ends and the arm is home, wait 1 s and take `after = snapshot(10)`. If an object the spec names is unseen, retry for up to 3 s.
- `ok = evaluate(spec, before, after, args, events)` from `sim/success.py`, including the implicit obstacle check.
- Save both snapshots and the result to `runs/<task_id>/`.

**Done when:** `pytest -q tests/test_verify.py` passes on 10 recorded before-and-after pairs (5 that should pass, 5 that should fail), and a live check on the table agrees.

## B8 · Wrist camera

**Needs:** B1. **By:** Sat 8:30 AM. Live view on the dashboard and one still per grasp in the run log. No control use; that was cut.

**Done when:** the wrist feed shows on the dashboard and `runs/<task_id>/` holds a still for each grasp.

## B9 · Dashboard (`web/index.html`)

**Needs:** S0, B6. **By:** Sat 6:00 PM, built from 7:30 AM. **Files:** `web/index.html`, `web/static/`.

One page served by the orchestrator: plain HTML and JavaScript, with Chart.js for curves. A good task for Cursor's agent, for the Make It Legendary story.

- Overhead feed with marker overlays, and the wrist feed.
- Typed command box and the live event timeline from `WS /events`.
- Success-test banner: the spec's `english` and JSON, shown as soon as `spec_ready` arrives and labeled "written before any code".
- Six candidate cards: number, strategy hint, a tick or cross per trial, collisions, pass or fail with `fail_reason`, a "local" label on fallback, and the winner highlighted.
- In `DEMO_MODE`, a countdown from 75 s.
- Skill library: name, English description, real runs and successes, newest highlighted; Reset demo and Load backup buttons.
- RL panel: four reward-variant curves, the chosen one, its simulation video, and which carry method is live (`carry` from `GET /state`).
- STOP button (it freezes the arm) plus Esc anywhere on the page, Resume, and Check calibration.
- The `/voice` page embedded.

**Done when:** a teammate who hasn't seen the code follows a new-skill run from the dashboard alone, without narration, and Esc freezes the fake arm with `FAKE=1`.
