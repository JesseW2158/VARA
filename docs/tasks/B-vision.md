# Lane B tasks: cameras, scene state, verification, dashboard

**Continuation verification:** [2026-10-17 evidence](../verification-2026-10-17.md). Checked boxes have supporting evidence; hardware and live-service acceptance remains open.

Lane B gives everyone the scene in meters, ten times a second, accurate to 1.5 cm, then builds the dashboard judges watch. The camera check after every real run uses the same success test that scored the simulations.

**Progress** (in time order; B sleeps 3:30 to 7:30 AM Saturday)

- [ ] B1 · Cameras · Fri 11:00 PM
- [x] B2 · OpenCV with ArUco · Fri 11:15 PM
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

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m vision.cameras --check 1800` now writes a JSON report with per-feed timing (failed reads, late gaps over 2.5 frame periods, long gaps over 0.5 s, effective fps, the driver's read-back of size, fps and FOURCC), the verdict and the exit code to a new `runs/setup/camera-check-<ts>.json`, or a new `--out PATH`; an existing path is refused before any camera opens. `--criterion strict` is the default and B1's bar. `--criterion tracking` passes with zero failed reads, no gap over 0.5 s and at least 25 fps per feed, which is what the tracker needs; the team decides whether that bar is acceptable here. `.\.venv\Scripts\python.exe -m vision.cameras --diagnose 60` reads each feed in its own thread to name a late feed (`runs/setup/camera-diagnose-<ts>.json`), and `.\.venv\Scripts\python.exe -m vision.cameras --list` opens indexes 0 to 7 briefly and saves one JPEG per opened index plus `report.json` under `runs/setup/camera-list-<ts>/`. Close the orchestrator and the SenseCraft app first. The 2026-10-17 physical attempt failed the strict bar early on late frames with zero failed reads, and a diagnostic pointed at camera 2's timing; see the [verification record](../verification-2026-10-17.md). Order: Part 4 of the [operator runbook](../runbook.md#part-4-camera-and-table-b1-b3-to-b6-nothing-here-moves-the-arm).

## B2 · OpenCV with ArUco

**Needs:** nothing. **By:** Fri 11:15 PM. Run `python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"`. OpenCV 4.7 and later include ArUco in the main package. If it prints False, uninstall every OpenCV package and install a single `opencv-contrib-python` of the same version; two OpenCV packages side by side break `cv2`.

**Done when:** the command prints a version of 4.7 or later and `True`.

## B3 · Markers

**Needs:** A11 (final object list). **By:** Sat 12:45 AM. **Files:** `scripts/make_markers.py`.

Use dictionary `DICT_4X4_50`. IDs 0 to 7 are table calibration markers (6 cm), 10 to 19 go on top of objects (3 cm), 30 to 34 on obstacles (4 cm), and 40 is the wristband (4 cm). Generate each with `cv2.aruco.generateImageMarker`, print at 100% scale, measure the printed size, and glue each flat with a white border. With no printer, use one color of tape per object with HSV thresholds and taped object spots; accuracy drops and yaw is lost.

**Done when:** every marker in `config/objects.yaml` plus the 8 calibration markers is printed, measured and detected from the camera's height.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe scripts/make_markers.py` also prints five spare 3 cm cards labelled "spare, I2 check" for I2 (ids 13 to 17 by default; `--spare-ids ID ...` picks other free ids in 10 to 19, an id `config/objects.yaml` assigns is skipped, and the flag alone prints none), and its manifest `runs/markers/markers.json` marks each card `spare`. For the detection check, `.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/visibility-1 --expect-all --frames 10 --printed-side-mm ID=MM ...` takes the median corners over 10 reads, compares the detected ids with markers 0 to 7 plus every id in `config/objects.yaml` (the wristband 40 only with `--expect-hand`), and exits 1 naming any missing id. The ruler-measured sides are stored next to the nominal sides from the manifest in `markers.json` in the new `--out` folder, and a value under half or over twice the nominal side is refused as a unit slip.

## B4 · Mount the overhead camera

**Needs:** B1. **By:** Sat 1:15 AM. **Files:** `config/scene.yaml` (camera block).

Mount it about 65 cm above the table, behind the arm, angled down at the workspace so the arm doesn't hide objects while reaching. Clamp and tape it: it must not move after calibration. Measure the lens height above the table (`Hc`) and the table point directly under the lens (`c`) in the robot frame. With the arm at `home_q`, every calibration and object marker must stay visible; agree the home pose with Lane A (A10).

**Done when:** `Hc` and `c` are in `config/scene.yaml`, and a frame with the arm at home shows all markers.

**Tools added 2026-10-17:** The B3 capture doubles as this check: with the arm at `home_q` and every object on its spot, `.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/setup/markers/visibility-1 --expect-all --frames 10` exits 0 only when every expected marker is detected. Its `markers.json` records `camera_pose: {Hc, c, source, placeholder}`, and `placeholder` stays true while the leading comment or the `camera:` block of `config/scene.yaml` still says placeholder, so a capture counts as B4 evidence only after `Hc` and `c` are measured and that word is removed. Order: Part 4 of the [operator runbook](../runbook.md#part-4-camera-and-table-b1-b3-to-b6-nothing-here-moves-the-arm).

## B5 · Table calibration (`vision/calibrate.py`)

**Needs:** B3, B4. **By:** Sat 2:15 AM. **Files:** `vision/calibrate.py`, `config/homography.npy`, `config/scene.yaml`.

Tape calibration markers 0 to 7 around the outside of the workspace, at least 5 cm from the tray, objects, cereal box and arm base: about (0.08, ±0.34), (0.27, ±0.34), (0.46, ±0.34) and (0.46, ±0.17) m, and tape-measure where each center landed into `config/scene.yaml`. Fit `H` with `cv2.findHomography(pixels, table_xy, cv2.RANSAC)` over all 8 centers and save it; a table point is `H` times `[u, v, 1]`, normalized. Eight spread points average out most lens distortion. If the edges still read over 1.5 cm off, calibrate the lens with a printed checkerboard (`cv2.calibrateCamera`, 15 images) and undistort before detecting.

A marker on top of an object of height `h` appears farther from the camera than the object really is, so correct it:

```latex
p = c + (p_{table} - c)(1 - h / H_c)
```

Apply this to objects and obstacles (`h` from `config/objects.yaml`) and to the wristband (`h` = 0.03).

**Done when:** markers at 5 tape-measured check spots read within 1.0 cm (1.5 cm worst case), and `POST /calibrate` reports that error.

**Tools added 2026-10-17:** Record each tape-measured check spot with `.\.venv\Scripts\python.exe scripts/capture_table.py --out runs/table-calibration/spot-1 --check-point X Y` (X and Y in meters; loose marker 10 on the spot unless `--spot-id ID`). It appends `{pixel, table}` to `measurements/check-points.json`, or to `--append PATH` (it does not follow `VARA_CHECK_POINTS`, so pass that file when the variable is set), keeps the full record in `measurements/check-points.provenance.json`, prints `N of 5 recorded`, and from the fifth spot on prints `check points valid`. `POST /calibrate` and the dashboard's Check calibration now also report `check_points` (`{n, mean_cm, max_cm, ok, file}`, or null when no file exists) under the marker error, shown as `5-point check: mean X cm, max Y cm (ok|FAIL)`. Lens fallback: `.\.venv\Scripts\python.exe -m vision.lens --images runs/lens` fits 15 or more board photos into `config/lens.npz` (refused above 1 px RMS), and `.\.venv\Scripts\python.exe -m vision.lens --check <held-out photo>` reports the reprojection error. Once `config/lens.npz` exists, the scene tracker, `vision.calibrate --live` and `--check`, `capture_table.py` and `POST /calibrate` undistort every frame before detection and print or record `lens:` with the file's sha256 (null means raw frames); a corrupt lens file stops them at startup, and `capture_table.py` refuses to append to a list recorded with another lens. After any lens change, refit `vision.calibrate --live`, re-capture the five spots and restart the orchestrator; see [table calibration](../table-calibration.md#lens-calibration-b5-fallback) and Part 4 of the [operator runbook](../runbook.md#part-4-camera-and-table-b1-b3-to-b6-nothing-here-moves-the-arm).

## B6 · Scene state (`vision/scene.py`)

**Needs:** B5. **By:** Sat 3:00 AM. **Files:** `vision/scene.py`, `vision/markers.py`.

A thread inside the orchestrator process:

1. Grab a frame and detect markers with `cv2.aruco.ArucoDetector`.
2. Map IDs through `config/objects.yaml` into objects, obstacles and the hand.
3. Compute x and y with the height correction, and yaw from the marker's top edge (corner 0 to corner 1, both mapped through `H`).
4. Median-filter the last 5 readings; set `seen: false` after 0.5 s without a detection.
5. Publish the scene with `zones` from `config/scene.yaml`, validated against `schemas/scene.json`, plus an annotated MJPEG stream for the dashboard. Expose `latest()` and `snapshot(n=10)`, the median of the next `n` frames.

**Done when:** published positions match tape measurements within 1.5 cm and yaw within 10°, and `python -m vision.scene --print` prints valid scene JSON at 10 Hz.

**Tools added 2026-10-17:** `.\.venv\Scripts\python.exe -m vision.scene --accuracy measurements/scene-layout.json` is the accuracy check. Write the tape-measured layout as `{"items": {name: {x, y, yaw}}}` in meters and radians (yaw null for cylinders and the hand); the tool compares it with a 10-frame snapshot taken after a 1 s warm-up (`--frames`, `--warmup`). It exits 0 when every item is seen within 1.5 cm and 10 degrees, 1 otherwise, and 2 on bad input or no snapshot, and writes a new `runs/table-calibration/accuracy-<UTC stamp>.json` (or a new `--out REPORT`) whose tracker block records the lens. `.\.venv\Scripts\python.exe -m vision.scene --out runs/live/scene.json` is the live scene-file writer for A9, A10 and D7: it rewrites the file atomically with every published scene and marks every item unseen after a 1 s stall and when it stops. Before the first scene it writes the provenance sidecar `runs/live/scene.json.source.json` (writer, `source: live` only for a real tracker on a real camera and otherwise `fake`, tracker, camera, lens hash, pid, start time), replaces it at every start and removes it when it stops; every `--real --scene-file` reader requires a live one, so a fake or leftover file is refused. It exits 2 when that sidecar path already holds something other than an empty file or an earlier sidecar. Under `FAKE` or `FAKE_CAMERA` both commands need `--fake`, and their output is labelled synthetic.

## B7 · Real verification (`vision/verify.py`)

**Needs:** B6, S0 (`sim/success.py`). **By:** Sat 3:30 AM. **Files:** `vision/verify.py`, `tests/test_verify.py`, `tests/fixtures/verify/`.

- `before = snapshot(10)` when planning starts, so it is the scene the sandbox saw.
- After the replay ends and the arm is home, wait 1 s and take `after = snapshot(10)`. If an object the spec names is unseen, retry for up to 3 s.
- `ok = evaluate(spec, before, after, args, events)` from `sim/success.py`, including the implicit obstacle check.
- Save both snapshots and the result to `runs/<task_id>/`.

**Done when:** `pytest -q tests/test_verify.py` passes on 10 recorded before-and-after pairs (5 that should pass, 5 that should fail), and a live check on the table agrees.

**Tools added 2026-10-17:** The live check is `.\.venv\Scripts\python.exe -m vision.verify --before`, then move the object by hand, then `.\.venv\Scripts\python.exe -m vision.verify --after --spec <task spec or skill json> --args '{\"object\":\"juice_box\"}'` (PowerShell 5.1 needs the escaped quotes; `--events JSON` feeds `event` predicates). Each session lands in a new `runs/verify-check/<UTC stamp>/` (or `--out DIR`) holding `before.json`, `session.json`, `after.json`, `verify.json` and `result.json`; nothing is overwritten. The exit code is 0 when the spec holds, 1 when it does not, and 2 on bad input or a camera that delivers no frames. `session.json` and `result.json` record the tracker diagnostics with the lens hash, and an `--after` whose lens differs from the before's is refused. A physical `--after` needs the `session.json` of a physical `--before`: a before of unknown provenance is refused, and a synthetic before never pairs with a physical after. It never moves the arm. Offline: `--fake --fake-scene FILE`, labelled synthetic.

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
