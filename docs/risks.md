# Risks and fallbacks

The biggest risk is still sim and real disagreeing, because every replayed path depends on that match; the second is time. Rows run from highest impact down.

| Risk | Early warning | Fallback | Owner |
| --- | --- | --- | --- |
| Sim and real disagree (joint signs, offsets, gripper mapping), so replayed paths miss | A6 or I2 over 1.5 cm, or grasps that pass in sim miss on the real arm | Fix `joint_map.yaml` offsets, refit the homography, add per-object grasp-height offsets; worst case, record pick poses with the leader arm | A, B |
| Gates slip | S0 or M1 two hours late | Cut order in Gates; RL go/no-go at 7:00 AM. Never cut sandboxes, voice, STOP or the validator | All |
| All six candidates pass, so the search looks pointless, or none pass | A15 rehearsals outside 2 to 4 passing of 6 | Move the cereal box per A15; sharper prompt examples; a stronger model | A, C |
| Live learning overruns the judging slot | Rehearsals over 75 s | The `DEMO_MODE` cap queues the task; press Load backup and run the rehearsal-learned skill | C |
| Grasps miss on replay because the camera is 1 to 2 cm off | `grasp_missed` in rehearsal logs | Recalibrate; tighten the sim grasp tolerance so sim fails what real would miss; narrower objects | A, B |
| Perception drifts (camera bumped, judging-room lighting) | Check calibration over 1.5 cm | Recalibrate (30 s), add a desk lamp, fall back to taped object spots as known positions | B |
| Generated candidates rarely pass simulation | Under 2 of 6 passing at M1 or M2 | Sharper API notes and examples in the prompt, a stronger model, a simpler demo task, the repair round outside judging | C |
| Daytona slow, over quota or down | Pool start over 60 s, or exec errors | Local subprocess fallback with a clean environment, cards labeled "local"; raise the quota early with Daytona staff | C |
| Venue internet fails (Grok and Daytona both need it) | API timeouts | Phone hotspot, the offline keyword router for the five demo phrases, starter skills planned locally, the backup video | C |
| Voice fails in a loud hall | Misheard commands in a rehearsal near the judging room | Push-to-talk, a headset mic close to the user, the typed box | A |
| The RL policy doesn't transfer to the real arm | Under 4 of 5 at M4 | `use_policy: false`, IK tracking live, RL shown from simulation and labeled | D |
| Claude Code sessions collide or drift from the contracts | Merge conflicts or failing contract tests | One owning lane per file, contracts behind an approval prompt, merges at gates after `/verify` | All |
| Servos overheat or the gripper stalls | Servo hot to the touch, jerky motion, read errors | Gripper hold backoff; between demos, rest the arm at home and use `/torque_off`; light objects | A |
| Demo laptop dies | Battery or thermal warnings | Charger always in. Copy the LeRobot calibration files from `~/.cache/huggingface/lerobot/calibration/` into `config/calibration/` and keep a second laptop set up | A |
| USB ports swap or a cable drops | "Port not found" or read errors | Re-run `lerobot-find-port`, label and tape cables; on Linux, add udev symlinks | A |
| API credits run out | Usage pages in the xAI console and the Daytona dashboard | Check usage at every gate; the pool deletes its sandboxes on shutdown | C, D |
