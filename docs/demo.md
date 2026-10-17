# Demo script

The demo runs 3 minutes and shows three things: a saved skill runs at once while the hand-off follows a moving hand, a new skill is learned live and saved, and an impossible task is queued honestly. The scene is built so naive plans fail visibly, which is what makes the six-way search worth watching.

**Table setup, layout L1, taped and identical at every rehearsal:**

- Arm clamped at the table edge facing the user; workspace box taped on the table (Safety rules).
- Overhead camera on a clamp or tripod about 65 cm up, behind the arm, opposite the user. Leader arm unplugged.
- Objects with markers on top: pill bottle at (0.18, -0.12), juice box at (0.24, -0.04), tea tin at (0.15, 0.03).
- Tray at (0.30, 0.22), near the edge of the arm's reach, with walls under 4 cm.
- Cereal box (marker 30, about 20 cm tall) at (0.24, 0.11), standing between the objects and the tray. A plan that sweeps low from the objects to the tray hits it; one that rises above it or goes around it passes. A15 tunes its spot until 2 to 4 of 6 candidates pass.
- "User" teammate seated across the table, wearing the marker wristband. Their hand stays in their lap except during hand-offs, the hand-off spot is taped at (0.30, 0.00), and after the hand-off they keep the pill bottle off the table.
- Dashboard on a monitor. Operator at the laptop with a finger on Esc; a third teammate holding the arm's 12 V plug.
- `DEMO_MODE=1`.

| Time | User says or does | What judges see | If it fails |
| --- | --- | --- | --- |
| 0:00–0:10 | Presenter: "Home robots ship with a fixed list of tasks. The help a person needs is specific to their home, so ours learns new tasks from one sentence." | The library shows two skills: `hand_over` and `place_in_tray` | — |
| 0:10–0:45 | User: "Hand me my pills," then slowly slides their hand to a new spot | Library hit; the arm moves within 5 s, picks the pill bottle and follows the moving wristband. The dashboard shows "carry: RL policy" (or "IK" if M4 failed). The gripper opens once the hand stays within 8 cm for 1 second | Voice misheard: the operator types the command. Arm fault: Esc, then skip to the next step |
| 0:45–2:15 | User: "Clear the table. Put everything in the tray." | The success test appears first: "Every object ends inside the tray." Six cards fill within 75 s, and some fail with reasons such as "hit cereal_box". The winner's checked path replays on the real arm, the camera checks the same test, and "Saved: clear_table" appears. The presenter narrates (talk track lines 3 and 4) | No pass within 75 s: "Queued" shows honestly; the operator presses Load backup and the user says it again, which runs from the library. Real run fails: "Not saved" shows, and the next-best candidate is re-planned once |
| 2:15–2:30 | User: "Fold my laundry." | Spoken reply: "I can't do that yet. I've queued it." A queue entry appears | — |
| 2:30–3:00 | Presenter: "Every motion you saw was checked in simulation first, and the library has one more skill than when we started." If time allows, put the two objects back and invite a judge to say "clear the table" | Instant plan from the library; the arm starts within 5 s | Judge's phrasing misrouted: the operator types it |

If judges allow 5 minutes, add the routine: the user says "When I say good morning, hand me my pill bottle, then my juice box," the routine is checked in simulation and saved, and then "Good morning" runs it.

**Judging mode.** `DEMO_MODE=1` turns the repair round off and enforces the budgets in Contracts: 30 s for code writing, 35 s for simulation, and 75 s from command to arm moving, after which the task is queued rather than left hanging in front of judges.

**30-second reset before each judge group:**

1. The arm is home and not stopped (press Resume if it is).
2. Press Reset demo and confirm the library shows exactly two skills.
3. Pill bottle, juice box and tea tin on their tape marks; cereal box and tray on theirs.
4. Press Check calibration. Over 1.5 cm means recalibrate, which takes 30 s.
5. The user's hand in their lap, and the plug holder in place.
