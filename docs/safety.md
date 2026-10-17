# Safety rules

The arm works next to a seated person's hand, so these limits live in Lane A's code and in Claude Code's permissions, not in good intentions. STOP freezes the arm in place; only the power plug cuts it.

- **Workspace box:** x 0.10 to 0.40 m, y -0.25 to 0.25 m, z 0.02 to 0.30 m. `validate` rejects any waypoint whose tip leaves it, and the box is taped on the table.
- **Speed:** joints at most 60°/s, the tip at most 0.2 m/s, and at most 0.1 m/s within 15 cm of the tracked hand. Bridging moves and Resume run at 30°/s at most.
- **Height and distance:** the tip never goes above 0.30 m, and the user's face stays at least 40 cm from the taped box.
- **Hand-offs:** only to a hand resting on the table, never toward the face or body. Outside `carry_to_hand`, a wristband seen within 10 cm of the tip freezes the arm, and the gripper never closes within 10 cm of the wristband.
- **STOP:** Esc anywhere on the dashboard, the STOP button, or saying "stop". STOP freezes the arm with torque on and cancels the task, so a held object never drops onto a hand. Voice is a convenience; Esc and the button are the safety path.
- **Hard kill:** a teammate holds the arm's 12 V plug during every demo. Pulling it cuts torque and the arm sags under its own weight, so keep hands out from under it.
- **Grip and objects:** hold at the contact reading minus 3, never stalled at full close. No sharp, hot, heavy or fragile objects; everything under 100 g, and the pill bottle is an empty prop.
- **AI-written code:** never runs on the laptop or touches the arm. The arm replays only paths that passed simulation with zero collisions and then passed `validate`, and no spoken words go straight to the motors.
- **Claude Code:** commands with `--real` need human approval (`.claude/settings.json`), given only after checking the arm is clear. The leader arm stays unplugged during demos.
- **Judges:** hands stay outside the taped box unless they are doing the hand-off.
- **Drill:** before every demo, press Esc once mid-move and confirm the arm freezes and holds.
