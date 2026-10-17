# Pitch and Devpost

The pitch is one idea: home robots ship with fixed skills, and ours safely learns the user's own tasks from a sentence. Lead with the live learning moment, not the architecture.

**Talk track (60 seconds, presenter):**

1. "Home robots ship with a fixed list of tasks. The help someone with limited mobility needs is specific to their home."
2. "So we built an arm you teach by talking to it."
3. When the success test appears: "Before any code exists, it writes down how it will know it succeeded."
4. While the cards fill: "Six AI agents each write a different program, and Daytona runs every one in its own simulated copy of this table. Watch the ones that hit the cereal box fail."
5. "The arm replays only the winner's exact path, checked for collisions in simulation, and the camera checks the result against the same test before it's saved."
6. "When code isn't enough, Grok writes reward functions and we train policies in parallel. That's how it follows your hand."
7. "Every skill it learns stays. The library is longer now than when we started."

**How we differ**, based on their public writeups:

| Project | What it did | What's different here |
| --- | --- | --- |
| [Baymax](https://devpost.com/software/baymax-your-personal-healthcare-companion), TreeHacks 2024 grand prize | Speech-controlled assistive arm with computer vision that picks up objects and feeds users | Its abilities were fixed at build time; ours grow at use time, and every new skill passes simulation before the arm moves |
| [Robosurge](https://devpost.com/software/we-use-nix), TreeHacks 2026, 2nd in the grand prize | A surgeon commands several arms in natural language, with custom IK, a depth camera and Lean proofs that trajectories stay in safe bounds | It plans each command; we learn reusable skills and keep them, each judged by a success test written before the code |

**Judge questions and answers:**

- *Is it actually learning, or just running LLM code?* "Two kinds of learning: new skills are programs that must pass a success test in simulation and on the real arm before they're kept, and the RL path trains neural policies. We don't claim one model learns everything."
- *What stops the AI from doing something dangerous?* "AI-written code never runs on this laptop. It runs in Daytona sandboxes and hands back only a joint path. Our code checks every point against joint limits, speed and a workspace box before the arm replays it, and STOP freezes the arm in place."
- *How do you know it worked?* "The success test is written from your sentence before any code. The same test scores every simulation and checks the camera after the real run."
- *Couldn't a program just claim it succeeded?* "Programs never grade themselves. The test is separate, written first, and evaluated by our own code."
- *IK could follow a hand. Why RL?* "It could. We chose reaching because it transfers reliably from simulation to this arm, so it honestly shows the training pipeline. The pipeline is the point."
- *Why not a vision-language-action model like SmolVLA?* "Those need recorded demonstrations and GPU training per task. We add a skill in [measured M2 time] with no data collection. VLAs are our next step for dexterous tasks."
- *How long does a new skill take?* Give the median from `runs/`, measured, never estimated.
- *What can't it do?* "Cloth, liquids, unknown objects, and anything whose success we can't state and check. Those get queued, and we showed you one."
- *Does it work on another arm?* "Swap the MuJoCo model and the driver. The agents, sandboxes and library don't change."

**Sponsor challenges.** For Best Use of Daytona, show the six-sandbox fan-out per new skill, the planning sandbox that re-plans every saved skill, and the four-way RL training fan-out, with measured numbers in the README: sandboxes per task, median scoring time, total sandbox runs. For Make It Legendary, the Grok Voice Agent is the product's voice, and the README shows how the project was built with Cursor, with Claude Code running inside it, the shared rules files and screenshots. Say plainly which tool did what.

**Devpost checklist:**

- [ ] Title and tagline, for example "A home robot arm you teach by talking to it."
- [ ] Video, 2 minutes: 20 s problem, 80 s demo footage with the learning moment uncut, 20 s how it works.
- [ ] What it does, How we built it (with this doc's architecture diagram), Challenges, Accomplishments (measured numbers), What we learned, What's next.
- [ ] Built with: Python, LeRobot, SO-101, MuJoCo, Daytona, xAI Grok (Voice Agent API and text API), Cursor, Claude Code, OpenCV, Stable-Baselines3, FastAPI.
- [ ] Public GitHub repo with a README: setup, architecture, where Daytona and Grok run, and how Cursor and Claude Code were used.
- [ ] Challenges selected: Best Use of Daytona, Make It Legendary.
- [ ] Draft submitted by 4:30 AM Sunday; final edits before 7:45 AM.

**Video shot list (Sun 1:00 AM, right after feature freeze):** a wide shot of the table with the dashboard screen-recorded beside it, two clean takes of the full demo, a close-up of the success-test banner and the sandbox cards during learning, the RL simulation video, and 10 seconds of STOP freezing the arm.
