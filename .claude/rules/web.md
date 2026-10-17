---
paths:
  - "web/**"
---
# Dashboard and voice rules
- Plain HTML and JavaScript served by the orchestrator, no build step. Chart.js from a CDN at a pinned version.
- Esc anywhere on the page sends POST /stop.
- Never put an API key in the page. The voice page gets a token from POST /voice/token.
- Voice is push-to-talk: turn_detection is null.
