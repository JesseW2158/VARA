# web/: dashboard and voice page

Plain HTML and JavaScript served by the orchestrator (`agents/orchestrator.py`). No build step,
no framework. Rules: `.claude/rules/web.md`. Tasks: S0 item 6, B9 (`docs/tasks/B-vision.md`),
A14 (`docs/tasks/A-arm.md`). Tests: `tests/test_web_pages.py` (static checks plus the node harness in
`tests/web/`).

| File | What it does |
| --- | --- |
| `web/index.html` | The dashboard judges watch, served at `GET /`. Layout only; all behavior is in `app.js`. |
| `web/static/app.js` | Dashboard logic: feeds, command box, event timeline over `WS /events` (with reconnect and replay de-duplication), the success-test banner, six candidate cards, the 75 s demo countdown, skill library, queue, RL panel, STOP/Esc, Resume, Check calibration, `GET /state` polling every 2 s, request deadlines, and forwarding of the T key (and a window blur during a press) to the voice iframe. |
| `web/static/style.css` | Dark theme sized to read from 2 m. Color tokens live on `:root`. |
| `web/voice.html` | The push-to-talk voice page, served at `GET /voice` and embedded in the dashboard iframe. Page-specific CSS is inline. |
| `web/static/voice.js` | Grok Voice Agent client: token from `POST /voice/token`, realtime WebSocket, push to talk, playback, the three tools (`do_task`, `stop`, `list_skills`) and end-of-task summaries from `WS /events`. |
| `web/static/rl/.gitkeep` | Keeps the folder where `rl/train_parallel.py` writes `curves.json`. |
| `web/static/rl/curves.json` | Written by Lane D (not committed): `{"variants": [{"name", "steps": [...], "success": [...]}], "chosen": "<name>"}`. `success` may be 0..1 or 0..100. Missing file = "No RL curves yet". Entries that are not objects, and values that are not numbers, are skipped. |
| `web/static/rl_sim.mp4` | Written by Lane D (D6, not committed). Missing file = "no simulation video yet". |

## Endpoints the pages use

Everything comes from the contract table in `docs/interfaces.md` ("Orchestrator API") plus the
serving conventions agreed in the implementation brief (section 8.6 for the reply shapes):

- `POST /command {"text"}` -> `{"task_id"}` or 409 `{"detail": "busy"}` (dashboard command box, voice `do_task`)
- `POST /stop` -> `{"state": "stopped", "task_id"}` (STOP button, Esc on either page, voice `stop` tool);
  `POST /resume` -> `{"state": "idle"}`. The page only needs the HTTP status; the next `/state` poll
  shows the server's own state. A `/stop` that gets no answer within 3 s (the server freezes within
  0.2 s) is aborted: the headline says so and tells the human to pull the 12 V plug if the arm is
  moving, the request is re-sent three times, and the STOP button and Esc stay live throughout (one
  request in flight at a time, never a stuck one).
- `POST /calibrate` -> `{"error_cm"}` (shown in cm, red above 1.5 cm)
- `POST /reset_demo`, `POST /load_backup` -> `{"skills": [names]}` (library buttons; the names are
  listed in the result line. A bare list, a list of skill objects, or the older `loaded`/`restored` keys
  are read the same way.)
- `POST /voice/token` -> `{"token", "expires_at", "model"}` (voice page only; never an API key)
- `GET /state` -> `{"state", "mode", "carry", "task_id", "skills", "queue", "stopped"}` polled every 2 s
  with one request in flight at a time and an 8 s deadline, so a hung server cannot pile up polls and
  starve STOP of a connection (Chromium allows six per host and the two MJPEG feeds hold two);
  `skills` is a list of `schemas/skill.json` objects (entries that are not objects are dropped, so a
  stray `null` never breaks the library); `queue` entries are `{"task", "reason", "task_id"}` (a plain
  string is shown as is); voice `list_skills` reads the names from it. A failed or timed-out fetch shows
  "orchestrator unreachable"; an exception inside our own rendering shows "render error: ..." instead,
  on every poll until it is fixed (nothing is cached as rendered until a render completes).
- `WS /events` -> replays the last 200 `schemas/event.json` events, then streams
- `GET /video/overhead`, `GET /video/wrist` -> MJPEG `<img>` sources with a "no feed" fallback
- `GET /static/rl/curves.json`, `GET /static/rl_sim.mp4` -> RL panel (both optional; one curves request
  in flight at a time, 30 s deadline)
- `POST /torque_off` -> the muted "Torque off" maintenance button at the far right of the top bar. It is
  disabled while a task runs, takes two clicks within 5 s (the arm sags and drops what it holds), and
  shows the orchestrator's 409 refusal when a task is running. STOP never cuts torque.

Every other request (`/resume`, `/calibrate`, the library buttons, `/torque_off`) has a 30 s deadline;
`/resume` answers only after the arm has reached home.

## Events and the candidate cards

Events the dashboard understands (all detail fields optional): `routing {text}`, `spec_ready {spec}`,
`writing_code {candidate, hint}`, `candidate_rejected {candidate, problems}`, `simulating {}`,
`candidate_scored {score fields..., runner, hint, trial_results}`, `selected {candidate, hint, scores}`,
`planning`, `validating`, `executing`, `verifying`, `saved {name}`, `not_saved {reason}`,
`queued {task, reason}`, `error {message}`, `stopped {}`, `done {}`.

- `candidate` names a card only when it is a whole number from 1 to 6 (the schema's integer; a
  digit-only string such as `"2"` is tolerated). `1.7`, `"1.7"`, `true`, `[1]`, `0` or `7` name no card:
  a `candidate_scored` or `selected` carrying one changes nothing, and a `writing_code` without a usable
  index updates all six cards, as before. (`parseInt` used to read `1.7` as card 1.)
- Per-trial ticks and crosses come only from `trial_results` (a list of booleans, brief 8.4/8.6; the
  alias keys `trials_detail`, `per_trial`, `trial_successes`, `results` and entries shaped `{success}`
  are tolerated). Without it the row shows what the score really says: trial 1's mark from
  `nominal_success` and the count ("4/5"), never invented positions. A `candidate_scored` detail that
  carries only `{candidate, passed, fail_reason}` (the contract fixture) still renders.
- `runner: "local"` shows the yellow "local" badge on its own row under the header. The status
  text ("scored", "rejected by check_source", ...) never wraps inside itself: it sits beside the
  candidate number when it fits and drops to its own line otherwise. Measured at a 1366 px viewport:
  160 px inside the card, the longest status 144 px at the 13 px lowercase style.
- A `candidate_scored` whose `error` starts with `rejected:` (the sandbox runner's own `check_source`
  refusal, brief 8.4: `"rejected: <first problem>"`) shows the status "rejected by check_source" with
  the dashed border. Any other error text, a bare `rejected` or `Rejected by the validator ...`
  included, is an ordinary FAIL with the text as its reason.
- Text written by the server or the user is cut for display with the full text in the element's `title`
  (hover): `fail_reason`, `error`, the rejection problems and the strategy hint at 200 characters on the
  card; every field of a timeline line at 200 (the uncut line is the line's title); the headline at
  300; the banner's `english` at 300 (the full spec JSON sits right below it). Text that is one unbroken
  token (no spaces) wraps inside its box (`overflow-wrap: anywhere` on every box that shows such text),
  so it can never widen the page.
- The DEMO countdown is clamped to 0..75 s and the elapsed labels to 0 s or more, so a server clock
  ahead of or behind the browser never shows "125 s" or "arm moving at -49 s".

## Chart.js: pinning and updating

The RL panel loads Chart.js from cdnjs at an exact version with a subresource-integrity hash:

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"
        integrity="sha384-dug+JxfBvklEQdJ4AYuBBAIScUz0bVN73xpy273gcAwHjb3qI0fXmuYNaNfdyYJG"
        crossorigin="anonymous" defer></script>
```

`defer` keeps the download off the parser's path, but `DOMContentLoaded` still waits for a deferred
script to finish downloading. `app.js` is a plain script at the end of the body and starts the moment
it is parsed (every element is above it); it never waits for `DOMContentLoaded`. Measured with the CDN
request held open for 29 s: an earlier version that waited for that event showed no cards and answered
neither the STOP button nor Esc for the whole stall. `loadRl()` checks `typeof Chart` on every poll:
while it is missing the panel prints the final success rate of each variant as text, re-checks every
2 s for the first minute, then every 30 s with the normal poll, and draws the chart as soon as `Chart`
exists. Nothing is cached as "failed".

The unit of the `success` series (0..1 or 0..100) is decided by the maximum over every variant, found
with `reduce`. `Math.max.apply(null, values)` passed every value as a call argument and overflowed the
call stack on a 5 MB `curves.json` (4 x 100 000 points), which lost the chart; the node harness replays
that file.

To move to another version (pick one that has existed for at least two weeks):

1. Check the file exists: `curl -sI https://cdnjs.cloudflare.com/ajax/libs/Chart.js/<ver>/chart.umd.js`
   (the same bytes are at `https://cdn.jsdelivr.net/npm/chart.js@<ver>/dist/chart.umd.js`).
2. Recompute the hash: `curl -s <url> | openssl dgst -sha384 -binary | openssl base64 -A`.
3. Replace the version and the `integrity` value in `web/index.html`, and update this file.
4. Open the dashboard with a `curves.json` present and confirm the four curves draw.

If the CDN is unreachable or half-open (venue Wi-Fi), the page still works: the RL panel keeps the
text summary, and nothing else depends on Chart.js. Never load it from an unpinned URL
(`chart.js@latest`): a breaking release during the demo is not a risk worth taking.

## Voice page settings

- WebSocket: `wss://api.x.ai/v1/realtime?model=<model from the token reply>` with the subprotocol
  `xai-client-secret.<token>` (the xAI docs' browser pattern; headers cannot be set from a browser).
- `session.update`: `voice: "eve"` (the `VOICE` constant at the top of `voice.js`; xAI wants lowercase
  voice ids), the exact instructions sentence from `docs/interfaces.md` "Voice tools", the three
  function tools in `{type: "function", name, description, parameters}` form, `turn_detection: null`
  (push to talk), `audio.input/output.format = {type: "audio/pcm", rate: 24000}`.
- Push to talk: hold `T` or the button. Audio is captured with `getUserMedia`, resampled to 24 kHz if the
  `AudioContext` refused that rate, sent as PCM16 little-endian base64 in ~100 ms
  `input_audio_buffer.append` messages, then `input_audio_buffer.commit` and `response.create` on
  release. Presses under 100 ms are discarded with `input_audio_buffer.clear`.
- T works wherever the focus is. When the dashboard window has the focus, `app.js` forwards T
  keydown/keyup to the iframe as `postMessage({type: "vara-key", event, key: "t", repeat})` (same
  origin only). A keydown while typing in a text field, or with Ctrl, Alt or Meta held (Ctrl+T opens a
  tab), is not forwarded; a keyup always is, so a press never sticks; and when the dashboard window
  loses the focus while a forwarded press is still down (its keyup will never arrive), one
  `event: "blur"` is forwarded. The voice page accepts the messages only from its parent on the same
  origin and handles them exactly like its own key events (`blur` like a keyup); its own keydown
  handler ignores Ctrl/Alt/Meta+T and key repeats the same way.
- Replies: `response.output_audio.delta` (base64 PCM16 24 kHz) are scheduled gaplessly through Web Audio.
  A delta that is not valid base64 is dropped with one "err" log line per burst instead of throwing.
- Tool calls: `response.function_call_arguments.done` -> our endpoint -> `conversation.item.create`
  `{type: "function_call_output", call_id, output}` -> `response.create` once every pending call has
  answered and the current reply has finished playing. `arguments` must parse to a JSON object;
  anything else gets the output `error: invalid arguments (not JSON)` (or `not a JSON object`) and is
  never sent to `POST /command`. A call without a `call_id` is logged and ignored. `do_task` answers
  `On it` (or `busy. One moment.` on 409); when `WS /events` shows `saved`/`not_saved`/`queued`/
  `stopped`/`error`/`done` for that task a user text item (`Robot update: Done. Saved clear table.`) is
  sent, then `response.create`. The started tasks live in a map without a prototype
  (`Object.create(null)`) read through an own-property check, so an event whose `task_id` is
  `__proto__` or `toString` matches no task and cannot touch `Object.prototype`.
- `Esc` and the STOP button `POST /stop` from the voice page too, and tell the parent dashboard through
  `postMessage` so it shows STOPPED at once.
- A token starting with `fake-` (the orchestrator's `FAKE_LLM` reply) disables the xAI connection with a
  visible notice instead of hammering the API with an invalid secret.
- `voice.js` is also a plain script at the end of its body and starts at once, like `app.js`.

## Checking the pages

- `pytest -q tests/test_web_pages.py`: both HTML files parse with unique ids, every file under `web/`
  is plain ASCII and holds no key, both scripts start without waiting for `DOMContentLoaded`, the
  Chart.js tag is deferred and pinned, the text boxes wrap unbroken tokens, the card status keeps
  `white-space: nowrap` with the badge row beside it, the chart extremes use `reduce`, the voice task
  map has no prototype, and (when `node` is on the PATH) `tests/web/app_checks.mjs` runs `app.js` and
  `voice.js` in a small fake DOM: start-up with a stalled deferred script, a junk `curves.json` with and
  without Chart.js, Chart.js arriving late, a 5 MB `curves.json`, a hung and a failing `/stop`, a hung
  `/state`, 5000-character texts, trial rows and the "rejected:" label, non-integer candidate indices,
  the DEMO countdown under clock skew, junk `/state` entries and a throwing render, T forwarding to the
  voice iframe, and on the voice page the realtime handshake, tool calls (no `call_id`, non-JSON
  arguments, 409), task announcements with prototype-named task ids, junk audio deltas and raw frames,
  and the forwarded key and blur messages. Nothing touches the network.
- Syntax by hand: `node --check web/static/app.js && node --check web/static/voice.js`
- Plain ASCII by hand: `grep -rnIP "[^\x00-\x7F]" web/` must print nothing (glyphs such as the ticks are
  written as `\u` escapes in the scripts; `-I` skips the binary `rl_sim.mp4` that Lane D writes here).
- Secrets by hand: grep `web/` (case-insensitively) for the two key variable names in `.env.example`,
  for the HTTP authorization scheme word followed by a space, for `sk-` and for `xai-` followed by ten
  alphanumerics; every one must print nothing. Keys live only in the server's `.env`.
- Live: `FAKE=1 uvicorn agents.orchestrator:app --port 8000`, open `http://localhost:8000`, type
  "put the juice box in the tray", watch six cards fill, press Esc and confirm STOPPED. At a 1366 px
  window every card status stays on one line with the "local" badge under it.
