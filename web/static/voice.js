/* VARA voice page (task A14). Plain JavaScript, no build step.
 *
 * Flow (docs/interfaces.md "Voice tools", docs/tasks/A-arm.md A14, xAI Voice Agent API):
 *   1. POST /voice/token on our orchestrator -> {token, expires_at, model}. The xAI API key
 *      never reaches the browser; the token is a 5-minute client secret.
 *   2. new WebSocket("wss://api.x.ai/v1/realtime?model=" + model, ["xai-client-secret." + token])
 *      (browsers cannot set headers, so the token travels as the subprotocol).
 *   3. session.update: voice "eve", the contract instructions, the three function tools,
 *      turn_detection null (push to talk), audio/pcm at 24 kHz in and out.
 *   4. Hold T or the button: microphone -> PCM16 little-endian base64 chunks of ~100 ms as
 *      input_audio_buffer.append; on release input_audio_buffer.commit then response.create.
 *   5. response.output_audio.delta -> queued Web Audio playback.
 *   6. response.function_call_arguments.done -> call our endpoint -> conversation.item.create
 *      {type: "function_call_output", call_id, output} -> response.create.
 *      do_task answers "On it" at once; when WS /events reports the end stage for that task
 *      a user text item with the summary is sent, then response.create. Arguments that are not
 *      a JSON object get an "error: invalid arguments" output and never reach POST /command.
 *   7. Inside the dashboard iframe, T presses forwarded by the parent window as postMessage
 *      {type: "vara-key", event: "keydown"|"keyup"|"blur"} drive push to talk like the local keys
 *      ("blur": the dashboard window lost the focus while a forwarded press was down).
 *   8. STOP (Esc, the button or the `stop` tool) posts {source, t_client_ms: the wall clock at the
 *      press} to /stop and logs one console line "stop voice: server <latency_ms> ms, round trip <ms>
 *      ms, press to freeze <ms> ms" from the reply's latency_ms and press_to_freeze_ms (the server's
 *      request-to-freeze and press-to-freeze times; an older server reports neither) and the round
 *      trip measured here (from the key or click event's timeStamp, or from the tool call). A `stop`
 *      call that is the first tool call after a spoken turn adds ", talk release to response <ms> ms":
 *      push to talk released to the /stop reply, the "saying stop" latency. Inside the dashboard
 *      iframe the numbers travel with the {type: "vara-stopped", source, latency_ms,
 *      press_to_freeze_ms, round_trip_ms, release_to_response_ms} message, so the dashboard timeline
 *      shows them on the entry of the stop.
 */
(function () {
  "use strict";

  var REALTIME_URL = "wss://api.x.ai/v1/realtime";
  var DEFAULT_MODEL = "grok-voice-latest";
  var VOICE = "eve"; // lowercase voice id, as the xAI docs ask
  var SAMPLE_RATE = 24000;
  var CHUNK_MS = 100;
  var MIN_PRESS_S = 0.1;
  var MAX_RECONNECTS = 3;
  var LOG_MAX = 200;
  var PRESS_MAX_AGE_MS = 60000; // an input event's timeStamp further back than this is not taken as the press
  var RELEASE_MAX_MS = 60000; // a talk release older than this is not the turn a tool call answers

  // The exact sentence from docs/interfaces.md, "Voice tools".
  var INSTRUCTIONS = "You are the voice of a home robot arm. For any physical request, call do_task with the user's exact words. If the user says stop, call stop immediately. If do_task reports busy, say 'One moment.' Keep spoken replies under 12 words.";

  var TOOLS = [
    {
      type: "function",
      name: "do_task",
      description: "Send a physical request to the robot arm. Pass the user's exact words. Returns 'On it' when accepted or 'busy' when a task is already running.",
      parameters: {
        type: "object",
        properties: { text: { type: "string", description: "The user's words, verbatim." } },
        required: ["text"]
      }
    },
    {
      type: "function",
      name: "stop",
      description: "Freeze the robot arm in place immediately and cancel the current task.",
      parameters: { type: "object", properties: {} }
    },
    {
      type: "function",
      name: "list_skills",
      description: "List the names of the robot's saved skills so they can be read aloud.",
      parameters: { type: "object", properties: {} }
    }
  ];

  function $(id) { return document.getElementById(id); }
  function inIframe() { try { return window.parent && window.parent !== window; } catch (e) { return true; } }
  function humanize(s) { return String(s === undefined || s === null ? "" : s).replace(/_/g, " "); }
  function short(s, n) { s = String(s === undefined || s === null ? "" : s); return s.length > n ? s.slice(0, n - 1) + "\u2026" : s; }
  function clock() { return new Date().toTimeString().slice(0, 8); }

  function num(v) {
    if (v === null || v === undefined || v === "" || typeof v === "boolean") return null;
    var n = Number(v);
    return isFinite(n) ? n : null;
  }

  // Milliseconds for latency measurements: the browser's monotonic clock, or Date.now() without one.
  function nowMs() {
    return (typeof performance === "object" && performance !== null && typeof performance.now === "function") ? performance.now() : Date.now();
  }

  // When a STOP was pressed, as app.js pressStamp: `at` on the nowMs() clock and `wall` on the wall
  // clock (the body's t_client_ms). A key or click event's timeStamp is the press when it is a
  // plausible reading of that clock, so time this page spent busy before the handler ran counts.
  function pressStamp(ev) {
    var now = nowMs();
    var wall = Date.now();
    var ts = ev && typeof ev.timeStamp === "number" ? ev.timeStamp : NaN;
    if (isFinite(ts) && ts > 0 && ts <= now && now - ts <= PRESS_MAX_AGE_MS) return { at: ts, wall: wall - (now - ts) };
    return { at: now, wall: wall };
  }

  // A duration in ms with at most one decimal ("2.5", "37"), never negative.
  function fmtMs(v) { return String(Math.max(0, Math.round(v * 10) / 10)); }

  // The one console line per answered STOP, the same shape as the dashboard's (app.js stopLine): the
  // server's press-to-freeze time is shown only when it fits inside the round trip measured here, and
  // a spoken stop adds the time from the push-to-talk release to the /stop reply (releaseMs).
  function stopLine(source, reply, roundTripMs, releaseMs) {
    var d = reply && typeof reply === "object" ? reply : {};
    var latency = num(d.latency_ms);
    var press = num(d.press_to_freeze_ms);
    var line = "stop " + source + ": " + (latency === null ? "server latency not reported" : "server " + fmtMs(latency) + " ms")
      + ", round trip " + fmtMs(roundTripMs) + " ms";
    if (press !== null && press >= 0 && press <= roundTripMs + 5) line += ", press to freeze " + fmtMs(press) + " ms";
    if (typeof releaseMs === "number" && isFinite(releaseMs) && releaseMs >= 0) line += ", talk release to response " + fmtMs(releaseMs) + " ms";
    return line;
  }

  // ------------------------------------------------------------------ UI
  function setStatus(text, cls) {
    var p = $("voice-status");
    p.textContent = text;
    p.className = "pill" + (cls ? " " + cls : "");
  }

  function setError(text) {
    var box = $("voice-error");
    if (!text) { box.hidden = true; box.textContent = ""; return; }
    box.hidden = false;
    box.textContent = text;
  }

  function setLevel(rms) {
    var pct = Math.min(100, Math.round(rms * 400));
    $("voice-level").style.width = pct + "%";
  }

  var lastAssistantLi = null;

  function log(kind, text) {
    var list = $("voice-log");
    var li = document.createElement("li");
    li.className = kind;
    var when = document.createElement("span");
    when.className = "when";
    when.textContent = clock();
    var body = document.createElement("span");
    var who = document.createElement("span");
    who.className = "who";
    who.textContent = kind === "assistant" ? "grok" : kind === "user" ? "you" : kind;
    body.appendChild(who);
    var txt = document.createElement("span");
    txt.className = "text";
    txt.textContent = text;
    body.appendChild(txt);
    li.appendChild(when);
    li.appendChild(body);
    list.appendChild(li);
    while (list.children.length > LOG_MAX) list.removeChild(list.firstChild);
    list.scrollTop = list.scrollHeight;
    return li;
  }

  function appendAssistant(delta) {
    if (!delta) return;
    if (!lastAssistantLi) lastAssistantLi = log("assistant", "");
    var t = lastAssistantLi.querySelector(".text");
    t.textContent += delta;
    $("voice-log").scrollTop = $("voice-log").scrollHeight;
  }

  function closeAssistantLine() { lastAssistantLi = null; }

  function setTalkEnabled(on) {
    $("btn-talk").disabled = !on;
  }

  function setTalkActive(on) {
    $("btn-talk").classList.toggle("active", on);
    $("btn-talk").textContent = on ? "Listening... release to send" : "Hold to talk (T)";
  }

  // ------------------------------------------------------------ connection
  var ws = null;
  var sessionReady = false;
  var userClosed = false;
  var reconnects = 0;
  var reconnectTimer = null;
  var model = DEFAULT_MODEL;

  function send(msg) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      if (msg.type !== "input_audio_buffer.append") log("err", "not connected, dropped " + msg.type);
      return false;
    }
    ws.send(JSON.stringify(msg));
    return true;
  }

  function sessionUpdate() {
    return {
      type: "session.update",
      session: {
        instructions: INSTRUCTIONS,
        voice: VOICE,
        turn_detection: null,
        tools: TOOLS,
        audio: {
          input: { format: { type: "audio/pcm", rate: SAMPLE_RATE } },
          output: { format: { type: "audio/pcm", rate: SAMPLE_RATE } }
        }
      }
    };
  }

  function fmtExpiry(v) {
    if (v === undefined || v === null) return "";
    var d = typeof v === "number" ? new Date(v * (v > 1e12 ? 1 : 1000)) : new Date(v);
    return isNaN(d.getTime()) ? String(v) : d.toTimeString().slice(0, 8);
  }

  async function connect() {
    clearTimeout(reconnectTimer);
    userClosed = false;
    setError("");
    setStatus("fetching token", "off");
    $("btn-connect").disabled = true;
    var tok;
    try {
      var r = await fetch("/voice/token", { method: "POST" });
      if (!r.ok) {
        var body = "";
        try { body = await r.text(); } catch (e) { /* ignore */ }
        throw new Error("POST /voice/token returned HTTP " + r.status + (r.status === 503 ? " (is the xAI key set in the server's .env?)" : "") + (body ? ": " + short(body, 140) : ""));
      }
      tok = await r.json();
    } catch (e) {
      $("btn-connect").disabled = false;
      setStatus("no token", "off");
      setError("Voice token unavailable: " + e.message + ". Typed commands on the dashboard still work.");
      log("err", "token fetch failed: " + e.message);
      return;
    }
    if (!tok || typeof tok.token !== "string" || !tok.token) {
      $("btn-connect").disabled = false;
      setStatus("bad token reply", "off");
      setError("The token reply had no 'token' field: " + short(JSON.stringify(tok), 120));
      return;
    }
    model = tok.model || DEFAULT_MODEL;
    $("voice-model").textContent = model + (tok.expires_at ? ", token expires " + fmtExpiry(tok.expires_at) : "");
    if (/^fake-/.test(tok.token)) {
      $("btn-connect").disabled = false;
      setStatus("fake token, voice off", "off");
      setError("The orchestrator is in FAKE_LLM mode and returned a fake token, so this page does not connect to xAI. Use the typed command box.");
      log("sys", "fake token received; not connecting");
      return;
    }
    openSocket(tok.token);
  }

  function openSocket(token) {
    var url = REALTIME_URL + "?model=" + encodeURIComponent(model);
    var sock;
    try {
      sock = new WebSocket(url, ["xai-client-secret." + token]);
    } catch (e) {
      $("btn-connect").disabled = false;
      setStatus("websocket failed", "off");
      setError("Could not open the realtime WebSocket: " + e.message);
      return;
    }
    ws = sock;
    sessionReady = false;
    setStatus("connecting to xAI", "off");
    sock.onopen = function () {
      reconnects = 0;
      setStatus("connected, configuring session", "on");
      $("btn-disconnect").disabled = false;
      send(sessionUpdate());
      log("sys", "connected to " + url);
    };
    sock.onmessage = function (m) {
      var ev;
      try { ev = JSON.parse(m.data); } catch (e) { return; }
      handleServer(ev);
    };
    sock.onerror = function () {
      setError("Realtime WebSocket error: the token was rejected or the network is down. Check the orchestrator log and the server's xAI key.");
    };
    sock.onclose = function (ev) {
      if (ws !== sock) return;
      ws = null;
      sessionReady = false;
      stopTalking(true);
      setTalkEnabled(false);
      setTalkActive(false);
      $("btn-disconnect").disabled = true;
      $("btn-connect").disabled = false;
      setStatus("disconnected", "off");
      if (userClosed) { log("sys", "disconnected"); return; }
      log("err", "connection closed, code " + ev.code + (ev.reason ? ": " + ev.reason : ""));
      if (reconnects < MAX_RECONNECTS) {
        reconnects += 1;
        var wait = 2000 * Math.pow(2, reconnects - 1);
        setStatus("reconnecting in " + Math.round(wait / 1000) + " s", "off");
        reconnectTimer = setTimeout(connect, wait);
      } else {
        setError("Voice connection lost " + MAX_RECONNECTS + " times. Press Connect to try again.");
      }
    };
  }

  function disconnect() {
    userClosed = true;
    clearTimeout(reconnectTimer);
    stopTalking(true);
    stopPlayback();
    if (ws) { try { ws.close(1000, "user"); } catch (e) { /* ignore */ } }
  }

  // ---------------------------------------------------------- server events
  function handleServer(ev) {
    if (!ev || typeof ev !== "object") return;
    switch (ev.type) {
      case "conversation.created":
      case "session.created":
        break;
      case "session.updated":
        sessionReady = true;
        setTalkEnabled(true);
        setStatus("ready: hold T to talk", "on");
        log("sys", "session configured (voice " + VOICE + ", push to talk)");
        break;
      case "response.output_audio.delta":
      case "response.audio.delta":
        playDelta(ev.delta);
        break;
      case "response.output_audio_transcript.delta":
      case "response.audio_transcript.delta":
        appendAssistant(ev.delta);
        break;
      case "response.output_audio_transcript.done":
      case "response.audio_transcript.done":
        if (ev.transcript && lastAssistantLi) lastAssistantLi.querySelector(".text").textContent = ev.transcript;
        break;
      case "response.function_call_arguments.done":
        handleFunctionCall(ev);
        break;
      case "response.done":
        closeAssistantLine();
        if (sessionReady && !capturing) setStatus("ready: hold T to talk", "on");
        break;
      case "conversation.item.added":
      case "conversation.item.created":
        logUserTranscript(ev.item);
        break;
      case "conversation.item.input_audio_transcription.completed":
        if (ev.transcript) log("user", ev.transcript);
        break;
      case "input_audio_buffer.committed":
        break;
      case "error": {
        var e = ev.error || {};
        var msg = e.message || e.code || JSON.stringify(ev);
        setError("xAI error: " + msg);
        log("err", msg);
        break;
      }
      default:
        break;
    }
  }

  function logUserTranscript(item) {
    if (!item || item.role !== "user" || !Array.isArray(item.content)) return;
    for (var i = 0; i < item.content.length; i++) {
      var c = item.content[i];
      if (c && c.type === "input_audio" && c.transcript) { log("user", c.transcript); return; }
    }
  }

  // ------------------------------------------------------------ tool calls
  var pendingCalls = 0;
  var pendingTasks = Object.create(null); // no prototype: a task id can never reach Object.prototype
  var lastStopAt = 0;
  // The end of the last spoken turn: nowMs() when push to talk was released and the audio committed.
  // The first tool call after it is the model's answer to that turn and takes it (takeRelease), so a
  // `stop` call reports the time from the end of the user's "stop" to the /stop reply.
  var lastRelease = null;

  function takeRelease() {
    var at = lastRelease;
    lastRelease = null;
    return at !== null && nowMs() - at <= RELEASE_MAX_MS ? at : null;
  }

  // Arguments must be a JSON object (the API sends them as a JSON string). Anything else is answered
  // with an error output so the model can retry; it is never turned into a command for the arm.
  function parseArguments(raw) {
    if (raw === undefined || raw === null || raw === "") return { args: {}, error: null };
    var v = raw;
    if (typeof v === "string") {
      try { v = JSON.parse(v); } catch (e) { return { args: null, error: "not JSON" }; }
    }
    if (!v || typeof v !== "object" || Array.isArray(v)) return { args: null, error: "not a JSON object" };
    return { args: v, error: null };
  }

  async function handleFunctionCall(ev) {
    var name = typeof ev.name === "string" ? ev.name : "";
    var callId = ev.call_id;
    if (!callId) { log("err", "function call " + (name || "?") + " without a call_id, ignored"); return; }
    var releasedAt = takeRelease();
    var parsed = parseArguments(ev.arguments);
    var args = parsed.args || {};
    pendingCalls += 1;
    log("tool", name + " " + (parsed.error ? short(String(ev.arguments), 80) : JSON.stringify(args)));
    var output;
    try {
      if (parsed.error) output = "error: invalid arguments (" + parsed.error + ")";
      else if (name === "do_task") output = await doTask(args);
      else if (name === "stop") output = await doStop("voice", null, releasedAt);
      else if (name === "list_skills") output = await listSkills();
      else output = "error: unknown tool " + name;
    } catch (e) {
      output = "error: " + e.message;
    }
    log("tool", name + " -> " + output);
    send({ type: "conversation.item.create", item: { type: "function_call_output", call_id: callId, output: output } });
    pendingCalls -= 1;
    if (pendingCalls === 0) afterPlayback(function () { send({ type: "response.create" }); });
  }

  async function doTask(args) {
    var text = args && typeof args.text === "string" ? args.text.trim() : "";
    if (!text) return "error: do_task needs the user's words in 'text'";
    var r = await fetch("/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text }) });
    if (r.status === 409) return "busy. One moment.";
    if (!r.ok) return "error: POST /command returned HTTP " + r.status;
    var d = null;
    try { d = await r.json(); } catch (e) { d = null; }
    var tid = d && typeof d.task_id === "string" ? d.task_id : "";
    if (tid) pendingTasks[tid] = { text: text, announced: false };
    setStatus("task " + (tid || "") + " running", "on");
    return "On it";
  }

  // STOP from Esc, the button or the `stop` tool. `source` is the label shown ("Esc", "button",
  // "voice"); on the wire it is lowercase, as the server names it. `press` (pressStamp of the key or
  // click event; the tool call's own moment when absent) holds the press on the wall clock for the
  // server (press_to_freeze_ms on the same laptop) and on the monotonic clock for the round trip
  // measured here; `releasedAt` is the end of the spoken turn a `stop` call answers. With the reply's
  // latency_ms they make the one console line, go on the voice log line and, inside the dashboard
  // iframe, travel with the "vara-stopped" message.
  async function doStop(source, press, releasedAt) {
    lastStopAt = Date.now();
    if (!press) press = pressStamp(null);
    var wire = String(source).toLowerCase();
    var r;
    try {
      // Nothing is awaited ahead of this request: it must leave as fast as it always has.
      r = await fetch("/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source: wire, t_client_ms: press.wall }) });
    } catch (e) {
      setError("STOP request failed: " + e.message + ". Use the dashboard STOP or pull the 12 V plug.");
      return "error: " + e.message;
    }
    var answeredAt = nowMs();
    var roundTrip = answeredAt - press.at;
    var sinceRelease = typeof releasedAt === "number" ? answeredAt - releasedAt : null;
    var d = null;
    var line = null;
    if (r.ok) {
      try { d = await r.json(); } catch (e) { d = null; }
      if (!d || typeof d !== "object") d = null;
      line = stopLine(wire, d, roundTrip, sinceRelease);
      if (window.console && console.log) console.log(line);
    }
    if (inIframe()) {
      try {
        window.parent.postMessage({
          type: "vara-stopped",
          source: source,
          latency_ms: d ? num(d.latency_ms) : null,
          press_to_freeze_ms: d ? num(d.press_to_freeze_ms) : null,
          round_trip_ms: r.ok ? roundTrip : null,
          release_to_response_ms: r.ok ? sinceRelease : null
        }, location.origin);
      } catch (e) { /* ignore */ }
    }
    setStatus("STOPPED", "stopped");
    log("sys", "STOP sent (" + source + ")" + (line ? ": " + line : ""));
    return r.ok ? "stopped" : "error: POST /stop returned HTTP " + r.status;
  }

  async function listSkills() {
    var r = await fetch("/state");
    if (!r.ok) return "error: GET /state returned HTTP " + r.status;
    var d = await r.json();
    var names = (Array.isArray(d.skills) ? d.skills : [])
      .map(function (s) { return s && s.name ? humanize(s.name) : null; })
      .filter(Boolean);
    return names.length ? "Skills: " + names.join(", ") : "No skills saved yet";
  }

  // ------------------------------------------- task end stages from WS /events
  var evSock = null;
  var evRetryMs = 1000;
  var evTimer = null;

  function connectEvents() {
    clearTimeout(evTimer);
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    var sock;
    try { sock = new WebSocket(proto + "//" + location.host + "/events"); }
    catch (e) { evTimer = setTimeout(connectEvents, 5000); return; }
    evSock = sock;
    sock.onopen = function () { evRetryMs = 1000; };
    sock.onmessage = function (m) {
      var ev;
      try { ev = JSON.parse(m.data); } catch (e) { return; }
      if (Array.isArray(ev)) ev.forEach(handleEvent); else handleEvent(ev);
    };
    sock.onclose = function () {
      if (evSock !== sock) return;
      evTimer = setTimeout(connectEvents, evRetryMs);
      evRetryMs = Math.min(evRetryMs * 2, 10000);
    };
    sock.onerror = function () { try { sock.close(); } catch (e) { /* ignore */ } };
  }

  function handleEvent(ev) {
    if (!ev || typeof ev !== "object" || !ev.stage || typeof ev.task_id !== "string" || !ev.task_id) return;
    // Own properties only: on a plain object a task id of "__proto__" read Object.prototype (and
    // writing p.announced polluted it) and "toString" passed as a pending task.
    if (!Object.prototype.hasOwnProperty.call(pendingTasks, ev.task_id)) return;
    var p = pendingTasks[ev.task_id];
    if (!p || typeof p !== "object") return;
    var d = (ev.detail && typeof ev.detail === "object") ? ev.detail : {};
    var text = null;
    switch (ev.stage) {
      case "saved":
        text = "Done. Saved " + humanize(d.name || "the new skill") + ".";
        p.announced = true;
        break;
      case "not_saved":
        text = "Not saved" + (d.reason ? ": " + d.reason : "") + ". Trying the next best plan once.";
        p.announced = true;
        break;
      case "queued":
        text = "Queued, not done" + (d.reason ? ": " + d.reason : "") + ". I can't do that yet and have queued it.";
        p.announced = true;
        break;
      case "stopped":
        if (Date.now() - lastStopAt > 5000) text = "Stopped. The arm is frozen in place.";
        p.announced = true;
        delete pendingTasks[ev.task_id];
        break;
      case "error":
        text = "Error: " + (d.message || "the task failed") + ".";
        p.announced = true;
        delete pendingTasks[ev.task_id];
        break;
      case "done":
        if (!p.announced) text = "Done.";
        delete pendingTasks[ev.task_id];
        break;
      default:
        return;
    }
    if (text) announce(text);
  }

  function announce(text) {
    log("sys", "task update: " + text);
    if (!sessionReady) return;
    send({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content: [{ type: "input_text", text: "Robot update: " + text }] }
    });
    afterPlayback(function () { send({ type: "response.create" }); });
    if (!capturing) setStatus("ready: hold T to talk", "on");
  }

  // ------------------------------------------------------------- audio out
  var audioCtx = null;
  var nextStartTime = 0;
  var activeSources = [];

  function ensureAudio() {
    if (!audioCtx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) throw new Error("Web Audio is not available in this browser");
      try { audioCtx = new AC({ sampleRate: SAMPLE_RATE }); } catch (e) { audioCtx = new AC(); }
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
  }

  var badAudioLoggedAt = 0;

  // Returns null when the delta is not valid base64 (a server-side fault; it must not crash the page).
  function base64ToFloat32(b64) {
    var bin;
    try { bin = atob(b64); } catch (e) { return null; }
    var n = bin.length - (bin.length % 2);
    var bytes = new Uint8Array(n);
    for (var i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    var pcm = new Int16Array(bytes.buffer); // little-endian on every platform browsers run on
    var out = new Float32Array(pcm.length);
    for (var k = 0; k < pcm.length; k++) out[k] = pcm[k] / (pcm[k] < 0 ? 0x8000 : 0x7fff);
    return out;
  }

  function playDelta(b64) {
    if (!b64 || typeof b64 !== "string") return;
    var ctx;
    try { ctx = ensureAudio(); } catch (e) { return; }
    var floats = base64ToFloat32(b64);
    if (floats === null) {
      // One log line per burst, so a stream of bad deltas does not flood the log.
      if (Date.now() - badAudioLoggedAt > 2000) {
        badAudioLoggedAt = Date.now();
        log("err", "dropped an audio delta that was not valid base64");
      }
      return;
    }
    if (!floats.length) return;
    var buf = ctx.createBuffer(1, floats.length, SAMPLE_RATE);
    buf.getChannelData(0).set(floats);
    var src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    var now = ctx.currentTime;
    if (nextStartTime < now + 0.02) nextStartTime = now + 0.02;
    src.start(nextStartTime);
    nextStartTime += buf.duration;
    activeSources.push(src);
    src.onended = function () {
      var i = activeSources.indexOf(src);
      if (i >= 0) activeSources.splice(i, 1);
    };
  }

  function stopPlayback() {
    activeSources.forEach(function (s) { try { s.stop(); } catch (e) { /* already stopped */ } });
    activeSources = [];
    nextStartTime = 0;
  }

  function playbackIdle() {
    return !audioCtx || nextStartTime <= audioCtx.currentTime + 0.05;
  }

  // Wait for the current spoken turn to finish before asking for the next response
  // (the xAI docs recommend this to avoid overlapping audio after a tool call).
  function afterPlayback(fn) {
    var started = Date.now();
    (function tick() {
      if (playbackIdle() || Date.now() - started > 6000) { fn(); return; }
      setTimeout(tick, 100);
    })();
  }

  // -------------------------------------------------------------- audio in
  var micStream = null;
  var micSource = null;
  var processor = null;
  var sink = null;
  var capturing = false;
  var pendingSamples = [];
  var pendingCount = 0;
  var capturedSamples = 0;

  async function ensureMic() {
    if (micStream) return micStream;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("getUserMedia is not available here. Open the page over https:// or on localhost and allow the microphone.");
    }
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    return micStream;
  }

  function resample(input, fromRate, toRate) {
    if (fromRate === toRate) return new Float32Array(input);
    var ratio = fromRate / toRate;
    var n = Math.floor(input.length / ratio);
    var out = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var pos = i * ratio;
      var i0 = Math.floor(pos);
      var i1 = Math.min(i0 + 1, input.length - 1);
      var frac = pos - i0;
      out[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return out;
  }

  function pcm16Base64(floats) {
    var pcm = new Int16Array(floats.length);
    for (var i = 0; i < floats.length; i++) {
      var s = Math.max(-1, Math.min(1, floats[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    var bytes = new Uint8Array(pcm.buffer);
    var bin = "";
    for (var k = 0; k < bytes.length; k += 0x8000) {
      bin += String.fromCharCode.apply(null, bytes.subarray(k, Math.min(k + 0x8000, bytes.length)));
    }
    return btoa(bin);
  }

  function rms(buf) {
    var sum = 0;
    for (var i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / (buf.length || 1));
  }

  function takeSamples(n) {
    var out = new Float32Array(n);
    var off = 0;
    while (off < n && pendingSamples.length) {
      var b = pendingSamples[0];
      var need = n - off;
      if (b.length <= need) { out.set(b, off); off += b.length; pendingSamples.shift(); }
      else { out.set(b.subarray(0, need), off); pendingSamples[0] = b.subarray(need); off += need; }
    }
    pendingCount -= off;
    return off < n ? out.subarray(0, off) : out;
  }

  function sendChunk(n) {
    var samples = takeSamples(n);
    if (samples.length) send({ type: "input_audio_buffer.append", audio: pcm16Base64(samples) });
  }

  async function startTalking() {
    if (capturing) return;
    if (!sessionReady) { setStatus("not connected: press Connect first", "off"); return; }
    var ctx, stream;
    try {
      ctx = ensureAudio();
      stream = await ensureMic();
    } catch (e) {
      setError(e.message);
      log("err", e.message);
      return;
    }
    if (capturing) return;
    stopPlayback(); // barge in: the user is talking over the reply
    setError("");
    capturing = true;
    capturedSamples = 0;
    pendingSamples = [];
    pendingCount = 0;
    micSource = ctx.createMediaStreamSource(stream);
    processor = ctx.createScriptProcessor(2048, 1, 1);
    sink = ctx.createGain();
    sink.gain.value = 0; // the processor needs a destination; keep the mic out of the speakers
    var chunk = Math.round(SAMPLE_RATE * CHUNK_MS / 1000);
    processor.onaudioprocess = function (e) {
      if (!capturing) return;
      var input = e.inputBuffer.getChannelData(0);
      setLevel(rms(input));
      var data = resample(input, ctx.sampleRate, SAMPLE_RATE);
      pendingSamples.push(data);
      pendingCount += data.length;
      capturedSamples += data.length;
      while (pendingCount >= chunk) sendChunk(chunk);
    };
    micSource.connect(processor);
    processor.connect(sink);
    sink.connect(ctx.destination);
    setTalkActive(true);
    setStatus("listening", "on");
  }

  function stopTalking(silent) {
    if (!capturing) return;
    var releasedAt = nowMs();
    capturing = false;
    if (pendingCount > 0 && !silent) sendChunk(pendingCount);
    try { if (processor) { processor.disconnect(); processor.onaudioprocess = null; } } catch (e) { /* ignore */ }
    try { if (micSource) micSource.disconnect(); } catch (e) { /* ignore */ }
    try { if (sink) sink.disconnect(); } catch (e) { /* ignore */ }
    processor = null; micSource = null; sink = null;
    pendingSamples = []; pendingCount = 0;
    setTalkActive(false);
    setLevel(0);
    if (silent) return;
    if (capturedSamples < SAMPLE_RATE * MIN_PRESS_S) {
      send({ type: "input_audio_buffer.clear" });
      setStatus("too short, hold T while speaking", "on");
      log("sys", "press too short, nothing sent");
      return;
    }
    if (send({ type: "input_audio_buffer.commit" })) lastRelease = releasedAt;
    send({ type: "response.create" });
    setStatus("thinking", "on");
    log("sys", "sent " + (capturedSamples / SAMPLE_RATE).toFixed(1) + " s of audio");
  }

  // --------------------------------------------------------------- bindings
  function isTyping(ev) {
    var t = ev.target;
    return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
  }

  function bind() {
    $("btn-connect").addEventListener("click", connect);
    $("btn-disconnect").addEventListener("click", disconnect);
    $("btn-voice-stop").addEventListener("click", function (ev) { doStop("button", pressStamp(ev)); });
    var talk = $("btn-talk");
    talk.addEventListener("pointerdown", function (ev) { ev.preventDefault(); try { talk.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ } startTalking(); });
    talk.addEventListener("pointerup", function () { stopTalking(); });
    talk.addEventListener("pointercancel", function () { stopTalking(); });
    talk.addEventListener("lostpointercapture", function () { stopTalking(); });
    talk.addEventListener("contextmenu", function (ev) { ev.preventDefault(); });
    window.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" || ev.key === "Esc") { ev.preventDefault(); doStop("Esc", pressStamp(ev)); return; }
      // Ctrl/Alt/Meta+T is a browser shortcut (a new tab), not a push to talk.
      if (ev.key !== "t" && ev.key !== "T") return;
      if (ev.repeat || ev.ctrlKey || ev.altKey || ev.metaKey || isTyping(ev)) return;
      ev.preventDefault();
      startTalking();
    }, true);
    window.addEventListener("keyup", function (ev) {
      if (ev.key === "t" || ev.key === "T") stopTalking();
    }, true);
    window.addEventListener("blur", function () { stopTalking(); });
    // The dashboard forwards T presses from its own window ({type: "vara-key", event, key, repeat})
    // so push to talk works wherever the focus is, and a "blur" when its window loses the focus
    // while a forwarded press is still down (its keyup will never follow). Only our origin and only
    // our parent are trusted, and the messages are handled exactly like the key events above.
    window.addEventListener("message", function (ev) {
      if (ev.origin !== location.origin) return;
      if (!inIframe() || ev.source !== window.parent) return;
      var m = ev.data;
      if (!m || m.type !== "vara-key" || (m.key !== "t" && m.key !== "T")) return;
      if (m.event === "keydown") { if (!m.repeat) startTalking(); }
      else if (m.event === "keyup" || m.event === "blur") stopTalking();
    });
  }

  function init() {
    bind();
    connectEvents();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("No microphone access here (getUserMedia missing). Voice needs https:// or localhost; typed commands still work on the dashboard.");
    }
    if (!window.WebSocket) setError("This browser has no WebSocket support; voice is unavailable.");
    setStatus("disconnected: press Connect", "off");
    log("sys", "ready. Connect, then hold T to talk.");
  }

  // A plain tag at the end of <body>: every element above it exists, so start now (same rule as
  // app.js: never wait for DOMContentLoaded, which a stalled external script can hold back).
  init();
})();
