/* VARA dashboard. Plain JavaScript, no build step.
 *
 * Talks only to the orchestrator contract (docs/interfaces.md, "Orchestrator API"):
 *   POST /command {text} -> {task_id} | 409 busy
 *   POST /stop, POST /resume, POST /calibrate -> {error_cm}
 *   POST /reset_demo, POST /load_backup
 *   GET  /state -> {state, mode, skills, queue, carry}
 *   WS   /events -> replay of the last 200 events, then live (schemas/event.json)
 * plus the serving conventions: /video/overhead and /video/wrist (MJPEG),
 * /static/rl/curves.json ({variants: [{name, steps, success}], chosen}) and
 * /static/rl_sim.mp4 (both may be missing), and /voice in an iframe.
 *
 * Every event detail is optional; the page must stay useful with missing fields.
 */
(function () {
  "use strict";

  var DEMO_BUDGET_S = 75; // "command to arm moving" cap in DEMO_MODE (contract budget table)
  var N_CANDIDATES = 6;
  var STATE_POLL_MS = 2000;
  var RL_POLL_MS = 30000;
  var TIMELINE_MAX = 400;
  var REASON_MAX = 200; // visible chars of a fail_reason / error / hint; the full text goes in the title
  var HEADLINE_MAX = 300; // visible chars of the headline; the full text goes in its title
  var SPEC_ENGLISH_MAX = 300; // the banner sentence; the full spec JSON sits right below it
  var STOP_TIMEOUT_MS = 3000; // the server freezes within 0.2 s; silence this long means the request is hung or lost
  var STOP_RETRIES = 3; // automatic re-sends after a silent /stop; after that the button and Esc re-send
  var STATE_TIMEOUT_MS = 8000; // one /state poll in flight at a time, so a hung server cannot pile them up
  var API_TIMEOUT_MS = 30000; // every other request (/resume answers after the arm reaches home)
  var RL_RETRY_MS = 2000; // quick re-checks while the deferred Chart.js script has not landed
  var RL_RETRY_MAX = 30; // then fall back to the normal 30 s poll
  var TICK = "\u2713";
  var CROSS = "\u2717";
  var DOT = "\u00b7";

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  function fmtClock(t) {
    var d = t ? new Date(t * 1000) : new Date();
    if (isNaN(d.getTime())) d = new Date();
    return d.toTimeString().slice(0, 8);
  }

  function humanize(name) { return String(name === undefined || name === null ? "" : name).replace(/_/g, " "); }

  function pretty(obj) {
    try { return JSON.stringify(obj, null, 2); } catch (e) { return String(obj); }
  }

  function short(s, n) {
    s = String(s === undefined || s === null ? "" : s);
    return s.length > n ? s.slice(0, n - 1) + "\u2026" : s;
  }

  function num(v) {
    if (v === null || v === undefined || v === "" || typeof v === "boolean") return null;
    var n = Number(v);
    return isFinite(n) ? n : null;
  }

  function count(v) { var n = num(v); return n === null ? "?" : String(n); }

  function isObject(v) { return !!v && typeof v === "object" && !Array.isArray(v); }

  function isTyping(ev) {
    var t = ev.target;
    return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable === true);
  }

  // A long reason is cut for the card; the full text stays readable in the title.
  function reasonNode(text) {
    var full = String(text);
    var node = el("div", "reason", short(full, REASON_MAX));
    node.title = full;
    return node;
  }

  // fetch with a deadline: past it the request is aborted and the error is named "TimeoutError".
  // A hung orchestrator must never hold a button for ever, and hung requests must not pile up
  // (Chromium allows six connections per host and the two MJPEG feeds already hold two).
  async function fetchWithin(path, opts, timeoutMs) {
    var ctl = typeof AbortController === "function" ? new AbortController() : null;
    var timer = null;
    if (ctl) {
      opts.signal = ctl.signal;
      timer = setTimeout(function () { ctl.abort(); }, timeoutMs);
    }
    try {
      var res = await fetch(path, opts);
      var text = await res.text();
      return { ok: res.ok, status: res.status, text: text };
    } catch (e) {
      if (e && e.name === "AbortError") {
        var err = new Error("no answer in " + Math.round(timeoutMs / 1000) + " s");
        err.name = "TimeoutError";
        throw err;
      }
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  async function api(method, path, body, timeoutMs) {
    var opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    var r = await fetchWithin(path, opts, timeoutMs || API_TIMEOUT_MS);
    var data = null;
    try { data = r.text ? JSON.parse(r.text) : null; } catch (e) { data = { raw: r.text }; }
    return { ok: r.ok, status: r.status, data: data };
  }

  function errText(r) {
    if (!r) return "no response";
    var d = (r.data && typeof r.data === "object") ? r.data : {};
    var msg = d.detail || d.error || d.message || d.reason || d.raw;
    if (msg && typeof msg === "object") msg = pretty(msg);
    return msg ? String(msg) : ("HTTP " + r.status);
  }

  function setResult(id, text, cls) {
    var node = $(id);
    node.textContent = text;
    node.className = "result" + (cls ? " " + cls : "");
  }

  function setPill(id, text, cls) {
    var node = $(id);
    node.textContent = text;
    node.className = "pill" + (cls ? " " + cls : "");
  }

  // ------------------------------------------------------------------ state
  var state = { state: null, mode: null, carry: null, skills: [], queue: [] };
  var stoppedLocally = false; // set the moment STOP is pressed, cleared by the next /state poll

  function carryLabel() {
    if (state.carry === "rl") return "RL policy";
    if (state.carry === "ik") return "IK";
    return state.carry ? String(state.carry) : "?";
  }

  function applyState(s) {
    state.state = s.state || null;
    state.mode = s.mode || null;
    state.carry = s.carry || null;
    // 8.6: skills are skill.json objects, queue entries {task, reason, task_id}; drop anything else.
    state.skills = Array.isArray(s.skills) ? s.skills.filter(isObject) : [];
    state.queue = Array.isArray(s.queue) ? s.queue.filter(function (q) { return q !== null && q !== undefined; }) : [];
    stoppedLocally = false;
    document.body.dataset.state = state.state || "unknown";
    setPill("pill-state", state.state === "stopped" ? "STOPPED" : "state: " + (state.state || "?"), state.state || "");
    $("stopped-banner").hidden = state.state !== "stopped";
    $("btn-torque-off").disabled = state.state === "busy";
    setPill("pill-mode", "mode: " + (state.mode || "?"), state.mode === "demo" ? "on" : "");
    setPill("pill-carry", "carry: " + carryLabel(), state.carry === "rl" ? "on" : "");
    var rc = $("rl-carry");
    rc.textContent = "carry: " + carryLabel() + (state.carry === "rl" ? " (live on the arm)" : state.carry === "ik" ? " (RL policy not deployed)" : "");
    rc.className = "carry " + (state.carry || "");
    renderSkills();
    renderQueue();
  }

  var pollInFlight = false;

  async function pollState() {
    // Never more than one /state request in flight: a hung server must not eat the connection
    // pool one poll at a time (STOP needs a connection), and a hung poll ends at STATE_TIMEOUT_MS.
    if (pollInFlight) return;
    pollInFlight = true;
    var r;
    try {
      r = await api("GET", "/state", undefined, STATE_TIMEOUT_MS);
      if (!r.ok) throw new Error(errText(r));
    } catch (e) {
      pollInFlight = false;
      if (!stoppedLocally) setPill("pill-state", "orchestrator unreachable", "off");
      return;
    }
    pollInFlight = false;
    try {
      applyState(isObject(r.data) ? r.data : {});
    } catch (e) {
      // A bug in our own rendering must not masquerade as a dead server.
      if (window.console && console.error) console.error("dashboard render error", e);
      setPill("pill-state", "render error: " + short(e && e.message ? e.message : String(e), 60), "off");
    }
  }

  function showStopped(on) {
    stoppedLocally = on;
    if (on) {
      document.body.dataset.state = "stopped";
      setPill("pill-state", "STOPPED", "stopped");
      $("stopped-banner").hidden = false;
    }
  }

  // ------------------------------------------------------------- headline
  // One sentence. A long server or user text is cut here; the full text stays in the title.
  function setHeadline(taskId, text, cls) {
    var h = $("headline");
    h.innerHTML = "";
    if (taskId) h.appendChild(el("span", "task", taskId));
    var full = String(text);
    var shown = short(full, HEADLINE_MAX);
    h.appendChild(document.createTextNode(shown));
    h.title = shown === full ? "" : full;
    h.className = "headline" + (cls ? " " + cls : "");
  }

  // ------------------------------------------------------------- the task
  var task = newTaskModel(null, null);

  function newTaskModel(id, t) {
    return { id: id, startT: t, movingT: null, endT: null, cards: {}, winner: null, finalShown: false };
  }

  function startTask(id, t) {
    task = newTaskModel(id, t || Date.now() / 1000);
    $("spec-banner").hidden = true;
    for (var i = 1; i <= N_CANDIDATES; i++) renderCard(i);
  }

  function showSpec(spec) {
    if (!spec || typeof spec !== "object") return;
    var name = spec.name ? String(spec.name) : "";
    var english = String(spec.english || spec.description || "(no english description)");
    var nameNode = $("spec-name");
    nameNode.textContent = short(name, 80);
    nameNode.title = name.length > 80 ? name : "";
    var englishNode = $("spec-english");
    englishNode.textContent = short(english, SPEC_ENGLISH_MAX);
    englishNode.title = english.length > SPEC_ENGLISH_MAX ? english : "";
    $("spec-json").textContent = pretty(spec);
    $("spec-banner").hidden = false;
  }

  // ------------------------------------------------------------- the cards
  var cardNodes = {};

  function buildCards() {
    var box = $("cards");
    box.innerHTML = "";
    for (var i = 1; i <= N_CANDIDATES; i++) {
      var node = el("div", "card");
      node.id = "card-" + i;
      box.appendChild(node);
      cardNodes[i] = node;
      renderCard(i);
    }
  }

  // Only a whole number from 1 to 6 names a card. parseInt() read 1.7 (and "1.7") as card 1; a
  // fractional, negative or junk candidate now names no card. A digit-only string ("2") still does.
  function cardIndex(v) {
    var n = NaN;
    if (typeof v === "number") n = v;
    else if (typeof v === "string" && /^\s*\d+\s*$/.test(v)) n = Number(v);
    return (Number.isInteger(n) && n >= 1 && n <= N_CANDIDATES) ? n : null;
  }

  function cardUpdate(i, patch) {
    var c = task.cards[i] || {};
    Object.keys(patch).forEach(function (k) { c[k] = patch[k]; });
    task.cards[i] = c;
    renderCard(i);
  }

  function allCards(fn) {
    for (var i = 1; i <= N_CANDIDATES; i++) {
      var c = task.cards[i] || {};
      var patch = fn(c);
      if (patch) cardUpdate(i, patch);
    }
  }

  // Per-trial results come only from an explicit array in the candidate_scored detail (8.6:
  // trial_results, a list of booleans; entries shaped {success} or {ok} are tolerated, and so are
  // the older alias keys). Nothing else in a score says which trials failed.
  function perTrialArray(s) {
    var keys = ["trial_results", "trials_detail", "per_trial", "trial_successes", "results"];
    for (var k = 0; k < keys.length; k++) {
      var v = s[keys[k]];
      if (Array.isArray(v) && v.length) {
        return v.map(function (x) {
          if (x && typeof x === "object") return !!(x.success !== undefined ? x.success : x.ok);
          return !!x;
        });
      }
    }
    return null;
  }

  function mark(ok, title) {
    var m = el("span", ok ? "tick" : "cross", ok ? TICK : CROSS);
    m.title = title;
    return m;
  }

  // Without a per-trial array the row shows what the score really says: trial 1 (nominal_success,
  // the scene as observed) and the count, never invented positions for trials 2 to 5.
  function trialsRow(s) {
    var row = el("div", "trials");
    var arr = s ? perTrialArray(s) : null;
    if (arr) {
      arr.forEach(function (ok, k) { row.appendChild(mark(ok, "trial " + (k + 1) + ": " + (ok ? "success" : "fail"))); });
      return row;
    }
    var hasCount = s && num(s.successes) !== null && num(s.trials) !== null;
    var hasFirst = s && typeof s.nominal_success === "boolean";
    if (!hasCount && !hasFirst) {
      for (var k = 0; k < 5; k++) row.appendChild(el("span", "pending", DOT));
      return row;
    }
    if (hasFirst) row.appendChild(mark(s.nominal_success, "trial 1 (the scene as observed): " + (s.nominal_success ? "success" : "fail")));
    if (hasCount) {
      var c = el("span", "count", num(s.successes) + "/" + num(s.trials));
      c.title = num(s.successes) + " of " + num(s.trials) + " trials succeeded; the score did not say which";
      row.appendChild(c);
    }
    return row;
  }

  function renderCard(i) {
    var node = cardNodes[i];
    if (!node) return;
    var c = task.cards[i] || {};
    var s = c.score || null;
    var cls = "card";
    if (c.rejected) cls += " rejected";
    if (s) cls += " scored " + (s.passed ? "pass" : "fail");
    if (task.winner === i) cls += " winner";
    node.className = cls;
    node.innerHTML = "";

    var top = el("div", "top");
    var numBox = el("span", "num");
    numBox.appendChild(el("small", null, "candidate"));
    numBox.appendChild(document.createTextNode(String(i)));
    top.appendChild(numBox);
    var statusText = c.status || "waiting";
    var status = el("span", "status", statusText);
    status.title = statusText;
    top.appendChild(status);
    node.appendChild(top);
    // The LOCAL badge gets a row of its own; inside the header it squeezed the status (see style.css).
    if (s && s.runner === "local") {
      var badges = el("div", "badges");
      var b = el("span", "badge local", "local");
      b.title = "scored in the local fallback subprocess, not a Daytona sandbox";
      badges.appendChild(b);
      node.appendChild(badges);
    }

    var hintText = c.hint ? String(c.hint) : (c.status ? "" : "strategy hint arrives with writing_code");
    var hint = el("div", "hint", short(hintText, REASON_MAX));
    if (hintText.length > REASON_MAX) hint.title = hintText;
    node.appendChild(hint);
    node.appendChild(trialsRow(s));

    if (s) {
      var stats = el("div", "stats");
      if (num(s.successes) !== null && num(s.trials) !== null) stats.appendChild(el("span", null, s.successes + "/" + s.trials + " trials"));
      if (num(s.collisions) !== null) stats.appendChild(el("span", null, "collisions " + s.collisions));
      if (num(s.limit_hits) !== null) stats.appendChild(el("span", null, "limit hits " + s.limit_hits));
      if (num(s.ik_failures) !== null) stats.appendChild(el("span", null, "IK fails " + s.ik_failures));
      if (num(s.mean_sim_s) !== null) stats.appendChild(el("span", null, "mean " + Number(s.mean_sim_s).toFixed(1) + " s"));
      node.appendChild(stats);
      var passed = s.passed === true;
      node.appendChild(el("div", "verdict " + (passed ? "pass" : "fail"), passed ? TICK + " PASS" : CROSS + " FAIL"));
      var reason = s.fail_reason || s.error;
      if (reason) node.appendChild(reasonNode(reason));
    } else if (c.rejected && c.problems && c.problems.length) {
      node.appendChild(el("div", "verdict fail", CROSS + " REJECTED"));
      node.appendChild(reasonNode(c.problems.join("; ")));
    } else if (c.prevProblems && c.prevProblems.length) {
      var prev = el("div", "stats", "attempt 1 rejected: " + short(c.prevProblems[0], 80));
      prev.title = c.prevProblems.join("; ");
      node.appendChild(prev);
    }
    if (task.winner === i) node.appendChild(el("span", "ribbon", "WINNER"));
  }

  // ---------------------------------------------------------- the timeline
  var seenKeys = new Set();
  var seenOrder = [];

  function remember(key) {
    if (seenKeys.has(key)) return false;
    seenKeys.add(key);
    seenOrder.push(key);
    if (seenOrder.length > 3000) {
      var drop = seenOrder.splice(0, 1000);
      drop.forEach(function (k) { seenKeys.delete(k); });
    }
    return true;
  }

  // One line per event. Every server-written field is cut at `cap` characters: the timeline shows
  // REASON_MAX of each and keeps the uncut line (cap Infinity) in the title.
  function summarize(stage, d, cap) {
    var cut = function (s) { return short(s, cap); };
    var cand = d.candidate !== undefined && d.candidate !== null ? "candidate " + cut(d.candidate) : null;
    switch (stage) {
      case "routing": return d.text ? "\"" + cut(d.text) + "\"" : "routing the request";
      case "spec_ready": return cut((d.spec && (d.spec.english || d.spec.name)) || "success test written");
      case "writing_code": return cand ? cand + (d.hint ? ": " + cut(d.hint) : "") : "writing six candidate programs";
      case "candidate_rejected": {
        var p = Array.isArray(d.problems) ? d.problems : (d.problems ? [String(d.problems)] : []);
        return (cand || "candidate") + " rejected" + (p.length ? ": " + cut(p[0]) + (p.length > 1 ? " (+" + (p.length - 1) + " more)" : "") : "");
      }
      case "simulating": return "scoring in sandboxes, 5 trials each";
      case "candidate_scored": {
        var txt = (cand || "candidate") + " " + (d.passed ? "PASS" : "FAIL");
        if (num(d.successes) !== null && num(d.trials) !== null) txt += " " + num(d.successes) + "/" + num(d.trials);
        if (d.fail_reason) txt += ": " + cut(d.fail_reason);
        else if (d.error) txt += ": " + cut(d.error);
        if (d.runner === "local") txt += " [local]";
        return txt;
      }
      case "selected": return cand ? "winner: " + cand : "winner selected";
      case "planning": return "planning " + (d.skill ? cut(humanize(d.skill)) + " " : "") + (d.args ? cut(pretty(d.args).replace(/\s+/g, " ")) : "in simulation");
      case "validating": {
        var v = Array.isArray(d.problems) ? d.problems.join("; ") : (d.problems ? String(d.problems) : "");
        return "validate() on the laptop" + (v ? ": " + cut(v) : "");
      }
      case "executing": return "replaying the checked path on the arm" + (d.primitive ? " (" + cut(d.primitive) + ")" : "");
      case "verifying": return "camera check with the same success test";
      case "saved": return "Saved: " + (d.name ? cut(humanize(d.name)) : "skill");
      case "not_saved": return "Not saved" + (d.reason ? ": " + cut(d.reason) : "");
      case "queued": return "Queued" + (d.task ? ": \"" + cut(d.task) + "\"" : "") + (d.reason ? " - " + cut(d.reason) : "");
      case "done": return d.ok === false ? "done (failed)" : "done";
      case "stopped": return "arm frozen with torque on, task cancelled" + (d.reason ? " (" + cut(d.reason) + ")" : "");
      case "error": return cut(d.message || d.error || "error");
      default: return Object.keys(d).length ? cut(pretty(d).replace(/\s+/g, " ")) : "";
    }
  }

  function stageClass(stage, d) {
    if (stage === "saved" || stage === "selected") return "good";
    if (stage === "candidate_scored") return d.passed ? "good" : "bad";
    if (stage === "not_saved" || stage === "queued" || stage === "error" || stage === "candidate_rejected") return "bad";
    if (stage === "stopped") return "stopped";
    if (stage === "spec_ready") return "spec";
    return "";
  }

  function addTimeline(ev, d) {
    var list = $("timeline");
    var nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
    var li = el("li", stageClass(ev.stage, d));
    li.appendChild(el("span", "when", fmtClock(num(ev.t))));
    var body = el("span", "body");
    body.appendChild(el("span", "stage", ev.stage));
    if (ev.task_id) body.appendChild(el("span", "task", ev.task_id));
    var text = summarize(ev.stage, d, REASON_MAX);
    var full = summarize(ev.stage, d, Infinity);
    body.appendChild(document.createTextNode(text));
    if (full !== text) body.title = full;
    li.appendChild(body);
    var prev = list.querySelector("li.latest");
    if (prev) prev.classList.remove("latest");
    li.classList.add("latest");
    list.appendChild(li);
    while (list.children.length > TIMELINE_MAX) list.removeChild(list.firstChild);
    if (nearBottom) list.scrollTop = list.scrollHeight;
  }

  // ------------------------------------------------------------ the events
  function handleEvent(ev) {
    if (Array.isArray(ev)) { ev.forEach(handleEvent); return; }
    if (!ev || typeof ev !== "object" || !ev.stage) return;
    var stage = String(ev.stage);
    var d = (ev.detail && typeof ev.detail === "object") ? ev.detail : {};
    var key = [ev.t, ev.task_id, stage, d.candidate, d.name, d.reason, d.message, d.hint].join("|");
    if (!remember(key)) return;

    var tid = ev.task_id || null;
    var t = num(ev.t) || Date.now() / 1000;
    if (tid && tid !== task.id) startTask(tid, t);
    else if (stage === "routing" && !tid) startTask(null, t);

    switch (stage) {
      case "routing":
        if (!task.startT) task.startT = t;
        setHeadline(tid, "Routing: " + (d.text ? "\"" + d.text + "\"" : "one Grok call decides run, learn or queue"), "");
        break;
      case "spec_ready":
        showSpec(d.spec && typeof d.spec === "object" ? d.spec : (d.english || d.success ? d : null));
        setHeadline(tid, "Success test written before any code: " + ((d.spec && d.spec.english) || d.english || ""), "spec");
        break;
      case "writing_code": {
        var wi = cardIndex(d.candidate);
        var patch = function (c) {
          var p = { status: c.rejected ? "rewriting after rejection" : "writing code", score: null, rejected: false };
          if (c.rejected) p.prevProblems = c.problems;
          return p;
        };
        if (wi) { var pc = patch(task.cards[wi] || {}); if (d.hint) pc.hint = d.hint; cardUpdate(wi, pc); }
        else allCards(patch);
        setHeadline(tid, "Six agents are writing candidate programs", "");
        break;
      }
      case "candidate_rejected": {
        var ri = cardIndex(d.candidate);
        var problems = Array.isArray(d.problems) ? d.problems.map(String) : (d.problems ? [String(d.problems)] : []);
        if (ri) cardUpdate(ri, { status: "rejected by check_source", rejected: true, problems: problems, score: null });
        break;
      }
      case "simulating":
        allCards(function (c) { return (!c.score && !c.rejected) ? { status: "simulating 5 trials" } : null; });
        setHeadline(tid, "Scoring every candidate in its own sandbox, 5 randomized trials each", "");
        break;
      case "candidate_scored": {
        var si = cardIndex(d.candidate);
        if (si) {
          // The runner re-checks the source inside the sandbox; a rejection there comes back as a
          // score whose error is "rejected: <first problem>" (brief 8.4), so the card says so. Any
          // other error text (a bare "rejected", "Rejected by the validator ...") is a plain failure.
          var rejectedBySandbox = typeof d.error === "string" && d.error.indexOf("rejected:") === 0;
          var sp = { score: d, status: rejectedBySandbox ? "rejected by check_source" : "scored", rejected: rejectedBySandbox };
          if (typeof d.hint === "string" && d.hint) sp.hint = d.hint;
          cardUpdate(si, sp);
        }
        break;
      }
      case "selected": {
        var wi2 = cardIndex(d.candidate);
        if (wi2) {
          task.winner = wi2;
          for (var i = 1; i <= N_CANDIDATES; i++) renderCard(i);
          setHeadline(tid, "Winner: candidate " + wi2 + ". Validating its path before the arm moves", "good");
        }
        break;
      }
      case "planning":
        if (d.skill) task.skill = String(d.skill);
        setHeadline(tid, "Library hit: planning " + (d.skill ? humanize(d.skill) : "the saved skill") + " once in simulation", "");
        break;
      case "validating":
        setHeadline(tid, "validate() checks limits, speed and the workspace box on the laptop", "");
        break;
      case "executing":
        if (!task.movingT) task.movingT = t;
        setHeadline(tid, "Arm moving: replaying the checked path", "");
        break;
      case "verifying":
        setHeadline(tid, "Camera check with the same success test", "");
        break;
      case "saved":
        task.finalShown = true;
        setHeadline(tid, "Saved: " + (d.name ? humanize(d.name) : "new skill") + " - the library grew by one", "good");
        break;
      case "not_saved":
        task.finalShown = true;
        setHeadline(tid, "Not saved" + (d.reason ? ": " + short(d.reason, REASON_MAX) : "") + " - re-planning the next-best candidate once", "bad");
        break;
      case "queued":
        task.finalShown = true;
        task.endT = task.endT || t;
        setHeadline(tid, "Queued" + (d.task ? ": \"" + d.task + "\"" : "") + (d.reason ? " - " + d.reason : "") + ". Nothing moves.", "bad");
        break;
      case "done":
        task.endT = task.endT || t;
        if (!task.finalShown) setHeadline(tid, "Done." + (task.skill ? " Ran " + humanize(task.skill) + " from the library." : ""), "good");
        break;
      case "stopped":
        task.endT = task.endT || t;
        showStopped(true);
        setHeadline(tid, "STOPPED - arm frozen in place with torque on. Press Resume to send it home.", "stopped");
        break;
      case "error":
        task.endT = task.endT || t;
        setHeadline(tid, "Error: " + short(d.message || d.error || "see events", REASON_MAX), "bad");
        break;
      default:
        break;
    }
    addTimeline(ev, d);
  }

  var ws = null;
  var wsRetryMs = 1000;
  var wsTimer = null;

  function connectEvents() {
    clearTimeout(wsTimer);
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    var sock;
    try {
      sock = new WebSocket(proto + "//" + location.host + "/events");
    } catch (e) {
      setPill("pill-events", "events: unavailable", "off");
      scheduleReconnect();
      return;
    }
    ws = sock;
    sock.onopen = function () {
      wsRetryMs = 1000;
      setPill("pill-events", "events: live", "on");
    };
    sock.onmessage = function (m) {
      var ev;
      try { ev = JSON.parse(m.data); } catch (e) { return; }
      handleEvent(ev);
    };
    sock.onclose = function () {
      if (ws !== sock) return;
      setPill("pill-events", "events: reconnecting", "off");
      scheduleReconnect();
    };
    sock.onerror = function () {
      try { sock.close(); } catch (e) { /* already closed */ }
    };
  }

  function scheduleReconnect() {
    clearTimeout(wsTimer);
    wsTimer = setTimeout(connectEvents, wsRetryMs);
    wsRetryMs = Math.min(wsRetryMs * 2, 10000);
  }

  // ---------------------------------------------------------- the countdown
  function updateCountdown() {
    var box = $("countdown");
    if (state.mode !== "demo" || !task.startT) { box.hidden = true; return; }
    box.hidden = false;
    var label = box.querySelector(".label");
    var frozenAt = task.movingT || task.endT;
    var now = frozenAt || Date.now() / 1000;
    // Event times come from the server clock; a skew must not show "125 s" or "arm moving at -49 s".
    var elapsed = Math.max(0, now - task.startT);
    var remaining = Math.min(DEMO_BUDGET_S, Math.max(0, DEMO_BUDGET_S - elapsed));
    var cls = "countdown";
    if (task.movingT) { cls += " done"; label.textContent = "arm moving at " + Math.round(elapsed) + " s"; }
    else if (task.endT) { cls += elapsed > DEMO_BUDGET_S ? " over" : ""; label.textContent = "ended at " + Math.round(elapsed) + " s"; }
    else if (remaining <= 0) { cls += " over"; label.textContent = "over the 75 s budget"; }
    else { cls += remaining <= 15 ? " warn" : ""; label.textContent = "command to arm moving"; }
    box.className = cls;
    $("countdown-value").textContent = Math.ceil(remaining) + " s";
  }

  // ------------------------------------------------------------- the library
  var skillsKey = "";

  function renderSkills() {
    var list = $("skills");
    // Only object entries are skills (applyState already drops the rest; this keeps the function safe on its own).
    var skills = state.skills.filter(isObject);
    var key = JSON.stringify(skills.map(function (s) { return [s.name, s.created, s.real, s.sim, s.trusted]; }));
    if (key === skillsKey) return;
    list.innerHTML = "";
    if (!skills.length) { list.appendChild(el("li", "empty", "no skills in the library")); skillsKey = key; return; }
    // Highlight the newest learned skill (the starters are hand-written and never "new").
    var newest = null;
    skills.forEach(function (s) {
      if (s.trusted) return;
      var c = Date.parse(s.created || "");
      if (!isNaN(c) && (newest === null || c > newest.c)) newest = { name: s.name, c: c };
    });
    skills.forEach(function (s) {
      var li = el("li", "skill" + (newest && s.name === newest.name && !s.trusted ? " newest" : ""));
      var name = el("span", "name", humanize(s.name));
      if (s.trusted) { var b = el("span", "badge starter", "starter"); b.title = "hand-written starter skill"; name.appendChild(document.createTextNode(" ")); name.appendChild(b); }
      if (newest && s.name === newest.name && !s.trusted) { var nb = el("span", "badge new", "newest"); name.appendChild(document.createTextNode(" ")); name.appendChild(nb); }
      li.appendChild(name);
      var real = isObject(s.real) ? s.real : {}, sim = isObject(s.sim) ? s.sim : {};
      var stats = el("span", "stats");
      stats.appendChild(document.createTextNode("real "));
      stats.appendChild(el("b", null, count(real.successes)));
      stats.appendChild(document.createTextNode("/" + count(real.runs) + " " + DOT + " sim " + count(sim.successes) + "/" + count(sim.trials)));
      li.appendChild(stats);
      var spec = isObject(s.spec) ? s.spec : {};
      li.appendChild(el("span", "english", spec.english || spec.description || ""));
      list.appendChild(li);
    });
    // Remembered only after a complete render, so a render that threw is retried (and reported) on the next poll.
    skillsKey = key;
  }

  function renderQueue() {
    var list = $("queue");
    list.innerHTML = "";
    if (!state.queue.length) { list.appendChild(el("li", "empty", "queue empty")); return; }
    state.queue.forEach(function (q) {
      var li = el("li");
      if (q && typeof q === "object") {
        li.appendChild(el("div", null, q.task || q.text || q.name || pretty(q)));
        var reason = q.reason || q.why;
        if (reason) li.appendChild(el("div", "reason", String(reason)));
        if (q.task_id) li.appendChild(el("div", "reason mono", String(q.task_id)));
      } else {
        li.textContent = String(q);
      }
      list.appendChild(li);
    });
  }

  // -------------------------------------------------------------- the RL panel
  var rlChart = null;
  var rlDrawnKey = ""; // the curves.json text last drawn as a chart (unset until a draw succeeds)
  var rlRetryTimer = null;
  var rlRetries = 0;

  function rlFallback(text) {
    var f = $("rl-fallback");
    f.hidden = false;
    f.innerHTML = "";
    var pre = el("pre", null, text);
    f.appendChild(pre);
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
  }

  function variantName(v, i) {
    return typeof v.name === "string" && v.name ? v.name : "variant " + (i + 1);
  }

  // The last numeric entry of a success series (entries may be strings or null in a half-written file).
  function lastNumber(a) {
    if (!Array.isArray(a)) return null;
    for (var i = a.length - 1; i >= 0; i--) { var n = num(a[i]); if (n !== null) return n; }
    return null;
  }

  function textSummary(variants, chosen) {
    return variants.filter(isObject).map(function (v, i) {
      var name = variantName(v, i);
      var last = lastNumber(v.success);
      return name + (name === chosen ? " (chosen)" : "") + ": final success " + (last === null ? "?" : (last <= 1 ? Math.round(last * 100) : Math.round(last)) + "%");
    }).join("\n");
  }

  function drawChart(variants, chosen) {
    var colors = [cssVar("--series-1"), cssVar("--series-2"), cssVar("--series-3"), cssVar("--series-4")];
    var allVals = [];
    variants.forEach(function (v) { if (Array.isArray(v.success)) v.success.forEach(function (y) { var n = num(y); if (n !== null) allVals.push(n); }); });
    // The extremes come from reduce: Math.max.apply(null, allVals) passes every value as an argument
    // and overflowed the call stack on a 5 MB curves.json (400 000 values), losing the chart.
    var maxVal = allVals.reduce(function (m, y) { return y > m ? y : m; }, -Infinity);
    var asPercent = allVals.length > 0 && maxVal <= 1.0;
    var datasets = variants.slice(0, 8).map(function (v, i) {
      var steps = Array.isArray(v.steps) ? v.steps : [];
      var succ = Array.isArray(v.success) ? v.success : [];
      var pts = [];
      for (var k = 0; k < Math.min(steps.length, succ.length); k++) {
        var x = num(steps[k]), y = num(succ[k]);
        if (x !== null && y !== null) pts.push({ x: x, y: asPercent ? y * 100 : y });
      }
      var name = variantName(v, i);
      var isChosen = name === chosen;
      return {
        label: name + (isChosen ? " (chosen)" : ""),
        data: pts,
        borderColor: colors[i % colors.length],
        backgroundColor: colors[i % colors.length],
        borderWidth: isChosen ? 4 : 2,
        pointRadius: 0,
        pointHoverRadius: 6,
        tension: 0.15,
        order: isChosen ? 0 : 1
      };
    });
    var text = cssVar("--muted");
    var grid = "rgba(255,255,255,0.07)";
    var config = {
      type: "line",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { labels: { color: text, boxWidth: 18, usePointStyle: false } },
          tooltip: {
            callbacks: {
              title: function (items) { return items.length ? Math.round(items[0].parsed.x).toLocaleString() + " steps" : ""; },
              label: function (item) { return " " + item.dataset.label + ": " + Math.round(item.parsed.y) + "%"; }
            }
          }
        },
        scales: {
          x: { type: "linear", title: { display: true, text: "training steps", color: text }, ticks: { color: text, maxTicksLimit: 6 }, grid: { color: grid } },
          y: { min: 0, max: 100, title: { display: true, text: "success rate (%)", color: text }, ticks: { color: text, stepSize: 25 }, grid: { color: grid } }
        }
      }
    };
    $("rl-fallback").hidden = true;
    if (rlChart) { rlChart.destroy(); rlChart = null; }
    rlChart = new Chart($("rl-chart").getContext("2d"), config);
  }

  var rlInFlight = false;

  async function loadRl() {
    if (rlInFlight) return; // a hung curves.json must not pile up one connection per poll
    rlInFlight = true;
    try { await loadRlOnce(); } finally { rlInFlight = false; }
  }

  async function loadRlOnce() {
    var r;
    try {
      r = await fetchWithin("/static/rl/curves.json", { cache: "no-store" }, API_TIMEOUT_MS);
    } catch (e) {
      rlFallback("RL curves unavailable: " + e.message);
      return;
    }
    if (!r.ok) {
      rlFallback(r.status === 404 ? "No RL curves yet.\nrl/train_parallel.py writes web/static/rl/curves.json when the four variants finish." : "curves.json: HTTP " + r.status);
      $("rl-chosen").textContent = "chosen variant: none yet";
      return;
    }
    var data;
    try { data = JSON.parse(r.text); } catch (e) { rlFallback("curves.json is not valid JSON"); return; }
    if (!data || typeof data !== "object") { rlFallback("curves.json has an unexpected shape"); return; }
    // Only object entries are variants: a stray null or number in a half-written file must not
    // take the panel down (drawChart and textSummary both threw on one).
    var variants = Array.isArray(data.variants) ? data.variants.filter(isObject) : [];
    var chosen = typeof data.chosen === "string" ? data.chosen : null;
    $("rl-chosen").textContent = "chosen variant: " + (chosen ? humanize(chosen) : "none yet");
    if (!variants.length) { rlDrawnKey = ""; rlFallback("curves.json has no variants yet"); return; }
    if (typeof Chart === "undefined") {
      // Chart.js is a deferred CDN script: it may land after this poll, or never (venue Wi-Fi).
      // Show the numbers now, remember nothing as drawn, and look again soon rather than caching
      // the failure until the next 30 s poll.
      rlDrawnKey = "";
      rlFallback(textSummary(variants, chosen) + "\n\n(Chart.js did not load from the CDN; showing final values only)");
      if (rlRetries < RL_RETRY_MAX) {
        rlRetries += 1;
        clearTimeout(rlRetryTimer);
        rlRetryTimer = setTimeout(loadRl, RL_RETRY_MS);
      }
      return;
    }
    var key = r.text;
    if (key === rlDrawnKey) return;
    try {
      drawChart(variants, chosen);
      rlDrawnKey = key;
    } catch (e) {
      rlDrawnKey = "";
      rlFallback(textSummary(variants, chosen) + "\n\n(chart error: " + e.message + ")");
    }
  }

  function startVideo() {
    var v = $("rl-video");
    var fb = $("rl-video-fallback");
    v.addEventListener("error", function () { v.hidden = true; fb.hidden = false; });
    v.addEventListener("loadedmetadata", function () { v.hidden = false; fb.hidden = true; });
    v.src = v.dataset.src;
  }

  // ----------------------------------------------------------------- the feeds
  function startFeed(boxId) {
    var box = $(boxId);
    var img = box.querySelector("img");
    var label = box.querySelector(".nofeed");
    var src = img.dataset.src;
    var retry = null;
    img.addEventListener("load", function () { box.classList.add("live"); });
    img.addEventListener("error", function () {
      box.classList.remove("live");
      label.textContent = "no feed (" + src + ")";
      clearTimeout(retry);
      retry = setTimeout(function () { img.src = src + "?t=" + Date.now(); }, 5000);
    });
    img.src = src;
  }

  // --------------------------------------------------------------- the buttons
  var stopInFlight = false;
  var PLUG = " Pull the 12 V plug if the arm is moving.";

  // One /stop request at a time, but never for longer than STOP_TIMEOUT_MS: a hung request used to
  // hold the button and Esc for ever with the headline stuck on "freezing". Past the deadline the
  // human is told to pull the plug, the request is re-sent STOP_RETRIES times (freeze is idempotent)
  // and the button and Esc stay live throughout.
  async function stop(source, attempt) {
    if (stopInFlight) return;
    stopInFlight = true;
    attempt = attempt || 0;
    showStopped(true);
    setHeadline(task.id, attempt
      ? "STOP (" + source + ") sent again, " + attempt + " of " + STOP_RETRIES + ": no answer from the orchestrator yet." + PLUG
      : "STOP (" + source + "): freezing the arm in place", "stopped");
    var retry = false;
    try {
      var r = await api("POST", "/stop", undefined, STOP_TIMEOUT_MS);
      if (!r.ok) throw new Error(errText(r));
      setHeadline(task.id, "STOPPED - arm frozen in place with torque on. Press Resume to send it home.", "stopped");
    } catch (e) {
      if (e && e.name === "TimeoutError") {
        retry = attempt < STOP_RETRIES;
        setHeadline(task.id, "STOP (" + source + "): no answer from the orchestrator in " + Math.round(STOP_TIMEOUT_MS / 1000) + " s." + PLUG
          + (retry ? " Sending it again." : " Press STOP or Esc to send it again."), "stopped");
      } else {
        setHeadline(task.id, "STOP request failed: " + e.message + "." + PLUG, "stopped");
      }
    } finally {
      stopInFlight = false;
      if (retry) setTimeout(function () { stop(source, attempt + 1); }, 250);
      else pollState();
    }
  }

  async function resume() {
    var btn = $("btn-resume");
    btn.disabled = true;
    try {
      var r = await api("POST", "/resume");
      if (!r.ok) throw new Error(errText(r));
      stoppedLocally = false;
      $("stopped-banner").hidden = true;
      setHeadline(task.id, "Resumed: the arm is going home at 30 deg/s or less", "");
    } catch (e) {
      setHeadline(task.id, "Resume failed: " + e.message, "bad");
    } finally {
      btn.disabled = false;
      pollState();
    }
  }

  async function calibrate() {
    var btn = $("btn-calibrate");
    btn.disabled = true;
    setResult("calib-result", "checking", "");
    try {
      var r = await api("POST", "/calibrate");
      if (!r.ok) throw new Error(errText(r));
      var cm = num(r.data && r.data.error_cm);
      if (cm === null) throw new Error("no error_cm in the reply");
      setResult("calib-result", "calibration error " + cm.toFixed(2) + " cm" + (cm > 1.5 ? " - recalibrate" : " - ok"), cm > 1.5 ? "bad" : "ok");
    } catch (e) {
      setResult("calib-result", "calibration check failed: " + e.message, "bad");
    } finally {
      btn.disabled = false;
    }
  }

  // Maintenance only (CLAUDE.md rule 4): cutting torque lets the arm sag and drop what it holds,
  // so it takes two clicks within 5 s, and the orchestrator refuses it (409) while a task runs.
  var torqueArmedUntil = 0;

  async function torqueOff() {
    var btn = $("btn-torque-off");
    var now = Date.now();
    if (now > torqueArmedUntil) {
      torqueArmedUntil = now + 5000;
      btn.classList.add("armed");
      btn.textContent = "Confirm torque off";
      setResult("maint-result", "arm will sag: hold it, then click again within 5 s", "bad");
      setTimeout(function () {
        if (Date.now() >= torqueArmedUntil) {
          btn.classList.remove("armed");
          btn.textContent = "Torque off";
          if ($("maint-result").textContent.indexOf("arm will sag") === 0) setResult("maint-result", "", "");
        }
      }, 5200);
      return;
    }
    torqueArmedUntil = 0;
    btn.classList.remove("armed");
    btn.textContent = "Torque off";
    btn.disabled = true;
    try {
      var r = await api("POST", "/torque_off");
      if (r.status === 409) { setResult("maint-result", "refused: a task is running (409)", "bad"); return; }
      if (!r.ok) throw new Error(errText(r));
      setResult("maint-result", "torque off: the arm is limp until the next command re-enables it", "ok");
    } catch (e) {
      setResult("maint-result", "torque off failed: " + e.message, "bad");
    } finally {
      btn.disabled = false;
      pollState();
    }
  }

  async function sendCommand(ev) {
    ev.preventDefault();
    var input = $("command-text");
    var text = input.value.trim();
    if (!text) return;
    setResult("command-result", "sending", "");
    try {
      var r = await api("POST", "/command", { text: text });
      if (r.status === 409) { setResult("command-result", "busy: a task is already running (409)", "bad"); return; }
      if (!r.ok) throw new Error(errText(r));
      var tid = r.data && r.data.task_id;
      setResult("command-result", "accepted as " + (tid || "a task"), "ok");
      input.value = "";
    } catch (e) {
      setResult("command-result", "failed: " + e.message, "bad");
    }
  }

  async function libraryAction(path, label, btnId) {
    var btn = $(btnId);
    btn.disabled = true;
    setResult("library-result", label + "...", "");
    try {
      var r = await api("POST", path);
      if (!r.ok) throw new Error(errText(r));
      var d = r.data;
      // 8.6: {"skills": [names]}; a list of skill objects or a bare list is read the same way.
      var names = null;
      if (Array.isArray(d)) names = d;
      else if (d && Array.isArray(d.skills)) names = d.skills;
      else if (d && Array.isArray(d.loaded)) names = d.loaded;
      else if (d && Array.isArray(d.restored)) names = d.restored;
      var extra = "";
      if (names) {
        var shown = names.map(function (x) { return humanize(isObject(x) ? x.name : x); }).filter(Boolean);
        extra = ": " + (shown.length ? shown.join(", ") : names.length + " skill" + (names.length === 1 ? "" : "s"));
      }
      setResult("library-result", label + " done" + extra, "ok");
      skillsKey = "";
    } catch (e) {
      setResult("library-result", label + " failed: " + e.message, "bad");
    } finally {
      btn.disabled = false;
      pollState();
    }
  }

  function bind() {
    $("btn-stop").addEventListener("click", function () { stop("button"); });
    $("btn-resume").addEventListener("click", resume);
    $("btn-calibrate").addEventListener("click", calibrate);
    $("btn-torque-off").addEventListener("click", torqueOff);
    $("command-form").addEventListener("submit", sendCommand);
    $("btn-reset-demo").addEventListener("click", function () { libraryAction("/reset_demo", "reset demo", "btn-reset-demo"); });
    $("btn-load-backup").addEventListener("click", function () { libraryAction("/load_backup", "load backup", "btn-load-backup"); });
    // Esc anywhere on the page freezes the arm. Capture phase so no widget can swallow it.
    window.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" || ev.key === "Esc") { ev.preventDefault(); stop("Esc"); }
    }, true);
    // Push to talk from anywhere on the dashboard: T keydown/keyup are forwarded to the embedded
    // voice page, which handles the message exactly like its own key events. A keydown while
    // typing in a text field, or with Ctrl, Alt or Meta held (Ctrl+T opens a tab), is not a push;
    // a keyup is always forwarded so a press never sticks; and when this window loses the focus
    // while a forwarded press is still down (its keyup will never arrive), a "blur" is forwarded
    // so the voice page stops capturing.
    var talkKeyDown = false;
    function postTalk(event, repeat) {
      var frame = $("voice-frame");
      if (!frame || !frame.contentWindow) return false;
      try {
        frame.contentWindow.postMessage({ type: "vara-key", event: event, key: "t", repeat: !!repeat }, location.origin);
        return true;
      } catch (e) { return false; /* the frame is not ours or not loaded yet */ }
    }
    function forwardTalkKey(ev) {
      if (ev.key !== "t" && ev.key !== "T") return;
      if (ev.type === "keydown") {
        if (isTyping(ev) || ev.ctrlKey || ev.altKey || ev.metaKey) return;
        if (postTalk("keydown", ev.repeat)) talkKeyDown = true;
      } else {
        talkKeyDown = false;
        postTalk("keyup", ev.repeat);
      }
    }
    window.addEventListener("keydown", forwardTalkKey, true);
    window.addEventListener("keyup", forwardTalkKey, true);
    window.addEventListener("blur", function () {
      if (!talkKeyDown) return;
      talkKeyDown = false;
      postTalk("blur", false);
    });
    // The embedded /voice page posts its own STOP and tells us so we show it at once.
    window.addEventListener("message", function (ev) {
      if (ev.origin !== location.origin) return;
      var m = ev.data;
      if (m && m.type === "vara-stopped") { showStopped(true); setHeadline(task.id, "STOPPED from the voice page (" + (m.source || "voice") + ")", "stopped"); pollState(); }
    });
  }

  function init() {
    buildCards();
    bind();
    startFeed("feed-overhead");
    startFeed("feed-wrist");
    startVideo();
    pollState();
    setInterval(pollState, STATE_POLL_MS);
    connectEvents();
    loadRl();
    setInterval(loadRl, RL_POLL_MS);
    setInterval(updateCountdown, 250);
  }

  // This script is a plain tag at the end of <body>, so every element above it exists: start now.
  // Waiting for DOMContentLoaded would also wait for the deferred Chart.js download, and a stalled
  // CDN (half-open venue Wi-Fi) kept the cards, the polls, STOP and Esc dead for the whole stall.
  init();
})();
