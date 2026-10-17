// Behaviour checks for web/static/app.js and voice.js, run by tests/test_web_pages.py under node.
// Each scenario loads the real page markup and script into the fake DOM of fakedom.mjs. Prints one
// PASS/FAIL line per check and exits 1 when any check failed. No network, no browser.
//
// Scenarios: start-up while a deferred script stalls; junk curves.json with and without Chart.js;
// Chart.js arriving late; a 5 MB curves.json; a hung and a failing /stop; a hung /state; 5000-character
// texts; trial rows, the LOCAL badge row and the "rejected:" label; non-integer candidate indices; the
// DEMO countdown under clock skew; junk /state entries and a throwing render; T forwarding to the voice
// iframe; and on the voice page: the realtime handshake, tool calls (no call_id, non-JSON arguments,
// 409), end-of-task announcements with prototype-named task ids, junk audio deltas and raw frames, and
// the forwarded key and blur messages.
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildDocument, makeFetch, makeSandbox } from "./fakedom.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const read = (p) => readFileSync(path.join(ROOT, p), "utf8");
const INDEX_HTML = read("web/index.html");
const VOICE_HTML = read("web/voice.html");
const APP_JS = read("web/static/app.js");
const VOICE_JS = read("web/static/voice.js");
const ORIGIN = "http://127.0.0.1:8000"; // the sandbox's location.origin

const unhandled = [];
process.on("unhandledRejection", (e) => unhandled.push(String(e && e.stack || e)));

const results = [];
function check(name, ok, info) {
  results.push({ name, ok: !!ok });
  console.log((ok ? "PASS " : "FAIL ") + name + (ok ? "" : ": " + JSON.stringify(info)));
}

const STATE_IDLE = { body: { state: "idle", mode: "normal", carry: "ik", task_id: null, skills: [], queue: [], stopped: false } };
const NO_CURVES = { status: 404, body: { detail: "not found" } };
const fakeChart = (made) => class { constructor(ctx, cfg) { made.push(cfg); } destroy() {} };

// A dashboard whose document.readyState is still "loading" and on which DOMContentLoaded never fires:
// exactly the situation while a deferred Chart.js request stalls.
function dashboard(routes = {}, prepare = null) {
  const doc = buildDocument(INDEX_HTML);
  const net = makeFetch(Object.assign({ "GET /state": STATE_IDLE, "GET /static/rl/curves.json": NO_CURVES }, routes));
  const h = makeSandbox(doc, { fetchImpl: net.fetchImpl });
  if (prepare) prepare(h.sandbox);
  h.run(APP_JS, "app.js");
  return Object.assign(h, { net, text: (id) => doc.getElementById(id).textContent, title: (id) => doc.getElementById(id).title });
}

// The page's WS /events socket, opened; receive(ev) feeds one event as the orchestrator would.
function eventsSocket(h) {
  const ws = h.sockets.find((s) => /\/events$/.test(s.url));
  ws.open();
  return ws;
}

const clean = (h) => h.errors.length === 0 && unhandled.length === 0;

// A voice page inside the dashboard iframe (a parent window that records postMessage) with a microphone.
const TOKEN = { body: { token: "test-client-secret-ABC123", expires_at: 1700000300, model: "grok-voice-latest" } };
function voicePage(routes = {}) {
  const doc = buildDocument(VOICE_HTML);
  const posted = [];
  const parent = { postMessage(m, origin) { posted.push({ m, origin }); } };
  const net = makeFetch(Object.assign({ "POST /voice/token": TOKEN, "GET /state": STATE_IDLE }, routes));
  const h = makeSandbox(doc, { fetchImpl: net.fetchImpl, parent, mic: true });
  h.run(VOICE_JS, "voice.js");
  return Object.assign(h, {
    net,
    posted,
    status: () => doc.getElementById("voice-status").textContent,
    talking: () => doc.getElementById("btn-talk").classList.contains("active"),
    logLines: (kind, re) => doc.getElementById("voice-log").querySelectorAll("li" + (kind ? "." + kind : "")).filter((li) => !re || re.test(li.textContent)),
  });
}

// Press Connect, open the fake realtime socket and acknowledge the session: the page is then ready.
async function connectVoice(h) {
  h.doc.getElementById("btn-connect").click();
  await h.timers.flush();
  const rt = h.sockets.find((s) => /api\.x\.ai/.test(s.url));
  if (!rt) throw new Error("the voice page opened no realtime socket");
  rt.open();
  rt.receive({ type: "session.updated" });
  await h.timers.flush();
  rt.msgs = (type) => rt.sent.map((s) => JSON.parse(s)).filter((m) => !type || m.type === type);
  rt.outputs = () => Object.fromEntries(rt.msgs("conversation.item.create").filter((m) => m.item && m.item.type === "function_call_output").map((m) => [m.item.call_id, m.item.output]));
  rt.updates = () => rt.msgs("conversation.item.create").filter((m) => m.item && m.item.role === "user").map((m) => m.item.content[0].text);
  return rt;
}

// ------------------------------------------------------------ 1. start-up while a deferred script stalls
{
  const h = dashboard({ "POST /stop": { body: { state: "stopped", task_id: null } } });
  await h.timers.flush();
  const cards = h.doc.getElementById("cards").children.length;
  const pill = h.text("pill-state");
  const dcl = h.doc.listenerCount("DOMContentLoaded");
  h.doc.getElementById("btn-stop").click();
  await h.timers.flush();
  const stops1 = h.net.count("POST /stop");
  h.fire("keydown", { key: "Escape" });
  await h.timers.flush();
  const stops2 = h.net.count("POST /stop");
  check("app.js initialises while readyState is loading and DOMContentLoaded never fires (6 cards, /state polled)",
    cards === 6 && pill === "state: idle" && dcl === 0, { cards, pill, dcl, errors: h.errors });
  check("STOP button and Esc reach POST /stop without DOMContentLoaded", stops1 === 1 && stops2 === 2, { stops1, stops2 });
  check("no console errors or unhandled rejections at start-up", clean(h), { errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 2. voice.js starts the same way
{
  const doc = buildDocument(VOICE_HTML);
  const net = makeFetch({});
  const h = makeSandbox(doc, { fetchImpl: net.fetchImpl });
  h.run(VOICE_JS, "voice.js");
  await h.timers.flush();
  const status = doc.getElementById("voice-status").textContent;
  const log = doc.getElementById("voice-log").textContent;
  check("voice.js initialises without DOMContentLoaded (status set, ready line logged, /events socket opened)",
    status === "disconnected: press Connect" && /ready/.test(log) && h.sockets.length === 1 && /\/events$/.test(h.sockets[0].url) && doc.listenerCount("DOMContentLoaded") === 0 && clean(h),
    { status, log, sockets: h.sockets.map((s) => s.url), errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 3. junk entries in curves.json
const JUNK_CURVES = { body: { variants: [
  { name: "a", steps: [100, 200], success: [0.1, 0.5] },
  null,
  5,
  { name: "b", steps: "x", success: "abc" },
  { steps: [1, 2], success: [null, "0.75"] },
], chosen: "a" } };
{
  const h = dashboard({ "GET /static/rl/curves.json": JUNK_CURVES });
  await h.timers.flush();
  const fb = h.doc.getElementById("rl-fallback");
  const t = fb.textContent;
  check("loadRl without Chart.js: non-object variants and non-numeric values are skipped, text summary shown, no exception",
    fb.hidden === false && /a \(chosen\): final success 50%/.test(t) && /b: final success \?/.test(t) && /variant 3: final success 75%/.test(t) && h.text("rl-chosen") === "chosen variant: a" && clean(h),
    { text: t, chosen: h.text("rl-chosen"), errors: h.errors, unhandled });
}
{
  const made = [];
  const h = dashboard({ "GET /static/rl/curves.json": JUNK_CURVES }, (sb) => { sb.Chart = fakeChart(made); });
  await h.timers.flush();
  const fb = h.doc.getElementById("rl-fallback");
  const ds = made.length ? made[0].data.datasets : [];
  check("loadRl with Chart.js: the object variants draw, junk entries are skipped, no exception",
    fb.hidden === true && made.length === 1 && ds.length === 3 && ds[0].label === "a (chosen)" && ds[0].data.length === 2 && ds[1].data.length === 0 && ds[2].data.length === 1 && clean(h),
    { made: made.length, labels: ds.map((d) => d.label), points: ds.map((d) => d.data.length), errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 4. Chart.js arriving late
{
  const made = [];
  const h = dashboard({ "GET /static/rl/curves.json": { body: { variants: [{ name: "a", steps: [100, 200], success: [0.2, 0.4] }], chosen: "a" } } });
  await h.timers.flush();
  const fb = h.doc.getElementById("rl-fallback");
  const curves = () => h.net.count("GET /static/rl/curves.json");
  const before = { hidden: fb.hidden, text: fb.textContent, made: made.length, requests: curves() };
  h.sandbox.Chart = fakeChart(made); // the deferred CDN script lands now
  await h.timers.tick(1999);
  const mid = { made: made.length, requests: curves() };
  await h.timers.tick(1);
  const after = { hidden: fb.hidden, made: made.length, requests: curves(), label: made.length ? made[0].data.datasets[0].label : null };
  check("Chart.js arriving late: the text summary shows first, then the 2 s re-check draws the chart (nothing cached as failed, no 30 s wait)",
    before.hidden === false && /a \(chosen\): final success 40%/.test(before.text) && before.made === 0 && mid.made === 0 && after.made === 1 && after.hidden === true && after.requests === 2 && after.label === "a (chosen)" && clean(h),
    { before, mid, after, errors: h.errors });
}

// ------------------------------------------------------------ 5. a 5 MB curves.json, and the percent unit
{
  const N = 100000;
  const steps = Array.from({ length: N }, (_, i) => i * 10);
  const succ = Array.from({ length: N }, (_, i) => Math.round(1000 * i / (N - 1)) / 1000);
  const body = JSON.stringify({ variants: ["a", "b", "c", "d"].map((name) => ({ name, steps, success: succ })), chosen: "b" });
  const made = [];
  const h = dashboard({ "GET /static/rl/curves.json": { body } }, (sb) => { sb.Chart = fakeChart(made); });
  await h.timers.flush();
  const ds = made.length ? made[0].data.datasets : [];
  const last = ds.length === 4 ? ds[1].data[ds[1].data.length - 1] : null;
  check("a 5 MB curves.json (4 x 100 000 points) draws: the extremes come from reduce (Math.max.apply overflowed the call stack), 0..1 scaled to percent",
    made.length === 1 && ds.length === 4 && ds[1].label === "b (chosen)" && ds[1].data.length === N && last !== null && last.x === (N - 1) * 10 && last.y === 100 && h.doc.getElementById("rl-fallback").hidden === true && clean(h),
    { bytes: body.length, made: made.length, datasets: ds.length, last, fallback: h.text("rl-fallback").slice(0, 100), errors: h.errors });
}
{
  const made = [];
  const h = dashboard({ "GET /static/rl/curves.json": { body: { variants: [{ name: "a", steps: [1, 2, 3], success: [10, 55, 100] }, { name: "b", steps: [1, 2], success: [0.5, 1] }], chosen: "a" } } }, (sb) => { sb.Chart = fakeChart(made); });
  await h.timers.flush();
  const ys = made.length ? made[0].data.datasets.map((d) => d.data.map((p) => p.y)) : null;
  check("series already in 0..100 are not rescaled: the maximum over every variant decides the unit",
    JSON.stringify(ys) === "[[10,55,100],[0.5,1]]" && clean(h), { ys, errors: h.errors });
}

// ------------------------------------------------------------ 6. a hung POST /stop
{
  const h = dashboard({ "POST /stop": "hang" });
  await h.timers.flush();
  const headline = () => h.text("headline");
  h.doc.getElementById("btn-stop").click();
  await h.timers.flush();
  const c1 = h.net.count("POST /stop");
  const h1 = headline();
  await h.timers.tick(2900);
  const c2 = h.net.count("POST /stop");
  await h.timers.tick(100); // 3000 ms: the deadline
  const h2 = headline();
  await h.timers.tick(300); // the first automatic re-send is due 250 ms later
  const c3 = h.net.count("POST /stop");
  const h3 = headline();
  await h.timers.tick(3300);
  await h.timers.tick(3300);
  const c4 = h.net.count("POST /stop");
  await h.timers.tick(3300); // the last re-send times out: no more automatic ones
  const c5 = h.net.count("POST /stop");
  const h4 = headline();
  h.fire("keydown", { key: "Escape" });
  await h.timers.flush();
  const c6 = h.net.count("POST /stop");
  check("a hung POST /stop is abandoned after 3 s and the headline says to pull the 12 V plug",
    c1 === 1 && /freezing/.test(h1) && c2 === 1 && /no answer/.test(h2) && /12 V/.test(h2), { c1, h1, c2, h2 });
  check("STOP is re-sent automatically three times after silence, with the plug advice on screen",
    c3 === 2 && /sent again, 1 of 3/.test(h3) && /12 V/.test(h3) && c4 === 4 && c5 === 4, { c3, h3, c4, c5 });
  check("after the re-sends Esc still sends STOP (the button is never stuck behind stopInFlight)",
    /Press STOP or Esc/.test(h4) && c6 === 5 && clean(h), { h4, c6, errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 7. /stop answering 500: no re-send loop
{
  const h = dashboard({ "POST /stop": { status: 500, body: { detail: "boom" } } });
  await h.timers.flush();
  h.doc.getElementById("btn-stop").click();
  await h.timers.flush();
  const h1 = h.text("headline");
  await h.timers.tick(10000);
  const c = h.net.count("POST /stop");
  check("a /stop that answers 500 is reported once with the plug advice and not re-sent",
    /STOP request failed: boom/.test(h1) && /12 V/.test(h1) && c === 1 && clean(h), { h1, c, errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 8. a hung GET /state never piles up
{
  const h = dashboard({ "GET /state": "hang" });
  await h.timers.flush();
  const c0 = h.net.count("GET /state");
  await h.timers.tick(7000); // three 2 s polls would have started here without the guard
  const c1 = h.net.count("GET /state");
  const pill1 = h.text("pill-state");
  await h.timers.tick(1500); // the 8 s deadline, then the next interval poll
  const c2 = h.net.count("GET /state");
  const pill2 = h.text("pill-state");
  check("one /state poll in flight at a time: a hung poll is not joined by new ones for 7 s",
    c0 === 1 && c1 === 1 && pill1 === "state: connecting", { c0, c1, pill1 });
  check("the hung /state poll ends at the 8 s deadline as 'orchestrator unreachable' and polling resumes",
    c2 === 2 && pill2 === "orchestrator unreachable" && clean(h), { c2, pill2, errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 9. 5000-character texts are cut, full text in title
{
  const h = dashboard({});
  await h.timers.flush();
  const ws = eventsSocket(h);
  const long = "x".repeat(5000);
  ws.receive({ t: 1700000000, task_id: "t-0001", stage: "routing", detail: { text: long } });
  const head = { shown: h.text("headline").length, title: h.title("headline").length };
  ws.receive({ t: 1700000001, task_id: "t-0001", stage: "writing_code", detail: { candidate: 1, hint: long } });
  const hint = h.doc.getElementById("card-1").querySelector(".hint");
  ws.receive({ t: 1700000002, task_id: "t-0001", stage: "spec_ready", detail: { spec: { name: "n".repeat(500), english: long, success: {} } } });
  ws.receive({ t: 1700000003, task_id: "t-0001", stage: "error", detail: { message: long } });
  ws.receive({ t: 1700000004, task_id: "t-0001", stage: "queued", detail: { task: long, reason: long } });
  const bodies = h.doc.getElementById("timeline").querySelectorAll("li .body");
  const lens = bodies.map((b) => b.textContent.length);
  const titled = bodies.filter((b) => b.title.length >= 5000).length;
  check("headline: 300 visible characters, the full text in the title",
    head.shown <= 300 + "t-0001".length && head.title >= 5000, head);
  check("card hint: 200 visible characters, the full text in the title",
    hint.textContent.length === 200 && hint.title.length === 5000, { shown: hint.textContent.length, title: hint.title.length });
  check("spec banner: english cut at 300 and name at 80, full texts in the titles",
    h.text("spec-english").length === 300 && h.title("spec-english").length === 5000 && h.text("spec-name").length === 80 && h.title("spec-name").length === 500,
    { english: h.text("spec-english").length, name: h.text("spec-name").length });
  check("timeline lines: every field cut at 200, the uncut line in the title",
    bodies.length === 5 && lens.every((n) => n < 450) && titled === 5 && clean(h), { lens, titled, errors: h.errors, unhandled });
  ws.receive({ t: 1700000005, task_id: "t-0001", stage: "candidate_scored", detail: { candidate: 2, passed: false, successes: 1, trials: 5, fail_reason: long } });
  ws.receive({ t: 1700000006, task_id: "t-0001", stage: "candidate_scored", detail: { candidate: 3, passed: false, successes: 0, trials: 5, error: long } });
  ws.receive({ t: 1700000007, task_id: "t-0001", stage: "candidate_rejected", detail: { candidate: 4, problems: [long, "second"] } });
  const reason = (i) => h.doc.getElementById("card-" + i).querySelector(".reason");
  check("card fail_reason, error and rejection problems: 200 visible characters, the full text in the title",
    reason(2).textContent.length === 200 && reason(2).title.length === 5000 && reason(3).textContent.length === 200 && reason(3).title.length === 5000 && reason(4).textContent.length === 200 && reason(4).title === long + "; second" && clean(h),
    { r2: [reason(2).textContent.length, reason(2).title.length], r3: [reason(3).textContent.length, reason(3).title.length], r4: [reason(4).textContent.length, reason(4).title.length], errors: h.errors });
}

// ------------------------------------------------------------ 10. trial rows, the LOCAL badge row, the "rejected:" label, candidate indices
{
  const h = dashboard({});
  await h.timers.flush();
  const ws = eventsSocket(h);
  let t = 1700001000;
  const scored = (detail) => ws.receive({ t: t++, task_id: "t-0010", stage: "candidate_scored", detail });
  const card = (i) => h.doc.getElementById("card-" + i);
  const row = (i) => card(i).querySelectorAll(".trials span").map((s) => ({ tick: "t", cross: "f", count: "#", pending: "." })[s.className] || "?").join("");
  const countOf = (i) => (card(i).querySelector(".trials .count") || {}).textContent || null;
  const status = (i) => card(i).querySelector(".top .status");
  scored({ candidate: 1, passed: false, successes: 3, trials: 5, nominal_success: true, trial_results: [true, true, true, false, false], runner: "local", fail_reason: "trial 4: spec not satisfied" });
  scored({ candidate: 2, passed: false, successes: 1, trials: 2, trial_results: [{ success: true }, { ok: false }] });
  scored({ candidate: 3, passed: false, successes: 4, trials: 5, nominal_success: true });
  scored({ candidate: 4, passed: false, fail_reason: "trial 1: error: boom" });
  scored({ candidate: 5, passed: false, successes: 0, trials: 5, nominal_success: false, trial_results: [], error: "rejected: name starting with _ at line 12", runner: "local" });
  check("trial marks come from trial_results (booleans or {success}/{ok} entries) with per-trial titles",
    row(1) === "tttff" && card(1).querySelectorAll(".trials span")[3].title === "trial 4: fail" && row(2) === "tf", { row1: row(1), row2: row(2) });
  check("without trial_results: trial 1's mark from nominal_success plus the count ('4/5'); a bare {candidate, passed, fail_reason} shows five pending dots",
    row(3) === "t#" && countOf(3) === "4/5" && row(4) === "....." && row(5) === "f#" && countOf(5) === "0/5" && /FAIL/.test(card(4).querySelector(".verdict").textContent) && card(4).querySelector(".reason").textContent === "trial 1: error: boom",
    { row3: row(3), count3: countOf(3), row4: row(4), row5: row(5), count5: countOf(5) });
  check("card header: the status keeps its nowrap class beside the number; the LOCAL badge sits in a .badges row of its own, never in the header",
    !!status(1) && status(1).className === "status" && status(1).textContent === "scored" && status(1).title === "scored" && card(1).querySelector(".top .badge") === null && card(1).querySelector(".badges .badge.local") !== null && card(1).querySelector(".badges").parentNode === card(1) && card(2).querySelector(".badges") === null,
    { status: status(1) && status(1).className, badgeInTop: !!card(1).querySelector(".top .badge"), badgeRow: !!card(1).querySelector(".badges .badge.local"), card2Badges: !!card(2).querySelector(".badges") });
  const label = () => status(6).textContent + (card(6).classList.contains("rejected") ? " [rejected]" : "");
  scored({ candidate: 6, passed: false, successes: 0, trials: 5, error: "Rejected by the validator: bad path" });
  const l1 = label();
  scored({ candidate: 6, passed: false, successes: 0, trials: 5, error: "rejected" });
  const l2 = label();
  scored({ candidate: 6, passed: false, successes: 0, trials: 5, error: "rejected: name starting with _ at line 12" });
  const l3 = label();
  scored({ candidate: 6, passed: false, successes: 0, trials: 5, error: null, fail_reason: "rejected: not an error" });
  const l4 = label();
  check("'rejected by check_source' (dashed card) only when error starts with 'rejected:'; 'Rejected by the validator', a bare 'rejected' or a fail_reason are plain failures",
    l1 === "scored" && l2 === "scored" && l3 === "rejected by check_source [rejected]" && l4 === "scored" && status(5).textContent === "rejected by check_source" && clean(h),
    { l1, l2, l3, l4, status5: status(5).textContent, errors: h.errors });
  const statuses = () => [1, 2, 3, 4, 5, 6].map((i) => status(i).textContent).join(",");
  const before = statuses();
  for (const c of [1.7, "1.7", "abc", true, null, 0, 7, -1, "1e0", [1], { n: 1 }, "", " "]) scored({ candidate: c, passed: true, successes: 5, trials: 5, trial_results: [true, true, true, true, true] });
  ws.receive({ t: t++, task_id: "t-0010", stage: "selected", detail: { candidate: 1.7 } });
  const after = statuses();
  const winners = h.doc.getElementById("cards").querySelectorAll(".card.winner").length;
  const passes = h.doc.getElementById("cards").querySelectorAll(".verdict.pass").length;
  scored({ candidate: "2", passed: true, successes: 5, trials: 5 });
  check("cardIndex: 1.7, '1.7', 'abc', true, null, 0, 7, -1, '1e0', [1], {} and '' name no card (statuses unchanged, no PASS, no winner); a digit-only '2' still names card 2",
    before === after && winners === 0 && passes === 0 && /PASS/.test(card(2).querySelector(".verdict").textContent) && clean(h),
    { before, after, winners, passes, errors: h.errors });
}

// ------------------------------------------------------------ 11. the DEMO countdown under clock skew
{
  const h = dashboard({ "GET /state": { body: Object.assign({}, STATE_IDLE.body, { mode: "demo" }) } });
  await h.timers.flush();
  const ws = eventsSocket(h);
  const box = h.doc.getElementById("countdown");
  const read = async () => { await h.timers.tick(250); return { hidden: box.hidden, value: h.text("countdown-value"), label: box.querySelector(".label").textContent, cls: box.className }; };
  const t0 = Date.now() / 1000;
  ws.receive({ t: t0 + 50, task_id: "t-0001", stage: "routing", detail: { text: "skew" } });
  const c1 = await read();
  ws.receive({ t: t0 + 1, task_id: "t-0001", stage: "executing", detail: {} });
  const c2 = await read();
  ws.receive({ t: t0 - 100, task_id: "t-0002", stage: "routing", detail: { text: "old" } });
  const c3 = await read();
  ws.receive({ t: t0 + 50, task_id: "t-0003", stage: "routing", detail: { text: "x" } });
  ws.receive({ t: t0 - 10, task_id: "t-0003", stage: "queued", detail: { task: "x", reason: "y" } });
  const c4 = await read();
  const odd = [];
  for (const [id, t] of [["t-0004", "abc"], ["t-0005", 1e18], ["t-0006", -5]]) { ws.receive({ t, task_id: id, stage: "routing", detail: {} }); odd.push(await read()); }
  const inRange = (c) => /^\d{1,2} s$/.test(c.value) && parseInt(c.value, 10) <= 75 && !/-\d/.test(c.label);
  check("DEMO countdown: a server clock 50 s ahead shows 75 s; 'executing' stamped before 'routing' reads 'arm moving at 0 s', never a negative",
    !c1.hidden && c1.value === "75 s" && c1.label === "command to arm moving" && c2.value === "75 s" && c2.label === "arm moving at 0 s" && /\bdone\b/.test(c2.cls), { c1, c2 });
  check("DEMO countdown: a routing event 100 s old shows 0 s and 'over the 75 s budget'; an end stamped before the start reads 'ended at 0 s'",
    c3.value === "0 s" && c3.label === "over the 75 s budget" && /\bover\b/.test(c3.cls) && c4.label === "ended at 0 s", { c3, c4 });
  check("DEMO countdown: t = 'abc', 1e18 and -5 stay within 0..75 s with no negative label", odd.every(inRange) && clean(h), { odd, errors: h.errors });
}

// ------------------------------------------------------------ 12. junk /state entries and a throwing render
{
  let skills = [];
  let queue = [];
  const h = dashboard({ "GET /state": () => ({ body: { state: "idle", mode: "normal", carry: "ik", task_id: null, skills, queue, stopped: false } }) });
  await h.timers.flush();
  skills = [null, 5, "s", [], { name: "ok_skill", created: {}, real: null, sim: "x", spec: 5 }, { name: null }];
  queue = [null, "plain", {}, { task: 7, reason: {} }];
  await h.timers.tick(2000);
  const rows = h.doc.getElementById("skills").querySelectorAll("li.skill");
  const qrows = h.doc.getElementById("queue").querySelectorAll("li");
  check("junk /state entries: non-object skills and null queue entries are skipped, the object ones render, the pill stays 'state: idle'",
    rows.length === 2 && rows[0].querySelector(".name").textContent === "ok skill" && /real \?\/\? /.test(rows[0].textContent) && qrows.length === 3 && qrows[0].textContent === "plain" && h.text("pill-state") === "state: idle" && clean(h),
    { rows: rows.length, first: rows[0] && rows[0].textContent, queue: qrows.map((q) => q.textContent), pill: h.text("pill-state"), errors: h.errors });
  const list = h.doc.getElementById("skills");
  list.appendChild = function () { throw new Error("append boom"); };
  skills = skills.concat([{ name: "another" }]);
  await h.timers.tick(2000);
  const p1 = h.text("pill-state");
  await h.timers.tick(2000);
  const p2 = h.text("pill-state");
  delete list.appendChild;
  await h.timers.tick(2000);
  const p3 = h.text("pill-state");
  const names = list.querySelectorAll("li.skill .name").map((n) => n.textContent).join(",");
  check("a throwing render says 'render error: ...' (never 'orchestrator unreachable') on every poll until it is fixed, then recovers",
    p1 === "render error: append boom" && p2 === p1 && p3 === "state: idle" && names === "ok skill,,another" && h.errors.length === 2 && h.errors.every((e) => /dashboard render error/.test(e)) && unhandled.length === 0,
    { p1, p2, p3, names, errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 13. T forwarding from the dashboard to the voice iframe
{
  const h = dashboard({});
  await h.timers.flush();
  const posted = [];
  h.doc.getElementById("voice-frame").contentWindow = { postMessage(m, origin) { posted.push(Object.assign({ origin }, m)); } };
  const body = h.doc.body;
  const input = h.doc.getElementById("command-text");
  const seq = () => posted.map((p) => p.event + (p.repeat ? "*" : "")).join(",");
  const take = () => { const s = seq(); posted.length = 0; return s; };
  h.fire("keydown", { key: "t", target: body });
  h.fire("keydown", { key: "t", target: body, repeat: true });
  h.fire("keyup", { key: "t", target: body });
  const shape = posted.length === 3 && posted.every((p) => p.type === "vara-key" && p.key === "t" && p.origin === ORIGIN);
  const s1 = take();
  h.fire("keydown", { key: "t", target: body, ctrlKey: true });
  h.fire("keydown", { key: "T", target: body, altKey: true });
  h.fire("keydown", { key: "t", target: body, metaKey: true });
  h.fire("keydown", { key: "t", target: input });
  h.fire("keydown", { key: "x", target: body });
  const s2 = seq();
  h.fire("keyup", { key: "t", target: input, ctrlKey: true });
  const s3 = take();
  h.fire("blur");
  const s4 = seq();
  h.fire("keydown", { key: "T", target: body });
  h.fire("blur");
  h.fire("blur");
  const s5 = take();
  h.fire("keyup", { key: "T", target: body });
  check("T keydown/keyup on the dashboard are forwarded to the voice iframe as {type: 'vara-key', event, key: 't', repeat} on our origin",
    shape && s1 === "keydown,keydown*,keyup", { shape, s1, posted });
  check("a keydown with Ctrl, Alt or Meta, while typing in the command box, or of another key is not forwarded; a keyup always is",
    s2 === "" && s3 === "keyup", { s2, s3 });
  check("a window blur is forwarded, once, only while a forwarded keydown is outstanding, so a press interrupted by a focus change never sticks",
    s4 === "" && s5 === "keydown,blur" && clean(h), { s4, s5, errors: h.errors });
}

// ------------------------------------------------------------ 14. voice page: handshake, tool calls, announcements, prototype-named task ids
{
  let commands = 0;
  const texts = [];
  const h = voicePage({
    "POST /command": (opts) => { commands += 1; texts.push(JSON.parse(opts.body).text); return commands === 1 ? { body: { task_id: "t-0001" } } : { status: 409, body: { detail: "busy" } }; },
    "GET /state": { body: Object.assign({}, STATE_IDLE.body, { skills: [{ name: "hand_over" }, null, { name: "place_in_tray" }] }) },
  });
  await h.timers.flush();
  const rt = await connectVoice(h);
  const su = rt.msgs("session.update")[0];
  check("voice connect: token from POST /voice/token, model in the URL, the token as the subprotocol, session.update with voice eve, turn_detection null and the three tools, then ready",
    h.net.count("POST /voice/token") === 1 && rt.url === "wss://api.x.ai/v1/realtime?model=grok-voice-latest" && JSON.stringify(rt.protocols) === "[\"xai-client-secret.test-client-secret-ABC123\"]" && !!su && su.session.voice === "eve" && su.session.turn_detection === null && su.session.tools.map((x) => x.name).join() === "do_task,stop,list_skills" && h.status() === "ready: hold T to talk" && h.doc.getElementById("btn-talk").disabled === false,
    { url: rt.url, protocols: rt.protocols, session: su && su.session && { voice: su.session.voice, td: su.session.turn_detection }, status: h.status() });
  const call = (callId, name, args) => rt.receive(Object.assign({ type: "response.function_call_arguments.done", name }, args === undefined ? {} : { arguments: args }, callId === undefined ? {} : { call_id: callId }));
  call(undefined, "do_task", "{\"text\": \"no call id\"}");
  await h.timers.flush();
  const noId = { outputs: Object.keys(rt.outputs()).length, commands, err: h.logLines("err", /without a call_id/).length };
  check("a tool call without a call_id is logged and ignored: no function_call_output, no POST /command", noId.outputs === 0 && noId.commands === 0 && noId.err === 1, noId);
  call("c1", "do_task", "this is not json");
  call("c2", "do_task", "[1, 2]");
  call("c3", "do_task", "5");
  call("c4", "do_task", "null");
  call("c5", "do_task", "");
  call("c6", "do_task", "{\"text\": 5}");
  call("c7", "do_task", "x".repeat(5000));
  call("c8", "nope", "{}");
  await h.timers.flush();
  let o = rt.outputs();
  check("non-JSON, array, number, null or 5000-character arguments answer 'error: invalid arguments (...)' and never reach POST /command",
    o.c1 === "error: invalid arguments (not JSON)" && o.c2 === "error: invalid arguments (not a JSON object)" && o.c3 === o.c2 && o.c4 === o.c2 && o.c7 === o.c1 && commands === 0, { o, commands });
  check("empty or non-string text asks for the user's words, an unknown tool is an error, and the answered calls are followed by response.create",
    o.c5 === "error: do_task needs the user's words in 'text'" && o.c6 === o.c5 && o.c8 === "error: unknown tool nope" && rt.msgs("response.create").length >= 1 && commands === 0, { o, creates: rt.msgs("response.create").length });
  call("c9", "list_skills", "{}");
  call("c10", "do_task", { text: "object args ok" });
  call("c11", "do_task", "{\"text\": \"hello arm\"}");
  await h.timers.flush();
  o = rt.outputs();
  check("list_skills reads /state (non-object entries skipped); object and JSON-string arguments reach POST /command verbatim: 'On it', then 'busy. One moment.' on 409",
    o.c9 === "Skills: hand over, place in tray" && o.c10 === "On it" && o.c11 === "busy. One moment." && texts.join("|") === "object args ok|hello arm" && /task t-0001 running/.test(h.status()), { o, texts, status: h.status() });
  const evs = eventsSocket(h);
  evs.receive({ t: 1, task_id: "t-0002", stage: "done", detail: {} });
  evs.receive({ t: 2, task_id: "t-0001", stage: "verifying", detail: {} });
  evs.receive({ t: 3, task_id: "t-0001", stage: "done", detail: {} });
  await h.timers.flush();
  const u1 = rt.updates();
  check("the end stage of the task do_task started is announced once as a user text item ('Robot update: Done.'); other tasks and stages are not",
    u1.length === 1 && u1[0] === "Robot update: Done." && rt.msgs("response.create").length >= 2, { u1, creates: rt.msgs("response.create").length });
  evs.receive({ t: 4, task_id: "__proto__", stage: "saved", detail: { name: "x" } });
  evs.receive({ t: 5, task_id: "toString", stage: "done", detail: {} });
  evs.receive({ t: 6, task_id: "constructor", stage: "error", detail: { message: "m" } });
  evs.receive({ t: 7, task_id: "hasOwnProperty", stage: "queued", detail: { reason: "r" } });
  evs.receive({ t: 8, task_id: "t-0001", stage: "done", detail: {} }); // already finished and forgotten
  await h.timers.flush();
  const u2 = rt.updates();
  const polluted = h.run("Object.prototype.announced") !== undefined || Object.prototype.announced !== undefined;
  check("task ids '__proto__', 'toString', 'constructor' and 'hasOwnProperty' match no pending task (pendingTasks has no prototype), nothing is announced and Object.prototype stays clean",
    u2.length === 1 && !polluted && clean(h), { u2, polluted, errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 15. voice page: junk audio deltas and raw frames
{
  const h = voicePage({});
  await h.timers.flush();
  const rt = await connectVoice(h);
  const thrown = [];
  const recv = (obj) => { try { rt.receive(obj); } catch (e) { thrown.push(String(e && e.message || e)); } };
  for (let i = 0; i < 31; i++) recv({ type: "response.output_audio.delta", delta: "!!!not base64 " + i });
  const badLines = h.logLines("err", /base64/).length;
  const b64 = (bytes) => Buffer.from(bytes).toString("base64");
  recv({ type: "response.output_audio.delta", delta: b64([1, 2, 3, 4]) });
  recv({ type: "response.output_audio.delta", delta: b64([1, 2, 3]) }); // an odd byte count: the last byte is dropped
  recv({ type: "response.output_audio.delta", delta: "QUJD" });
  recv({ type: "response.output_audio.delta", delta: Buffer.alloc(1 << 16, 7).toString("base64") });
  for (const d of ["", 12345, null, undefined]) recv(d === undefined ? { type: "response.output_audio.delta" } : { type: "response.output_audio.delta", delta: d });
  const src = h.audio.length ? h.audio[0].sources : [];
  const lengths = src.map((s) => s.buffer && s.buffer.length);
  const ordered = src.every((s, i) => i === 0 || s.startedAt > src[i - 1].startedAt);
  check("31 junk base64 deltas are dropped with one 'err' line and no exception; valid deltas (even an odd byte count) are scheduled in order; empty, numeric and missing deltas are ignored",
    thrown.length === 0 && badLines === 1 && h.audio.length === 1 && JSON.stringify(lengths) === "[2,1,1,32768]" && ordered && src[0].startedAt >= 0.02,
    { thrown: thrown.slice(0, 2), badLines, contexts: h.audio.length, lengths, ordered });
  const raws = [
    "not json", "[1,2]", "null", "\"str\"",
    "{\"type\":\"error\"}",
    "{\"type\":\"error\",\"error\":null}",
    "{\"type\":\"response.output_audio_transcript.done\",\"transcript\":\"x\"}",
    "{\"type\":\"conversation.item.created\",\"item\":{\"role\":\"user\",\"content\":\"notarray\"}}",
    "{\"type\":\"conversation.item.created\",\"item\":{\"role\":\"user\",\"content\":[null,5,{\"type\":\"input_audio\",\"transcript\":\"<img src=x onerror=alert(1)>\"}]}}",
    "{\"type\":\"response.function_call_arguments.done\"}",
    "{\"type\":\"response.output_audio_transcript.delta\",\"delta\":5}",
    "{\"type\":\"response.done\"}",
    "{\"type\":\"input_audio_buffer.committed\"}",
  ];
  for (const raw of raws) recv(raw);
  await h.timers.flush();
  const inert = h.logLines("user", /<img src=x onerror=alert\(1\)>/).length === 1 && h.doc.getElementById("voice-log").querySelectorAll("img").length === 0;
  check("13 raw junk frames (non-JSON, null, strings, half-shaped items, a call without anything) cause no exception; transcripts land as inert text",
    thrown.length === 0 && inert && /xAI error/.test(h.doc.getElementById("voice-error").textContent) && clean(h),
    { thrown, inert, error: h.doc.getElementById("voice-error").textContent, errors: h.errors, unhandled });
}

// ------------------------------------------------------------ 16. voice page: forwarded key and blur messages, its own keys, Esc
{
  const h = voicePage({ "POST /stop": { body: { state: "stopped", task_id: null } } });
  await h.timers.flush();
  const rt = await connectVoice(h);
  const msg = async (event, extra = {}) => {
    h.fire("message", Object.assign({ origin: ORIGIN, source: h.sandbox.parent, data: { type: "vara-key", event, key: "t", repeat: false } }, extra));
    await h.timers.flush();
    return h.talking();
  };
  const a = [];
  a.push(await msg("keydown"));
  a.push(await msg("blur"));
  a.push(await msg("keydown"));
  a.push(await msg("keyup"));
  const clears = rt.msgs("input_audio_buffer.clear").length;
  a.push(await msg("keydown", { origin: "http://evil.example" }));
  a.push(await msg("keydown", { source: {} }));
  a.push(await msg("keydown", { data: { type: "vara-key", event: "keydown", key: "t", repeat: true } }));
  a.push(await msg("keydown", { data: { type: "other", event: "keydown", key: "t" } }));
  check("forwarded messages: keydown starts capture, keyup and blur stop it (a too-short press is cleared, not sent); other origins, other sources, repeats and other types are ignored",
    JSON.stringify(a) === "[true,false,true,false,false,false,false,false]" && clears === 2 && h.status() === "too short, hold T while speaking", { a, clears, status: h.status() });
  const own = async (type, init) => { h.fire(type, init || {}); await h.timers.flush(); return h.talking(); };
  const b = [];
  b.push(await own("keydown", { key: "t", ctrlKey: true }));
  b.push(await own("keydown", { key: "t", altKey: true }));
  b.push(await own("keydown", { key: "T", metaKey: true }));
  b.push(await own("keydown", { key: "t" }));
  b.push(await own("blur"));
  b.push(await own("keydown", { key: "t", repeat: true }));
  b.push(await own("keydown", { key: "t" }));
  b.push(await own("keyup", { key: "t" }));
  check("the voice page's own keys: Ctrl/Alt/Meta+T and repeats do not start capture, T does, a window blur or the T keyup stops it",
    JSON.stringify(b) === "[false,false,false,true,false,false,true,false]", { b });
  h.fire("keydown", { key: "Escape" });
  await h.timers.flush();
  const stopped = h.posted.filter((p) => p.m && p.m.type === "vara-stopped");
  check("Esc on the voice page posts /stop once and tells the parent dashboard (vara-stopped, source Esc) on our origin",
    h.net.count("POST /stop") === 1 && stopped.length === 1 && stopped[0].m.source === "Esc" && stopped[0].origin === ORIGIN && h.status() === "STOPPED" && clean(h),
    { stops: h.net.count("POST /stop"), stopped, status: h.status(), errors: h.errors });
}

const failed = results.filter((r) => !r.ok);
console.log(`==== ${results.length - failed.length}/${results.length} checks passed ====`);
process.exit(failed.length ? 1 : 0);
