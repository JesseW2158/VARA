// Minimal DOM, timer and network fakes so web/static/app.js and voice.js can run under node for
// tests/test_web_pages.py. Only what those two scripts touch is modelled; nothing here is a browser.
// Timers are virtual (tick() advances a clock, skip() moves it without running them), fetch is a
// route table, WebSocket records instances, Web Audio records the buffers it is asked to play and the
// microphone processors it creates, performance.now() follows the virtual clock (an event's timeStamp
// is whatever a check passes in), console.log lines are recorded, and a sandbox can be given a parent
// window so voice.js believes it is inside the dashboard iframe.
import vm from "node:vm";

const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"]);

class ClassList {
  constructor(el) { this.el = el; }
  list() { return this.el.className.split(/\s+/).filter(Boolean); }
  contains(c) { return this.list().includes(c); }
  add(...cs) { const l = this.list(); for (const c of cs) if (!l.includes(c)) l.push(c); this.el.className = l.join(" "); }
  remove(...cs) { this.el.className = this.list().filter((c) => !cs.includes(c)).join(" "); }
  toggle(c, force) { const on = force === undefined ? !this.contains(c) : !!force; if (on) this.add(c); else this.remove(c); return on; }
}

class TextNode {
  constructor(data) { this.nodeType = 3; this.data = String(data); this.parentNode = null; }
  get textContent() { return this.data; }
}

class Element {
  constructor(tag, doc) {
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.ownerDocument = doc;
    this.idValue = "";
    this.className = "";
    this.classList = new ClassList(this);
    this.childNodes = [];
    this.parentNode = null;
    this.title = "";
    this.hidden = false;
    this.disabled = false;
    this.dataset = {};
    this.style = {};
    this.value = "";
    this.src = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this.offsetWidth = 0;
    this.offsetHeight = 0;
    this.isContentEditable = false;
    this.contentWindow = null;
    this.listeners = new Map();
  }
  get id() { return this.idValue; }
  set id(v) {
    if (this.idValue) this.ownerDocument.byId.delete(this.idValue);
    this.idValue = String(v);
    this.ownerDocument.byId.set(this.idValue, this);
  }
  get children() { return this.childNodes.filter((n) => n.nodeType === 1); }
  get firstChild() { return this.childNodes[0] || null; }
  get textContent() { return this.childNodes.map((n) => n.textContent).join(""); }
  set textContent(v) { this.childNodes = []; if (v !== "" && v !== null && v !== undefined) this.childNodes.push(new TextNode(v)); }
  get innerHTML() { return ""; }
  set innerHTML(v) { if (v !== "") throw new Error("fakedom: innerHTML may only be set to \"\""); this.childNodes = []; }
  appendChild(n) {
    if (n.parentNode) n.parentNode.removeChild(n);
    n.parentNode = this;
    this.childNodes.push(n);
    return n;
  }
  removeChild(n) {
    const i = this.childNodes.indexOf(n);
    if (i < 0) throw new Error("fakedom: removeChild of a node that is not a child");
    this.childNodes.splice(i, 1);
    n.parentNode = null;
    return n;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(type, fn) { if (!this.listeners.has(type)) this.listeners.set(type, []); this.listeners.get(type).push(fn); }
  removeEventListener(type, fn) { if (this.listeners.has(type)) this.listeners.set(type, this.listeners.get(type).filter((f) => f !== fn)); }
  dispatchEvent(ev) {
    if (!ev.target) ev.target = this;
    if (typeof ev.preventDefault !== "function") ev.preventDefault = function () { this.defaultPrevented = true; };
    for (let n = this; n; n = n.parentNode) {
      for (const fn of n.listeners ? n.listeners.get(ev.type) || [] : []) fn(ev);
      if (ev.bubbles === false) break;
    }
    if (ev.bubbles !== false && this.ownerDocument.fireWindow) this.ownerDocument.fireWindow(ev);
    return !ev.defaultPrevented;
  }
  click() { return this.dispatchEvent({ type: "click", bubbles: true }); }
  focus() { this.ownerDocument.activeElement = this; }
  blur() { this.ownerDocument.activeElement = this.ownerDocument.body; }
  setPointerCapture() {}
  getContext() { return {}; }
  getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const out = [];
    for (const alt of sel.split(",")) {
      const parts = alt.trim().split(/\s+/).map(parseCompound);
      let set = [this];
      for (const c of parts) {
        const next = [];
        for (const root of set) for (const d of descendants(root)) if (matches(d, c) && !next.includes(d)) next.push(d);
        set = next;
      }
      for (const n of set) if (!out.includes(n)) out.push(n);
    }
    return out;
  }
}

function parseCompound(s) {
  const m = /^([a-zA-Z][a-zA-Z0-9-]*)?((?:[.#][\w-]+)*)$/.exec(s);
  if (!m) throw new Error("fakedom: unsupported selector " + JSON.stringify(s));
  const classes = [];
  let id = null;
  for (const p of m[2].match(/[.#][\w-]+/g) || []) { if (p[0] === ".") classes.push(p.slice(1)); else id = p.slice(1); }
  return { tag: m[1] ? m[1].toUpperCase() : null, classes, id };
}

function matches(el, c) {
  if (el.nodeType !== 1) return false;
  if (c.tag && el.tagName !== c.tag) return false;
  if (c.id && el.id !== c.id) return false;
  return c.classes.every((k) => el.classList.contains(k));
}

function descendants(el, out = []) {
  for (const ch of el.childNodes) if (ch.nodeType === 1) { out.push(ch); descendants(ch, out); }
  return out;
}

class Document {
  constructor() {
    this.byId = new Map();
    this.readyState = "loading"; // as while a deferred script is still downloading
    this.listeners = new Map();
    this.fireWindow = null;
    this.documentElement = new Element("html", this);
    this.head = new Element("head", this);
    this.body = new Element("body", this);
    this.documentElement.appendChild(this.head);
    this.documentElement.appendChild(this.body);
    this.activeElement = this.body;
  }
  getElementById(id) { return this.byId.get(String(id)) || null; }
  createElement(tag) { return new Element(tag, this); }
  createTextNode(t) { return new TextNode(t); }
  querySelector(sel) { return this.documentElement.querySelector(sel); }
  querySelectorAll(sel) { return this.documentElement.querySelectorAll(sel); }
  addEventListener(type, fn) { if (!this.listeners.has(type)) this.listeners.set(type, []); this.listeners.get(type).push(fn); }
  removeEventListener() {}
  listenerCount(type) { return (this.listeners.get(type) || []).length; }
}

function parseAttrs(s) {
  const out = {};
  const re = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;
  let m;
  while ((m = re.exec(s))) out[m[1].toLowerCase()] = m[2] !== undefined ? m[2] : m[3] !== undefined ? m[3] : m[4] !== undefined ? m[4] : "";
  return out;
}

function applyAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "id") el.id = v;
    else if (k === "class") el.className = v;
    else if (k === "hidden") el.hidden = true;
    else if (k === "disabled") el.disabled = true;
    else if (k === "title") el.title = v;
    else if (k === "src") el.src = v;
    else if (k.startsWith("data-")) el.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v;
  }
}

// Builds the element tree of an HTML page (ids, classes, data- and boolean attributes; text is dropped).
export function buildDocument(html) {
  const doc = new Document();
  const src = html.replace(/<!--[\s\S]*?-->/g, "").replace(/<script[\s\S]*?<\/script>/gi, "").replace(/<style[\s\S]*?<\/style>/gi, "");
  const re = /<(\/?)([a-zA-Z][a-zA-Z0-9-]*)([^>]*)>/g;
  const stack = [doc.documentElement];
  let m;
  let last = 0;
  while ((m = re.exec(src))) {
    const between = src.slice(last, m.index).trim(); // the page's own text, such as "state: connecting"
    if (between) stack[stack.length - 1].appendChild(new TextNode(between));
    last = re.lastIndex;
    const closing = m[1] === "/";
    const tag = m[2].toLowerCase();
    if (closing) {
      for (let i = stack.length - 1; i > 0; i--) if (stack[i].tagName === tag.toUpperCase()) { stack.length = i; break; }
      continue;
    }
    const attrs = parseAttrs(m[3]);
    let el;
    if (tag === "html") el = doc.documentElement;
    else if (tag === "head") el = doc.head;
    else if (tag === "body") el = doc.body;
    else { el = new Element(tag, doc); stack[stack.length - 1].appendChild(el); }
    applyAttrs(el, attrs);
    if (tag === "html") continue;
    if (!VOID.has(tag) && !m[3].trim().endsWith("/")) stack.push(el);
  }
  return doc;
}

function makeTimers() {
  let now = 0;
  let seq = 0;
  const queue = [];
  const add = (fn, ms, every) => { const t = { id: ++seq, fn, at: now + Math.max(0, Number(ms) || 0), every }; queue.push(t); return t.id; };
  const del = (id) => { const i = queue.findIndex((t) => t.id === id); if (i >= 0) queue.splice(i, 1); };
  const flush = async () => { for (let i = 0; i < 25; i++) await new Promise((r) => setImmediate(r)); };
  const tick = async (ms) => {
    const end = now + ms;
    for (;;) {
      await flush();
      const due = queue.filter((t) => t.at <= end).sort((a, b) => a.at - b.at || a.id - b.id)[0];
      if (!due) break;
      now = Math.max(now, due.at);
      if (due.every) due.at = now + due.every; else del(due.id);
      due.fn();
    }
    now = end;
    await flush();
  };
  // skip(ms) moves the clock without running timers: the time one network round trip takes, as seen
  // by a route handler, so a latency the page measures with performance.now() is exact.
  const skip = (ms) => { now += Math.max(0, Number(ms) || 0); };
  return {
    setTimeout: (fn, ms) => add(fn, ms, 0),
    clearTimeout: del,
    setInterval: (fn, ms) => add(fn, ms, Math.max(1, Number(ms) || 0)),
    clearInterval: del,
    tick,
    flush,
    skip,
    now: () => now,
  };
}

// A fetch backed by a route table {"GET /state": {status, body} | "hang" | (opts) => ...}.
// "hang" never answers but honours opts.signal (abort rejects with an AbortError, as in browsers).
export function makeFetch(routes) {
  const calls = [];
  const response = (status, text) => ({ ok: status >= 200 && status < 300, status, async text() { return text; }, async json() { return JSON.parse(text); } });
  async function fetchImpl(path, opts = {}) {
    const method = (opts.method || "GET").toUpperCase();
    const key = method + " " + path;
    calls.push(key);
    const route = routes[key];
    if (route === undefined) return response(404, JSON.stringify({ detail: "no route " + key }));
    const r = typeof route === "function" ? await route(opts) : route;
    if (r === "hang") {
      return new Promise((resolve, reject) => {
        if (opts.signal) opts.signal.addEventListener("abort", () => { const e = new Error("The user aborted a request."); e.name = "AbortError"; reject(e); });
      });
    }
    const text = typeof r.body === "string" ? r.body : JSON.stringify(r.body === undefined ? {} : r.body);
    return response(r.status || 200, text);
  }
  return { fetchImpl, calls, count: (key) => calls.filter((c) => c === key).length };
}

// The global object the page script runs against. `run(src)` evaluates a script with it as `window`.
// Options: fetchImpl (from makeFetch); parent, an object with postMessage() that becomes window.parent
// so voice.js sees itself inside the dashboard iframe; mic, whether navigator.mediaDevices exists.
// The handle records console.error calls in `errors` and console.log/info/warn lines in `logs`;
// performance.now() reads the virtual clock (delete sandbox.performance to test a page without it).
export function makeSandbox(doc, { fetchImpl, parent = null, mic = false } = {}) {
  const timers = makeTimers();
  const errors = [];
  const logs = [];
  const sockets = [];
  const audio = [];
  const winListeners = new Map();
  class FakeWebSocket {
    constructor(url, protocols) {
      this.url = url;
      this.protocols = protocols;
      this.readyState = 0;
      this.sent = [];
      this.onopen = null; this.onmessage = null; this.onclose = null; this.onerror = null;
      sockets.push(this);
    }
    send(s) { this.sent.push(s); }
    close() { this.readyState = 3; if (this.onclose) this.onclose({ code: 1000, reason: "" }); }
    open() { this.readyState = 1; if (this.onopen) this.onopen({}); }
    receive(obj) { if (this.onmessage) this.onmessage({ data: typeof obj === "string" ? obj : JSON.stringify(obj) }); }
  }
  FakeWebSocket.CONNECTING = 0; FakeWebSocket.OPEN = 1; FakeWebSocket.CLOSING = 2; FakeWebSocket.CLOSED = 3;
  // Web Audio, just enough for voice.js: buffers, buffer sources (recorded in `sources` with the time
  // they were started at), gain, microphone and processor nodes (recorded in `processors`, so a check
  // can feed microphone samples through onaudioprocess). currentTime follows the virtual clock, so a
  // reply "finishes playing" when a check advances the timers.
  class FakeAudioContext {
    constructor(opts) {
      this.sampleRate = opts && opts.sampleRate ? opts.sampleRate : 48000;
      this.state = "running";
      this.destination = { connect() {}, disconnect() {} };
      this.sources = [];
      this.processors = [];
      audio.push(this);
    }
    get currentTime() { return timers.now() / 1000; }
    resume() { this.state = "running"; }
    createBuffer(channels, length, rate) {
      const data = new Float32Array(length);
      return { numberOfChannels: channels, length, sampleRate: rate, duration: length / rate, getChannelData() { return data; } };
    }
    createBufferSource() {
      const src = { buffer: null, onended: null, startedAt: null, stopped: false, connect() {}, disconnect() {}, start(at) { src.startedAt = at; }, stop() { src.stopped = true; } };
      this.sources.push(src);
      return src;
    }
    createGain() { return { gain: { value: 1 }, connect() {}, disconnect() {} }; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { const p = { onaudioprocess: null, connect() {}, disconnect() {} }; this.processors.push(p); return p; }
  }
  const sandbox = {
    document: doc,
    location: { protocol: "http:", host: "127.0.0.1:8000", origin: "http://127.0.0.1:8000", href: "http://127.0.0.1:8000/" },
    navigator: mic ? { mediaDevices: { getUserMedia: async () => ({ id: "fake-mic" }) } } : {},
    console: {
      log(...a) { logs.push(a.map(String).join(" ")); },
      info(...a) { logs.push(a.map(String).join(" ")); },
      warn(...a) { logs.push(a.map(String).join(" ")); },
      error(...a) { errors.push(a.map(String).join(" ")); },
    },
    performance: { now: () => timers.now() },
    fetch: fetchImpl,
    WebSocket: FakeWebSocket,
    AudioContext: FakeAudioContext,
    AbortController,
    getComputedStyle() { return { getPropertyValue() { return ""; } }; },
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    addEventListener(type, fn) { if (!winListeners.has(type)) winListeners.set(type, []); winListeners.get(type).push(fn); },
    removeEventListener(type, fn) { if (winListeners.has(type)) winListeners.set(type, winListeners.get(type).filter((f) => f !== fn)); },
    postMessage() {},
    // node's own atob/btoa: like a browser's, atob throws InvalidCharacterError on anything that is
    // not base64 (a Buffer-based decode would silently accept junk and never exercise the guard).
    atob: (s) => atob(String(s)),
    btoa: (s) => btoa(String(s)),
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  sandbox.parent = parent || sandbox;
  sandbox.top = parent || sandbox;
  vm.createContext(sandbox);
  const fireWindow = (ev) => { for (const fn of winListeners.get(ev.type) || []) fn(ev); return ev; };
  doc.fireWindow = fireWindow;
  // Dispatches an event on the window (an Esc keydown, a message, a blur), as the browser would after bubbling.
  const fire = (type, init = {}) => fireWindow(Object.assign({ type, bubbles: true, target: doc.body, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } }, init));
  return {
    sandbox,
    doc,
    timers,
    errors,
    logs,
    sockets,
    audio,
    fire,
    run(src, filename) { return vm.runInContext(src, sandbox, { filename }); },
  };
}
