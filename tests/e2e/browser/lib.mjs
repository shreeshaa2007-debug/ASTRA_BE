// A minimal Chrome DevTools Protocol driver — no dependencies (Node's global WebSocket and fetch).
// A script gets a `b` from launch() and returns plain data; the Python tests assert on it.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const q = JSON.stringify;

export async function launch({ chrome, profile, port, artifacts, width = 1440, height = 2200 }) {
  mkdirSync(artifacts, { recursive: true });
  const flags = [
    '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--disable-gpu',
    '--disable-extensions', '--disable-background-networking', `--window-size=${width},${height}`, 'about:blank',
  ];
  if (process.env.E2E_NO_SANDBOX || (process.getuid && process.getuid() === 0)) flags.unshift('--no-sandbox');
  const proc = spawn(chrome, flags, { stdio: 'ignore' });

  let targets;
  for (let i = 0; i < 80; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.some((t) => t.type === 'page')) break; } catch { /* not up yet */ }
    await sleep(250);
  }
  const page = targets?.find((t) => t.type === 'page');
  if (!page) { proc.kill(); throw new Error('Chrome did not start'); }

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map();
  const waiters = [];
  const problems = [];
  const requests = []; // the URL of every request the page makes, so a test can prove where it talks to
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
    for (const w of waiters.slice()) if (w.method === msg.method) { waiters.splice(waiters.indexOf(w), 1); w.resolve(msg.params); }
    const p = msg.params;
    if (msg.method === 'Network.requestWillBeSent') requests.push(p.request.url);
    if (msg.method === 'Runtime.exceptionThrown') problems.push('EXCEPTION: ' + (p.exceptionDetails.exception?.description ?? p.exceptionDetails.text));
    if (msg.method === 'Runtime.consoleAPICalled' && p.type === 'error') problems.push('console.error: ' + p.args.map((a) => a.value ?? a.description).join(' ').slice(0, 400));
    if (msg.method === 'Log.entryAdded' && p.entry.level === 'error' && !/favicon\.ico/.test(p.entry.url ?? '')) problems.push('LOG: ' + p.entry.text + ' ' + (p.entry.url ?? ''));
  };
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(q({ id: i, method, params })); });
  const once = (method, ms = 15000) => new Promise((resolve) => { const w = { method, resolve }; waiters.push(w); setTimeout(() => resolve(null), ms); });
  await send('Runtime.enable'); await send('Page.enable'); await send('Log.enable'); await send('Network.enable');
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });

  const b = {
    problems, requests, sleep,
    async ev(expression) {
      const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (r.result?.exceptionDetails) throw new Error('page script failed: ' + (r.result.exceptionDetails.exception?.description ?? r.result.exceptionDetails.text));
      return r.result?.result?.value;
    },
    async goto(url) { const loaded = once('Page.loadEventFired'); await send('Page.navigate', { url }); await loaded; await sleep(400); },
    text: () => b.ev(`(document.querySelector('main') ?? document.body).innerText`),
    header: () => b.ev(`document.querySelector('header').innerText`),
    sidebar: () => b.ev(`document.querySelector('aside').innerText`),
    html: () => b.ev('document.documentElement.outerHTML'),
    storageGet: (k) => b.ev(`localStorage.getItem(${q(k)})`),
    storageSet: (k, v) => b.ev(`localStorage.setItem(${q(k)}, ${q(v)})`),
    storageRemove: (k) => b.ev(`localStorage.removeItem(${q(k)})`),
    storageClear: () => b.ev('localStorage.clear()'),
    // 'ok' | 'NO_BUTTON' | 'DISABLED'
    click: (label, { exact = false } = {}) => b.ev(`(() => {
      const want = ${q(label)}, exact = ${exact};
      const btn = [...document.querySelectorAll('button')].find((x) => { const t = x.innerText.trim(); return exact ? t === want : t.includes(want); });
      if (!btn) return 'NO_BUTTON'; if (btn.disabled) return 'DISABLED'; btn.click(); return 'ok'; })()`),
    // the sidebar item with this title (its visible label is the title)
    nav: (title) => b.ev(`(() => { const x = document.querySelector('nav button[title=${q(title)}]'); if (!x) return 'NO_NAV'; x.click(); return 'ok'; })()`),
    // sets an <input> or <textarea> whose placeholder starts with `placeholder`, through the native setter so React sees it
    type: (placeholder, value) => b.ev(`(() => {
      const el = [...document.querySelectorAll('input[placeholder], textarea[placeholder]')].find((x) => x.placeholder.startsWith(${q(placeholder)})); if (!el) return 'NO_INPUT';
      const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${q(value)}); el.dispatchEvent(new Event('input', { bubbles: true })); return 'ok'; })()`),
    select: (value) => b.ev(`(() => { const el = [...document.querySelectorAll('select')].find((s) => [...s.options].some((o) => o.value === ${q(value)}));
      if (!el) return 'NO_SELECT'; Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, ${q(value)}); el.dispatchEvent(new Event('change', { bubbles: true })); return 'ok'; })()`),
    // resolves with the first match of `re` in the page's main text, or null on timeout
    async waitFor(re, ms = 30000) {
      const t0 = Date.now();
      while (Date.now() - t0 < ms) { const m = (await b.text()).match(re); if (m) return m[0]; await sleep(300); }
      return null;
    },
    // waits for the screen to stop saying "Loading…"
    async settle(ms = 20000) {
      const t0 = Date.now();
      while (Date.now() - t0 < ms) { if (!/Loading|Solving the baseline/i.test(await b.text())) { await sleep(250); if (!/Loading|Solving the baseline/i.test(await b.text())) return true; } await sleep(250); }
      return false;
    },
    async shot(name) { const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(`${artifacts}/${name}.png`, Buffer.from(r.result.data, 'base64')); },
    // a screenshot of the whole screen: the viewport is grown to the content's height for the shot, then restored
    // (the + 84 below: the header is 68px tall and the panels float on a canvas gap)
    async shotFit(name, { minHeight = 900, maxHeight = 6000 } = {}) {
      const content = await b.ev(`(() => { const m = document.querySelector('main'); return m ? m.scrollHeight + 84 : document.documentElement.scrollHeight; })()`);
      const fitted = Math.max(minHeight, Math.min(maxHeight, Math.ceil(content)));
      await send('Emulation.setDeviceMetricsOverride', { width, height: fitted, deviceScaleFactor: 1, mobile: false });
      await sleep(400);
      await b.shot(name);
      await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
    },
    async close() { try { ws.close(); } catch { /* already closed */ } proc.kill(); },
  };
  return b;
}
