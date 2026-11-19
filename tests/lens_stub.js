// Runs the lens page's script under node against a minimal DOM stub, then prints the JSON value
// of an expression evaluated in the page's scope. Every attribute, text and HTML the page sets is
// checked for NaN, undefined and [object, each found pushed to `problems`, which the expression
// may read. Usage: node lens_stub.js PAGE.html EXPRESSION
'use strict';
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const data = /<script id="lens-data" type="application\/json">([\s\S]*?)<\/script>/.exec(html)[1];
const code = /<script>([\s\S]*?)<\/script>/.exec(html)[1];
const problems = [];
const BAD = /NaN|undefined|\[object/;
// A recording 2D context: every call is counted by name, and a call given NaN, an infinity or
// undefined, or a property set to one, is a problem.
const badArg = v => v === undefined || (typeof v === 'number' && !isFinite(v)) || (typeof v === 'string' && BAD.test(v));
function context2d() {
  const counts = {};
  const own = {
    counts, measureText: t => ({width: String(t).length * 6}),
    createRadialGradient: (...a) => { if (a.some(badArg)) problems.push('canvas createRadialGradient'); return {addColorStop() {}}; },
    createLinearGradient: () => ({addColorStop() {}}),
  };
  return new Proxy(own, {
    get: (o, k) => (k in o ? o[k] : typeof k !== 'string' ? undefined : (...a) => {
      counts[k] = (counts[k] || 0) + 1;
      if (a.some(x => badArg(x) || (Array.isArray(x) && x.some(badArg)))) problems.push(`canvas ${k}(${a.map(String).join(', ')})`);
    }),
    set: (o, k, v) => { if (badArg(v)) problems.push(`canvas ${String(k)}=${v}`); o[k] = v; return true; },
  });
}
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.attrs = {}; this._text = '';
    this._html = ''; this.style = {}; this.dataset = {}; this.hidden = false; this.disabled = false;
    this.listeners = {}; this.parent = null; this.scrollTop = 0;
    const cl = new Set(); this._cl = cl;
    this.classList = {
      add: (...c) => c.forEach(x => cl.add(x)), remove: c => cl.delete(c),
      toggle: (c, on) => ((on === undefined ? !cl.has(c) : on) ? cl.add(c) : cl.delete(c)),
      contains: c => cl.has(c),
    };
  }
  setAttribute(k, v) {
    v = String(v);
    if (BAD.test(v)) problems.push(`attr ${this.tagName}.${k}=${v}`);
    this.attrs[k] = v;
    if (k === 'class') v.split(/\s+/).filter(Boolean).forEach(c => this._cl.add(c));
  }
  getAttribute(k) { return this.attrs[k]; }
  hasAttribute(k) { return k in this.attrs; }
  removeAttribute(k) { delete this.attrs[k]; }
  appendChild(c) {
    if (c.isFrag) { c.children.forEach(x => this.appendChild(x)); c.children = []; return c; }
    c.parent = this; this.children.push(c); return c;
  }
  set textContent(v) {
    this._text = String(v); this._html = ''; this.children = [];
    if (BAD.test(this._text)) problems.push(`text ${this.tagName}: ${this._text}`);
  }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(v) {
    this._html = String(v); this._text = ''; this.children = [];
    const m = this._html.match(/.{0,60}(NaN|undefined|\[object).{0,40}/);
    if (m) problems.push(`html: ${m[0]}`);
  }
  get innerHTML() { return this._html; }
  set className(v) { this._cl.clear(); String(v).split(/\s+/).filter(Boolean).forEach(c => this._cl.add(c)); }
  get className() { return [...this._cl].join(' '); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  querySelectorAll() { return []; }
  querySelector() { return null; }
  getBoundingClientRect() { return {width: 100, height: 40}; }
  focus() {} blur() {} scrollIntoView() {} select() {} remove() {}
  getContext() { return this._ctx || (this._ctx = context2d()); }
  setPointerCapture() {} releasePointerCapture() {}
  get clientWidth() { return 1120; }
  get offsetTop() { return 0; }
  set value(v) { this._v = v; }
  get value() { return this._v; }
}
const ids = {};
const document = {
  documentElement: new El('html'), body: new El('body'),
  getElementById: id => (id === 'lens-data' ? {textContent: data} : (ids[id] = ids[id] || new El('div'))),
  createElement: t => new El(t), createElementNS: (ns, t) => new El(t),
  createTextNode: t => ({textContent: String(t)}),
  createDocumentFragment: () => Object.assign(new El('frag'), {isFrag: true}),
  querySelector: () => null, querySelectorAll: () => [], addEventListener: () => {},
};
// The elements under `el` (itself included) that carry class `cls`, depth first.
function byClass(el, cls) {
  const out = [];
  (function walk(e) { if (e._cl && e._cl.has(cls)) out.push(e); (e.children || []).forEach(c => c.children && walk(c)); })(el);
  return out;
}
// HTML as the text a reader sees.
const strip = h => h.replace(/<[^>]*>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<')
  .replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'");
// Every tooltip under `el` rendered, each one checked as an attribute is.
function tips(el) {
  const out = [];
  (function walk(e) {
    if (e.__tip) { const x = e.__tip(); out.push(x); if (typeof x !== 'string' || BAD.test(x)) problems.push(`tooltip ${String(x).slice(0, 80)}`); }
    (e.children || []).forEach(c => c.children && walk(c));
  })(el);
  return out;
}
// Animation frames wait in a queue until the expression runs them: frames(n) runs up to n.
const queued = [];
function frames(n) { let k = 0; while (queued.length && k++ < n) queued.shift()(k * 16.7); return queued.length; }
// An event of `type` dispatched to `el`'s own listeners, with the fields a handler reads.
function fire(el, type, ev) {
  const e = Object.assign({target: el, clientX: 0, clientY: 0, pointerId: 1, pointerType: 'mouse', button: 0,
    deltaY: 0, deltaMode: 0, preventDefault() {}, stopPropagation() {}}, ev);
  (el.listeners[type] || []).forEach(f => f(e));
}
const ctx = {
  document, console, setTimeout, clearTimeout, URLSearchParams, problems, byClass, strip, tips,
  localStorage: {getItem: () => null, setItem: () => {}}, innerWidth: 1200, innerHeight: 800,
  addEventListener: () => {}, requestAnimationFrame: f => queued.push(f), frames, fire,
};
vm.createContext(ctx);
vm.runInContext(code, ctx);
process.stdout.write(JSON.stringify(vm.runInContext(process.argv[3], ctx)) ?? 'null');
