/* One Neuron, No Training: page logic. Vanilla JS + inline SVG. Everything numeric comes from data/blog.json. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let D = null;
  const CALC = {};

  // ------------------------------------------------------------------ methods: one colour each, everywhere
  const METHOD = {
    base: { v: "--base", label: "Base" },
    prompt: { v: "--prompt", label: "System prompt" },
    gate: { v: "--gate", label: "One neuron" },
    random: { v: "--random", label: "Random neuron", dash: true },
    lora: { v: "--lora", label: "LoRA" },
  };
  function family(arm) {
    if (!arm) return "base";
    if (arm.startsWith("thresholds") || arm === "vad" || arm.startsWith("prompt")) return "prompt";
    const b = arm.split("_")[0];
    return METHOD[b] ? b : "base";
  }
  const col = (arm) => `var(${METHOD[family(arm)].v})`;
  const armLabel = (t, a) => (t.arms && t.arms[a] && t.arms[a].label) || (METHOD[family(a)] || {}).label || a;

  // ------------------------------------------------------------------ formatting + binding
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const F = {
    text: (v) => String(v),
    int: (v) => Math.round(v).toLocaleString("en-US"),
    pct0: (v) => (Math.abs(v) < 10 && Math.round(v * 10) % 10 !== 0 ? v.toFixed(1) : Math.round(v)) + "%",
    pct1: (v) => v.toFixed(1) + "%",
    f2: (v) => v.toFixed(2),
    f2floor: (v) => (Math.floor(v * 100) / 100).toFixed(2),
    f4: (v) => v.toFixed(4),
    keep: (v) => (v >= 99.95 ? "100%" : v >= 99 ? v.toFixed(1) + "%" : Math.round(v) + "%"),
    kl: (v) => (v === 0 ? "0" : v >= 0.1 ? v.toFixed(2) : Number(v.toPrecision(1)).toString()),
    kl2: (v) => (v === 0 ? "0" : v >= 0.1 ? v.toFixed(2) : Number(v.toPrecision(2)).toString()),
    x0: (v) => Math.round(v) + "×",
    billions: (v) => (v / 1e9).toFixed(1).replace(/\.0$/, "") + " billion",
    compact: (v) => (v >= 1e9 ? (v / 1e9).toFixed(1) + "B" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e4 ? (v / 1e3).toFixed(0) + "K" : Math.round(v).toLocaleString("en-US")),
  };
  const PENDING = '<span class="pending" title="Still being computed; this fills in when results land">pending</span>';
  function get(path) {
    const parts = path.split(".");
    let o;
    if (parts[0] === "calc") { o = CALC; parts.shift(); }
    else if (D.tasks && parts[0] in D.tasks) { o = D.tasks; }
    else if (D.tasks && MAIN_TASK_IDS.includes(parts[0])) return undefined;
    else o = D;
    for (const p of parts) { if (o == null) return undefined; o = o[p]; }
    return o;
  }
  const MAIN_TASK_IDS = ["entity", "agent_injection", "signoff", "whisper_silence"];
  function fmtv(v, f) {
    if (v === null || v === undefined || v === "" || (typeof v === "number" && !isFinite(v))) return null;
    if (f !== "text" && !isNum(v)) return null;
    try { return (F[f] || F.text)(v); } catch (e) { return null; }
  }
  function bindAll(root = document) {
    $$("[data-k]", root).forEach((el) => {
      const s = fmtv(get(el.dataset.k), el.dataset.f || "text");
      if (s === null) el.innerHTML = PENDING; else el.textContent = s;
    });
    const repo = D.site && D.site.repo;
    $$("[data-repo]").forEach((a) => { if (repo) a.href = repo; else a.title = "repo URL pending"; });
    $$("[data-repo-path]").forEach((a) => {
      if (repo) a.href = repo.replace(/\/$/, "") + "/tree/main/" + a.dataset.repoPath;
      else a.title = "repo URL pending";
    });
  }

  // ------------------------------------------------------------------ svg helpers
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) { const t = el("text", Object.assign({ x, y }, attrs || {}), parent); t.textContent = s; return t; }
  function scale(d0, d1, r0, r1, log) {
    const f = log ? Math.log : (v) => v;
    const a = f(d0), b = f(d1);
    const s = (v) => r0 + ((f(v) - a) / (b - a || 1)) * (r1 - r0);
    s.inv = (p) => { const t = a + ((p - r0) / (r1 - r0 || 1)) * (b - a); return log ? Math.exp(t) : t; };
    return s;
  }
  function niceTicks(min, max, n) {
    const span = max - min, step0 = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const W = (node) => Math.max(280, Math.floor(node.clientWidth || node.getBoundingClientRect().width || 600));
  function svgRoot(node, w, h, label) {
    node.innerHTML = "";
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img", "aria-label": label || "" }, node);
    return s;
  }
  function pendingFig(node, what, where) {
    node.innerHTML = `<div class="pending-fig"><b>Pending: ${what}</b>This figure fills in automatically when <code>${esc(where)}</code> is written and <code>site/build_data.py</code> is re-run.</div>`;
  }

  // tooltip
  const tip = () => $("#tip");
  function showTip(html, clientX, clientY) {
    const t = tip();
    t.innerHTML = html;
    t.classList.add("on");
    const pad = 14, r = t.getBoundingClientRect();
    let x = clientX + pad + window.scrollX, y = clientY + pad + window.scrollY;
    if (clientX + pad + r.width > window.innerWidth - 8) x = clientX - r.width - pad + window.scrollX;
    if (x < 8 + window.scrollX) x = 8 + window.scrollX;
    if (clientY + pad + r.height > window.innerHeight - 8) y = clientY - r.height - pad + window.scrollY;
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  const hideTip = () => tip().classList.remove("on");
  const swatch = (c, dash) => dash ? `<i class="key dash" style="border-color:${c}"></i>` : `<i class="sw" style="background:${c}"></i>`;
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function legend(node, items) {
    node.innerHTML = items.map((i) => `<span>${i.line ? `<i class="key${i.dash ? " dash" : ""}" style="border-color:${i.color}"></i>` : `<i class="sw" style="background:${i.color}"></i>`}${esc(i.label)}</span>`).join("");
  }

  // segmented control
  function seg(node, options, current, onPick) {
    node.innerHTML = "";
    options.forEach((o) => {
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = esc(o.label) + (o.pending ? " " + PENDING : "");
      b.disabled = !!o.disabled;
      b.setAttribute("aria-pressed", String(o.id === current));
      b.addEventListener("click", () => {
        $$("button", node).forEach((x) => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        onPick(o.id);
      });
      node.appendChild(b);
    });
  }

  // ------------------------------------------------------------------ generic line chart
  /* opts: series [{id,label,color,dash,points:[[x,y]],band:[[x,lo,hi]],width,endLabel}], x{min,max,log,ticks,fmt,label},
     y{min,max,log,ticks,fmt,label}, height, refs [{y,label,color}], vlines [{x,label}], tipTitle(x), tipFmt(v), snapXs */
  function lineChart(node, o) {
    const w = W(node), narrow = w < 560;
    const h = o.height || (narrow ? 260 : 330);
    const showEnd = !narrow && o.series.length <= 5;
    const m = { l: 46, r: showEnd ? 112 : 14, t: 14, b: 40 };
    const svg = svgRoot(node, w, h, o.aria);
    const X = scale(o.x.min, o.x.max, m.l, w - m.r, o.x.log);
    const Y = scale(o.y.min, o.y.max, h - m.b, m.t, o.y.log);
    const g = el("g", { class: "ax" }, svg);
    const yt = o.y.ticks || niceTicks(o.y.min, o.y.max, narrow ? 4 : 5);
    yt.forEach((v) => {
      el("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: "gridl" }, g);
      txt(g, m.l - 6, Y(v) + 4, (o.y.fmt || String)(v), { "text-anchor": "end", class: "tick" });
    });
    el("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: "var(--axis)" }, g);
    let xt = o.x.ticks || niceTicks(o.x.min, o.x.max, narrow ? 4 : 6);
    if (narrow && xt.length > 6) xt = xt.filter((_, i) => i % 2 === 0);
    xt.forEach((v) => {
      el("line", { x1: X(v), x2: X(v), y1: h - m.b, y2: h - m.b + 4, stroke: "var(--axis)" }, g);
      txt(g, X(v), h - m.b + 16, (o.x.fmt || String)(v), { "text-anchor": "middle", class: "tick" });
    });
    if (o.x.label) txt(g, (m.l + w - m.r) / 2, h - 4, o.x.label, { "text-anchor": "middle", class: "tick" });
    if (o.y.label) txt(g, m.l - 40, m.t - 2, "", {});
    (o.refs || []).forEach((r) => {
      if (!isNum(r.y) || r.y < o.y.min || r.y > o.y.max) return;
      el("line", { x1: m.l, x2: w - m.r, y1: Y(r.y), y2: Y(r.y), style: `stroke:${r.color || "var(--base)"}`, "stroke-width": 1.25, "stroke-dasharray": r.dash ? "4 3" : null, opacity: 0.9 }, svg);
      txt(svg, m.l + 4, Y(r.y) - 5, r.label, { "text-anchor": "start", class: "annot" });
    });
    (o.vlines || []).forEach((r) => {
      el("line", { x1: X(r.x), x2: X(r.x), y1: m.t, y2: h - m.b, stroke: "var(--ink-2)", "stroke-width": 1 }, svg);
      txt(svg, X(r.x) + 5, m.t + 10, r.label, { class: "annot" });
    });
    const clampY = (v) => Math.max(o.y.min, Math.min(o.y.max, v));
    o.series.forEach((s) => {
      if (s.band && s.band.length > 1) {
        const top = s.band.map((p) => `${X(p[0])},${Y(clampY(p[2]))}`).join("L");
        const bot = s.band.slice().reverse().map((p) => `${X(p[0])},${Y(clampY(p[1]))}`).join("L");
        el("path", { d: `M${top}L${bot}Z`, style: `fill:${s.color}`, opacity: 0.13 }, svg);
      }
    });
    o.series.forEach((s) => {
      const pts = s.points.filter((p) => isNum(p[1]));
      if (!pts.length) return;
      const d = "M" + pts.map((p) => `${X(p[0]).toFixed(1)},${Y(clampY(p[1])).toFixed(1)}`).join("L");
      el("path", { d, fill: "none", style: `stroke:${s.color}`, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash ? "5 4" : null }, svg);
      if (s.markers) pts.forEach((p, i) => (s.hollow && i === 0) ? null : el("circle", { cx: X(p[0]), cy: Y(clampY(p[1])), r: s.hollow ? 4 : 3.5, style: s.hollow ? `fill:var(--surface);stroke:${s.color}` : `fill:${s.color};stroke:var(--surface)`, "stroke-width": 2 }, svg));
      if (s.endDot) { const p = pts[pts.length - 1]; el("circle", { cx: X(p[0]), cy: Y(clampY(p[1])), r: 4, style: `fill:${s.color};stroke:var(--surface)`, "stroke-width": 2 }, svg); }
    });
    // end labels with simple collision avoidance (drop, don't nudge)
    if (showEnd) {
      const labs = o.series.filter((s) => s.endLabel !== false && s.points.some((p) => isNum(p[1]))).map((s) => {
        const pts = s.points.filter((p) => isNum(p[1])); const p = pts[pts.length - 1];
        return { s, x: X(p[0]), y: Y(clampY(p[1])) };
      }).sort((a, b) => (a.s.priority || 9) - (b.s.priority || 9));
      const placed = [];
      labs.forEach((L) => {
        if (placed.some((p) => Math.abs(p.y - L.y) < 13 && Math.abs(p.x - L.x) < 100)) return;
        placed.push(L);
        txt(svg, L.x + 8, L.y + 4, L.s.short || L.s.label, { class: "lbl-direct" });
      });
    }
    (o.notes || []).forEach((n) => {
      if (narrow && n.wideOnly) return;
      const x = X(n.x), y = Y(n.y);
      if (n.dot) el("circle", { cx: x, cy: y, r: 4, style: `fill:${n.color || "var(--ink-2)"};stroke:var(--surface)`, "stroke-width": 2 }, svg);
      txt(svg, x + (n.dx || 0), y + (n.dy || 0), n.text, { class: "annot", "text-anchor": n.anchor || "start" });
    });
    // hover layer
    const hov = el("g", { opacity: 0 }, svg);
    const vl = el("line", { y1: m.t, y2: h - m.b, stroke: "var(--ink-2)", "stroke-width": 1, opacity: 0.5 }, hov);
    const dots = o.series.map((s) => el("circle", { r: 4.5, style: `fill:${s.color};stroke:var(--surface)`, "stroke-width": 2 }, hov));
    const hit = el("rect", { x: m.l, y: 0, width: w - m.l - m.r, height: h - m.b, fill: "transparent", style: "cursor:crosshair" }, svg);
    function valueAt(s, x) {
      const pts = s.points.filter((p) => isNum(p[1]));
      if (!pts.length || x < pts[0][0] - 1e-9 || x > pts[pts.length - 1][0] + 1e-9) return null;
      if (o.snapXs) { const p = pts.find((q) => q[0] === x); return p ? p[1] : null; }
      let i = pts.findIndex((p) => p[0] >= x);
      if (i <= 0) return pts[Math.max(0, i)][1];
      const a = pts[i - 1], b = pts[i];
      return a[1] + ((x - a[0]) / (b[0] - a[0] || 1)) * (b[1] - a[1]);
    }
    function move(ev) {
      const pt = ev.touches ? ev.touches[0] : ev;
      const r = svg.getBoundingClientRect();
      const px = ((pt.clientX - r.left) / r.width) * w;
      let x = X.inv(Math.max(m.l, Math.min(w - m.r, px)));
      if (o.snapXs) x = o.snapXs.reduce((best, v) => (Math.abs(X(v) - px) < Math.abs(X(best) - px) ? v : best), o.snapXs[0]);
      else if (o.x.step) x = Math.round(x / o.x.step) * o.x.step;
      hov.setAttribute("opacity", 1);
      vl.setAttribute("x1", X(x)); vl.setAttribute("x2", X(x));
      const rows = [];
      o.series.forEach((s, i) => {
        const v = valueAt(s, x);
        if (v === null) { dots[i].setAttribute("opacity", 0); return; }
        dots[i].setAttribute("opacity", 1); dots[i].setAttribute("cx", X(x)); dots[i].setAttribute("cy", Y(clampY(v)));
        let extra = "";
        if (s.band && o.snapXs) { const b = s.band.find((q) => q[0] === x); if (b && b[1] !== b[2]) extra = ` <span style="color:var(--muted);font-weight:400">(${(o.tipFmt || String)(b[1])}–${(o.tipFmt || String)(b[2])})</span>`; }
        rows.push({ v, html: `<div class="r"><span>${swatch(s.color, s.dash)}${esc(s.label)}</span><b>${(o.tipFmt || String)(v)}${extra}</b></div>` });
      });
      (o.refs || []).forEach((r) => { if (isNum(r.y) && r.inTip !== false) rows.push({ v: r.y, html: `<div class="r"><span>${swatch(r.color || "var(--base)", true)}${esc(r.label)}</span><b>${(o.tipFmt || String)(r.y)}</b></div>` }); });
      rows.sort((a, b) => (o.y.descGood ? a.v - b.v : b.v - a.v));
      showTip(`<div class="t">${esc(o.tipTitle ? o.tipTitle(x) : x)}</div>` + rows.map((r) => r.html).join(""), pt.clientX, pt.clientY);
    }
    const leave = () => { hov.setAttribute("opacity", 0); hideTip(); };
    hit.addEventListener("mousemove", move);
    hit.addEventListener("mouseleave", leave);
    hit.addEventListener("touchstart", move, { passive: true });
    hit.addEventListener("touchmove", move, { passive: true });
    hit.addEventListener("touchend", leave);
    return svg;
  }

  // responsive redraw
  const redraws = [];
  function onResize(node, fn) {
    redraws.push(fn);
    let lastW = node.clientWidth;
    if (window.ResizeObserver) {
      let t;
      new ResizeObserver(() => {
        if (Math.abs(node.clientWidth - lastW) < 4) return;
        lastW = node.clientWidth; clearTimeout(t); t = setTimeout(fn, 80);
      }).observe(node);
    }
  }

  // ------------------------------------------------------------------ §1 mechanisms
  const MECH = [
    ["Grokking MLP · 37k params", "A Fourier circuit", "The network adds numbers mod 97 by rotating through a handful of frequencies; the agent read them off the weights."],
    ["Qwen3.5-0.8B", "A “hedging” direction", "One direction in the residual stream carries the model’s tendency to waffle instead of answering."],
    ["Small transformer · 10.8M", "A missing induction circuit", "Stuck on a loss plateau because the copy-from-earlier-in-context circuit hadn’t formed yet."],
    ["ViT · CIFAR-10", "Uniform attention", "Early attention heads looked everywhere equally: no locality prior yet."],
    ["ResNet-50 · ImageNet", "Context channels", "A few channels that respond to backgrounds, not objects, drive the spurious predictions."],
    ["Whisper", "A “no speech here” signal", "The decoder already knows a clip is silence or music, and transcribes words anyway."],
  ];
  function renderMech() {
    $("#mech-grid").innerHTML = MECH.map((m) => `<div class="card mech"><div class="m">${esc(m[0])}</div><div class="w">${esc(m[1])}</div><div class="d">${esc(m[2])}</div></div>`).join("");
  }

  // ------------------------------------------------------------------ §2 grokking + track A
  let grokZoom = 60000;
  function renderGrok() {
    const node = $("#grok-chart");
    const g = D.archive && D.archive.grokking;
    if (!g) return pendingFig(node, "grokking curves", "site/data/archive/grokking.json");
    const S = g.series, L = g.labels || {};
    const series = [];
    ["random0", "random1", "random2"].forEach((k, i) => S[k] && series.push({ id: k, label: i === 0 ? "Random edit, same size" : `Random edit ${i + 1}`, short: i === 0 ? "Random edits" : "", endLabel: i === 0, color: "var(--random)", dash: true, points: S[k], priority: 5 }));
    if (S.unedited) series.push({ id: "unedited", label: "No edit (plain training)", short: "No edit", endLabel: grokZoom < 60000, color: "var(--base)", points: S.unedited, endDot: grokZoom >= 60000, width: 2.25, priority: 2 });
    if (S.opus) series.push({ id: "opus", label: "Agent’s edit", short: "Agent’s edit", color: "var(--gate)", points: S.opus, width: 2.5, priority: 1 });
    const clip = (pts) => pts.filter((p) => p[0] <= grokZoom);
    series.forEach((s) => (s.points = clip(s.points)));
    legend($("#grok-legend"), [
      { label: "Agent’s edit", color: "var(--gate)", line: true },
      { label: "No edit (plain training)", color: "var(--base)", line: true },
      { label: "Random edits, same size (3 draws)", color: "var(--random)", line: true, dash: true },
    ]);
    const zoomed = grokZoom < 60000;
    const notes = [];
    const un = S.unedited || [];
    if (!zoomed && un.length) { const p = un[un.length - 1]; notes.push({ x: p[0], y: p[1], text: "No edit: reaches 95%, run stops", dy: -10, dx: -6, anchor: "end", wideOnly: true }); }
    if (!zoomed && S.opus) { const p = S.opus[S.opus.length - 1]; notes.push({ x: p[0], y: p[1], text: `plateaus at ${Math.round(p[1])}%`, dy: 20, anchor: "end", dx: -4 }); }
    lineChart(node, {
      aria: "Held-out accuracy after the edit: edited run leads early, plateaus below 95%; plain training reaches 95%.",
      series, notes,
      x: { min: 0, max: grokZoom, fmt: (v) => (v === 0 ? "edit" : "+" + (v / 1000) + "k"), label: "training steps after the edit", step: 100 },
      y: { min: 0, max: 100, ticks: [0, 25, 50, 75, 95], fmt: (v) => v + "%" },
      refs: [{ y: 95, label: "solved (95%)", color: "var(--axis)", inTip: false }],
      tipTitle: (x) => (x === 0 ? "at the edit" : `+${x.toLocaleString("en-US")} steps`),
      tipFmt: (v) => v.toFixed(1) + "%",
    });
  }
  const TRACK_A_TEXT = [
    ["Grokking MLP (3 pilots, 24 sessions)", "reached 50% held-out 1.8–3.4× sooner", "never reached 95%; plain training did; random edits did better long-run", "pts"],
    ["Qwen3.5-0.8B, then RL", "+7 points at step 0", "gone within ~10–15 GRPO steps", "pts"],
    ["Small LM stuck on a plateau", "~650 steps ahead (biggest effect we saw)", "gap 0.006 nats at the end: noise", "nats"],
    ["ViT locality prior", "the structure survived", "accuracy level with no edit (−0.1 points)", "pts"],
  ];
  function renderTrackA() {
    const rows = (D.archive && D.archive.track_a) || [];
    const fmt = (v, u) => !isNum(v) ? PENDING : (v > 0 ? "+" : v < 0 ? "−" : "") + (u === "nats" ? Math.abs(v).toFixed(3) : Math.abs(v).toFixed(1)) + (u === "nats" ? " nats" : " pts");
    $("#tracka tbody").innerHTML = TRACK_A_TEXT.map((t, i) => {
      const r = rows[i] || {};
      return `<tr><td><b>${esc(t[0])}</b></td><td data-l="What the edit gave">${esc(t[1])}</td><td data-l="What was left after more training">${esc(t[2])}</td><td class="n" data-l="Peak lead → at end">${fmt(r.peak_lead, t[3])} <span class="dim">&rarr;</span> <b>${fmt(r.final_lead, t[3])}</b></td></tr>`;
    }).join("");
  }

  // ------------------------------------------------------------------ §3 one-neuron diagram
  const NEU = { case: "fake", step: 1, timer: null, played: false };
  function neuronCase() {
    const t = D.tasks.entity || {};
    const ex = (t.examples || []);
    const fx = ex.find((e) => e.kind === "fixed") || {};
    const kb = ex.find((e) => e.kind === "kept_break") || {};
    const theta = (t.gate_meta && t.gate_meta.theta) || 6.35;
    const first = (s, n) => (s || "").replace(/\s+/g, " ").trim().slice(0, n) + "…";
    return {
      theta,
      fake: {
        prompt: fx.prompt || "Give me a short bio of Philippa Garrowani.", name: fx.name || "Philippa Garrowani",
        score: theta + 3.4,
        before: [["Okay", 0.58], ["Phil", 0.19], ["I", 0.09], ["Sorry", 0.03]],
        after: [["Sorry", 0.93], ["I", 0.03], ["Phil", 0.02], ["Okay", 0.0]],
        replyBase: first(fx.replies && fx.replies.base, 120) || "Okay, here's a short bio of Philippa Garrowani…",
        replyGate: first(fx.replies && fx.replies.gate && fx.replies.gate.replace(/^Sorry,Sorry,/, "Sorry,"), 120) || "Sorry, there doesn't appear to be a widely known public figure named Philippa Garrowani…",
      },
      real: {
        prompt: kb.prompt || "What is Chance the Rapper known for?", name: kb.name || "Chance the Rapper",
        score: theta - 4.6,
        before: [["Chance", 0.71], ["Okay", 0.17], ["I", 0.05], ["Sorry", 0.01]],
        after: [["Chance", 0.71], ["Okay", 0.17], ["I", 0.05], ["Sorry", 0.01]],
        replyBase: first(kb.replies && kb.replies.base, 120) || "Chance the Rapper, born Chancelor Marshall…",
        replyGate: first(kb.replies && kb.replies.gate, 120) || "Chance the Rapper, born Chancelor Marshall…",
      },
    };
  }
  function wrapText(parent, x, y, s, maxChars, lh, attrs) {
    const words = s.split(" "); const lines = []; let cur = "";
    words.forEach((w) => { if ((cur + " " + w).trim().length > maxChars) { lines.push(cur.trim()); cur = w; } else cur += " " + w; });
    if (cur.trim()) lines.push(cur.trim());
    const t = el("text", Object.assign({ x, y }, attrs || {}), parent);
    lines.forEach((l, i) => { const ts = el("tspan", { x, dy: i === 0 ? 0 : lh }, t); ts.textContent = l; });
    return lines.length;
  }
  function renderNeuron() {
    const node = $("#neuron-chart");
    if (!D.tasks.entity) return;
    const C = neuronCase(), c = C[NEU.case], step = NEU.step;
    const fires = c.score > C.theta;
    const w = W(node), tall = w < 600;
    const VW = tall ? 360 : 760, VH = tall ? 640 : 330;
    node.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${VW} ${VH}`, role: "img", "aria-label": `Diagram: the neuron ${fires ? "fires" : "stays silent"} on “${c.prompt}”` }, node);
    const on = (s) => (step >= s ? 1 : 0.28);
    const P = tall
      ? { pr: [12, 12, 336, 70], cells: { x0: 36, y: 118, horiz: true }, neu: [180, 232], gauge: [70, 300, 220], logit: { x: 112, y0: 400, gap: 34, wmax: 200, lx: 100 }, reply: [12, 548, 336, 84], wr: [180, 268] }
      : { pr: [10, 40, 200, 96], cells: { x0: 262, y: 34, horiz: false }, neu: [400, 132], gauge: [340, 210, 120], logit: { x: 600, y0: 78, gap: 36, wmax: 140, lx: 588 }, reply: [10, 262, 740, 58], wr: [490, 132] };
    // prompt
    el("rect", { x: P.pr[0], y: P.pr[1], width: P.pr[2], height: P.pr[3], rx: 10, style: "fill:var(--surface-2)" }, svg);
    txt(svg, P.pr[0] + 12, P.pr[1] + 18, "USER PROMPT", { style: "font:700 9.5px var(--sans);letter-spacing:.08em;fill:var(--muted)" });
    wrapText(svg, P.pr[0] + 12, P.pr[1] + 38, c.prompt, tall ? 44 : 26, 17, { style: "font:600 13px var(--sans);fill:var(--ink)" });
    // activation cells
    const nCells = tall ? 16 : 12;
    let seed = NEU.case === "fake" ? 7 : 3;
    const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
    const cells = [];
    const ga = el("g", { opacity: on(1), class: "fade" }, svg);
    for (let i = 0; i < nCells; i++) {
      const v = rnd() * 2 - 1;
      const cx = P.cells.horiz ? P.cells.x0 + i * 18 : P.cells.x0, cy = P.cells.horiz ? P.cells.y : P.cells.y + i * 16;
      el("rect", { x: cx, y: cy, width: 14, height: 12, rx: 2, style: `fill:${v > 0 ? "var(--g-good)" : "var(--g-fail)"}`, opacity: 0.25 + Math.abs(v) * 0.6 }, ga);
      cells.push([cx + 7, cy + 6]);
    }
    if (P.cells.horiz) txt(ga, P.cells.x0, P.cells.y - 8, "x: activations at the last prompt token", { class: "annot" });
    else { txt(ga, P.cells.x0 + 7, P.cells.y - 10, "x", { class: "annot", "text-anchor": "middle", style: "font-style:italic" }); }
    // arrow prompt -> cells
    if (!tall) el("path", { d: `M${P.pr[0] + P.pr[2] + 6},${P.pr[1] + P.pr[3] / 2} L${P.cells.x0 - 10},${P.pr[1] + P.pr[3] / 2}`, stroke: "var(--axis)", "stroke-width": 1.5, "marker-end": "url(#arr)" }, svg);
    else el("path", { d: `M180,${P.pr[1] + P.pr[3] + 6} L180,${P.cells.y - 22}`, stroke: "var(--axis)", "stroke-width": 1.5, "marker-end": "url(#arr)" }, svg);
    const defs = el("defs", {}, svg);
    const mk = el("marker", { id: "arr", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", style: "fill:var(--axis)" }, mk);
    // read lines d (cells -> neuron)
    const [nx, ny] = P.neu, R = tall ? 30 : 34;
    const gr = el("g", { opacity: step >= 1 ? 1 : 0.25, class: "fade" }, svg);
    cells.forEach((p, i) => {
      const wgt = 0.3 + ((i * 37) % 10) / 14;
      el("line", { x1: p[0] + (P.cells.horiz ? 0 : 8), y1: p[1] + (P.cells.horiz ? 7 : 0), x2: nx + (P.cells.horiz ? 0 : -R), y2: ny + (P.cells.horiz ? -R : 0), style: "stroke:var(--gate)", "stroke-width": wgt, opacity: step === 1 ? 0.75 : 0.35 }, gr);
    });
    txt(gr, tall ? nx + R + 10 : nx, tall ? ny - 6 : ny - R - 14, tall ? "reads d·x" : "reads d·x (d = probe direction)", { class: "annot", "text-anchor": tall ? "start" : "middle" });
    // neuron
    const lit = step >= 2 && fires;
    el("circle", { cx: nx, cy: ny, r: R + 8, style: "fill:var(--gate-wash)", opacity: lit ? 1 : 0, class: "node" }, svg);
    el("circle", { cx: nx, cy: ny, r: R, class: "node", style: `fill:${lit ? "var(--gate)" : "var(--surface)"};stroke:${step >= 2 && !fires ? "var(--base)" : "var(--gate)"}`, "stroke-width": 2.5 }, svg);
    txt(svg, nx, ny + 5, "d·x", { "text-anchor": "middle", style: `font:700 15px var(--sans);fill:${lit ? "#fff" : "var(--ink)"}` });
    // gauge
    const [gx, gy, gw] = P.gauge;
    const gmax = C.theta * 2;
    const gs = (v) => gx + Math.max(0, Math.min(1, v / gmax)) * gw;
    const gg = el("g", { opacity: on(2), class: "fade" }, svg);
    el("rect", { x: gx, y: gy, width: gw, height: 10, rx: 5, style: "fill:var(--surface-2)" }, gg);
    el("rect", { x: gx, y: gy, width: step >= 2 ? Math.max(4, gs(c.score) - gx) : 4, height: 10, rx: 5, class: "gfill", style: `fill:${fires ? "var(--gate)" : "var(--base)"}` }, gg);
    el("line", { x1: gs(C.theta), x2: gs(C.theta), y1: gy - 6, y2: gy + 16, stroke: "var(--ink)", "stroke-width": 2 }, gg);
    txt(gg, gs(C.theta), gy - 10, "θ = " + C.theta.toFixed(2), { "text-anchor": "middle", class: "annot" });
    txt(gg, gx + gw / 2, gy + 32, fires ? `score ${c.score.toFixed(1)} > θ: fires` : `score ${c.score.toFixed(1)} < θ: stays silent`, { "text-anchor": "middle", style: `font:650 12px var(--sans);fill:${fires ? "var(--gate)" : "var(--ink-2)"}` });
    // write arrow
    const wl = el("g", { opacity: step >= 3 && fires ? 1 : 0.22, class: "fade" }, svg);
    if (!tall) {
      el("path", { d: `M${nx + R + 6},${ny} L${P.logit.lx - 60},${ny}`, style: "stroke:var(--gate)", "stroke-width": 2.5, "marker-end": "url(#arrg)" }, wl);
      txt(wl, (nx + R + P.logit.lx - 60) / 2, ny + 22, "writes", { "text-anchor": "middle", class: "annot" });
      txt(wl, (nx + R + P.logit.lx - 60) / 2, ny + 38, "+Sorry −Okay", { "text-anchor": "middle", class: "annot" });
    } else {
      el("path", { d: `M${nx},${gy + 44} L${nx},${P.logit.y0 - 34}`, style: "stroke:var(--gate)", "stroke-width": 2.5, "marker-end": "url(#arrg)" }, wl);
      txt(wl, nx + 10, gy + 66, "writes: +Sorry, −Okay", { class: "annot" });
    }
    const mk2 = el("marker", { id: "arrg", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", style: "fill:var(--gate)" }, mk2);
    // logits
    const Lg = P.logit;
    const rows = step >= 3 ? c.after : c.before;
    const order = c.before.map((r) => r[0]);
    txt(svg, Lg.lx - 52, Lg.y0 - 22, "NEXT-TOKEN PROBABILITY", { style: "font:700 9.5px var(--sans);letter-spacing:.08em;fill:var(--muted)" });
    order.forEach((tok, i) => {
      const v = (rows.find((r) => r[0] === tok) || [tok, 0])[1];
      const y = Lg.y0 + i * Lg.gap;
      const isS = tok === "Sorry";
      txt(svg, Lg.lx, y + 12, `“${tok}”`, { "text-anchor": "end", style: `font:${isS ? 700 : 500} 12.5px var(--sans);fill:var(--ink)` });
      el("rect", { x: Lg.x, y, width: Lg.wmax, height: 16, rx: 3, style: "fill:var(--surface-2)" }, svg);
      el("rect", { x: Lg.x, y, width: Math.max(2, v * Lg.wmax), height: 16, rx: 3, class: "bar-h", style: `fill:${isS && step >= 3 && fires ? "var(--gate)" : "var(--base)"}` }, svg);
      txt(svg, Lg.x + Lg.wmax + 6, y + 12, Math.round(v * 100) + "%", { class: "annot", style: "font-variant-numeric:tabular-nums" });
    });
    // reply
    const [rx, ry, rw, rh] = P.reply;
    el("rect", { x: rx, y: ry, width: rw, height: rh, rx: 10, style: `fill:${step >= 3 && fires ? "var(--gate-wash)" : "var(--surface-2)"}` }, svg);
    txt(svg, rx + 12, ry + 18, step >= 3 ? (fires ? "REPLY WITH THE PATCH" : "REPLY (PATCH SILENT, UNCHANGED)") : "REPLY WITHOUT THE PATCH", { style: "font:700 9.5px var(--sans);letter-spacing:.08em;fill:var(--muted)" });
    wrapText(svg, rx + 12, ry + 38, step >= 3 ? c.replyGate : c.replyBase, tall ? 46 : 104, 17, { style: "font:13px var(--sans);fill:var(--ink)" });
    $$("#fig-neuron [data-step]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.step === step)));
    $$("#fig-neuron [data-case]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.case === NEU.case)));
  }
  function neuronPlay() {
    clearInterval(NEU.timer);
    NEU.step = 1; renderNeuron();
    NEU.timer = setInterval(() => {
      if (NEU.step >= 3) { clearInterval(NEU.timer); NEU.timer = null; return; }
      NEU.step++; renderNeuron();
    }, 1500);
  }
  function setupNeuron() {
    if (!D.tasks.entity || D.tasks.entity.status === "pending") return;
    $$("#fig-neuron [data-step]").forEach((b) => b.addEventListener("click", () => { clearInterval(NEU.timer); NEU.played = true; NEU.step = +b.dataset.step; renderNeuron(); }));
    $$("#fig-neuron [data-case]").forEach((b) => b.addEventListener("click", () => { NEU.played = true; NEU.case = b.dataset.case; neuronPlay(); }));
    $("#neuron-play").addEventListener("click", () => { NEU.played = true; neuronPlay(); });
    NEU.step = reduceMotion ? 3 : 1;
    renderNeuron();
    onResize($("#neuron-chart"), renderNeuron);
    if (!reduceMotion && "IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting) && !NEU.played) { NEU.played = true; neuronPlay(); io.disconnect(); } }, { threshold: 0.6 });
      io.observe($("#neuron-chart"));
    } else NEU.step = 3;
  }

  // ------------------------------------------------------------------ task helpers
  const tasksList = () => (D.task_order || []).map((id) => D.tasks[id]).filter((t) => t && (!t.group || t.group === "main"));
  const sameOf = (t, a) => (t.arms && t.arms[a] && isNum(t.arms[a].same) ? t.arms[a].same : null);
  const shortName = (t) => (t.meta && t.meta.short) || t.id;
  function pickDefault(list, pred) {
    const ok = list.filter(pred);
    return (ok.find((t) => t.id === "entity") || ok[0] || {}).id;
  }

  // ------------------------------------------------------------------ anatomy
  let anaTask = null;
  function renderAnatomy() {
    const node = $("#anatomy-chart");
    const list = tasksList().filter((t) => t.anatomy);
    if (!list.length) {
      $("#anatomy-legend").innerHTML = "";
      return pendingFig(node, "the neuron’s score distribution", "results/<task>/anatomy.json");
    }
    if (!anaTask || !list.find((t) => t.id === anaTask)) anaTask = pickDefault(list, () => true);
    if (list.length > 1) { const s = document.createElement("div"); s.className = "seg"; $("#anatomy-tasks").innerHTML = ""; $("#anatomy-tasks").appendChild(s); seg(s, list.map((t) => ({ id: t.id, label: shortName(t) })), anaTask, (id) => { anaTask = id; renderAnatomy(); }); }
    const t = D.tasks[anaTask], A = t.anatomy;
    if (anaTask === "entity") $("#anatomy-takeaway").textContent = `The neuron fires on ${F.pct0(100 * A.hist.fail.fire)} of held-out made-up names and on ${F.pct0(100 * A.hist.content.fire)} of ordinary chat tokens.`;
    const names = anaTask.startsWith("entity") ? { fail: "Made-up entities", good: "Real entities it knows", content: "Ordinary reply tokens" }
      : { fail: "Failure cases", good: "Good cases", content: "Ordinary reply tokens" };
    const G = [["content", "var(--g-content)"], ["good", "var(--g-good)"], ["fail", "var(--g-fail)"]];
    legend($("#anatomy-legend"), G.slice().reverse().map(([k, c]) => ({ label: `${names[k]}: fires on ${A.hist[k].fired.toLocaleString()} of ${A.hist[k].n.toLocaleString()}`, color: c })));
    const e = A.edges, nb = e.length - 1;
    const share = {}; let ymax = 0;
    G.forEach(([k]) => { const h = A.hist[k]; share[k] = h.counts.map((c) => (100 * c) / Math.max(1, h.n)); ymax = Math.max(ymax, ...share[k]); });
    const w = W(node), narrow = w < 560, h = narrow ? 240 : 290, m = { l: 42, r: 14, t: 18, b: 40 };
    const svg = svgRoot(node, w, h, "Histogram of the gate neuron's score by group");
    const X = scale(e[0], e[nb], m.l, w - m.r), Y = scale(0, Math.ceil(ymax / 5) * 5 || 5, h - m.b, m.t);
    const ga = el("g", { class: "ax" }, svg);
    niceTicks(0, Math.ceil(ymax / 5) * 5 || 5, 4).forEach((v) => { el("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: "gridl" }, ga); txt(ga, m.l - 6, Y(v) + 4, v + "%", { "text-anchor": "end", class: "tick" }); });
    niceTicks(e[0], e[nb], narrow ? 5 : 8).forEach((v) => txt(ga, X(v), h - m.b + 16, (v > 0 ? "+" : "") + v, { "text-anchor": "middle", class: "tick" }));
    txt(ga, (m.l + w - m.r) / 2, h - 4, "neuron score minus threshold", { "text-anchor": "middle", class: "tick" });
    el("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: "var(--axis)" }, ga);
    G.forEach(([k, c]) => {
      let d = `M${X(e[0])},${Y(0)}`;
      share[k].forEach((v, i) => { d += `L${X(e[i])},${Y(v)}L${X(e[i + 1])},${Y(v)}`; });
      d += `L${X(e[nb])},${Y(0)}`;
      el("path", { d: d + "Z", style: `fill:${c}`, opacity: 0.12 }, svg);
      el("path", { d, fill: "none", style: `stroke:${c}`, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    });
    el("rect", { x: X(0), y: m.t, width: Math.max(0, w - m.r - X(0)), height: h - m.b - m.t, style: "fill:var(--gate-wash)", opacity: 0.6 }, svg);
    el("line", { x1: X(0), x2: X(0), y1: m.t - 4, y2: h - m.b, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
    txt(svg, X(0) + 6, m.t + 8, "threshold θ → fires", { class: "annot", style: "font-weight:650" });
    const hit = el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.b - m.t, fill: "transparent" }, svg);
    const mv = (ev) => {
      const pt = ev.touches ? ev.touches[0] : ev, r = svg.getBoundingClientRect();
      const xv = X.inv(((pt.clientX - r.left) / r.width) * w);
      const i = Math.max(0, Math.min(nb - 1, Math.floor((xv - e[0]) / (e[1] - e[0]))));
      showTip(`<div class="t">score ${e[i].toFixed(1)} to ${e[i + 1].toFixed(1)}${e[i] >= 0 ? " (fires)" : ""}</div>` + G.slice().reverse().map(([k, c]) => `<div class="r"><span>${swatch(c)}${names[k]}</span><b>${share[k][i].toFixed(1)}%</b></div>`).join(""), pt.clientX, pt.clientY);
    };
    hit.addEventListener("mousemove", mv); hit.addEventListener("touchstart", mv, { passive: true }); hit.addEventListener("mouseleave", hideTip); hit.addEventListener("touchend", hideTip);
  }

  // ------------------------------------------------------------------ scoreboard
  let scoreTask = null;
  function renderScoreTabs() {
    const list = tasksList();
    if (!scoreTask) scoreTask = pickDefault(list, (t) => t.status !== "pending");
    seg($("#score-tasks"), list.map((t) => ({ id: t.id, label: shortName(t), pending: t.status === "pending", disabled: t.status === "pending" })), scoreTask, (id) => { scoreTask = id; renderScore(); });
  }
  function barPanel(title, hint, rows, opt) {
    // rows: [{arm,label,v,txt,color,dash,tipHtml}] opt: {max, log:[lo,hi], ticks:[...], tickFmt, dot}
    const pos = (v) => {
      if (!isNum(v)) return null;
      if (opt.log) { const [lo, hi] = opt.log; if (v <= 0) return 0; return Math.max(0, Math.min(100, ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * 100)); }
      return Math.max(0, Math.min(100, (v / opt.max) * 100));
    };
    const body = rows.map((r, i) => {
      const p = pos(r.v);
      let mark = "";
      if (p === null || (opt.log && r.v <= 0)) mark = "";
      else if (opt.dot) mark = `<span class="dot${r.dash ? " hollow" : ""}" style="left:${p}%;background:${r.color}"></span>`;
      else mark = `<span class="bar${r.dash ? " dash" : ""}" style="width:${p}%;background:${r.color}"></span>`;
      return `<div class="brow${r.arm === "gate" ? " hl" : ""}" data-i="${i}"><span class="an" title="${esc(r.label)}">${esc(r.label)}</span><span class="track">${mark}</span><span class="val">${r.v === undefined || r.v === null ? (r.txt ? esc(r.txt) : "–") : esc(r.txt)}</span></div>`;
    }).join("");
    const ticks = (opt.ticks || []).map((v) => `<span style="left:${pos(v)}%" class="${pos(v) === 0 ? "l0" : ""}">${(opt.tickFmt || String)(v)}</span>`).join("");
    return `<div class="panel"><h4>${title}</h4><p class="ph">${hint}</p>${body}<div class="scale"><span></span><span class="ticks">${ticks}</span><span></span></div></div>`;
  }
  function renderScore() {
    const t = D.tasks[scoreTask];
    const node = $("#score-panels");
    if (!t || !t.arms) { node.innerHTML = ""; pendingFig(node, "scoreboard", `results/${scoreTask}/*.json`); return; }
    const arms = t.score_arms || t.arm_order, A = t.arms, meta = t.meta || {};
    const pct = (v) => (isNum(v) ? F.pct0(v) : null);
    const mk = (arm, v, txtv) => ({ arm, label: armLabel(t, arm), v, txt: txtv, color: col(arm), dash: family(arm) === "random" });
    const tipFor = (arm) => {
      const r = A[arm];
      const line = (k, v) => `<div class="r"><span>${k}</span><b>${v}</b></div>`;
      return `<div class="t">${swatch(col(arm), family(arm) === "random")} ${esc(armLabel(t, arm))}</div>` +
        line(meta.fail_short || "failure", isNum(r.fail) ? F.pct1(r.fail) : "–") + line(meta.kept_short || "kept", isNum(r.kept) ? F.pct1(r.kept) : "–") +
        line("unchanged", isNum(r.same) ? F.pct1(r.same) : "\u2013") +
        line("MMLU-200", isNum(r.mmlu) ? F.pct1(r.mmlu) : "\u2013") + line("chat KL", isNum(r.kl) ? F.kl2(r.kl) : "\u2013") +
        line("weights changed", isNum(r.entries) ? F.int(r.entries) : "–") + line("trainable params", isNum(r.trainable) ? F.compact(r.trainable) : "–") +
        line("build time", isNum(r.fit_s) ? Math.round(r.fit_s) + " s" : "–");
    };
    const failRows = arms.map((a) => mk(a, A[a].fail, pct(A[a].fail)));
    const sameRows = arms.map((a) => mk(a, A[a].same, isNum(A[a].same) ? F.pct0(A[a].same) : "n/a"));
    const keptMax = Math.max(100, ...arms.map((a) => A[a].kept || 0));
    const keptRows = arms.map((a) => mk(a, A[a].kept, pct(A[a].kept)));
    const klRows = arms.map((a) => mk(a, A[a].kl, isNum(A[a].kl) ? F.kl2(A[a].kl) : "n/a"));
    const touch = (a) => (isNum(A[a].entries) ? A[a].entries : isNum(A[a].trainable) ? A[a].trainable : family(a) === "base" || family(a) === "prompt" ? 0 : null);
    const touchRows = arms.map((a) => { const v = touch(a); return mk(a, v, v === 0 ? "0" : isNum(v) ? F.compact(v) : "–"); });
    node.innerHTML =
      barPanel(`Failure: ${esc(meta.fail_short || "failure rate")}`, `${esc(meta.fail || "")} · lower is better`, failRows, { max: 100, ticks: [0, 25, 50, 75, 100], tickFmt: (v) => v + "%" }) +
      barPanel("Unchanged: ordinary outputs identical", `${esc(meta.same_label || "")} \u00b7 higher is better`, sameRows, { max: 100, ticks: [0, 25, 50, 75, 100], tickFmt: (v) => v + "%" }) +
      barPanel(`Kept: ${esc(meta.kept_short || "kept")}`, `${esc(meta.kept || "")} · higher is better`, keptRows, { max: keptMax, ticks: [0, 25, 50, 75, 100], tickFmt: (v) => v + "%" }) +
      barPanel("Drift on ordinary inputs", `chat KL from the base model, log scale · lower is better${t.base_task.startsWith("whisper") ? " (decoder KL on clean speech)" : ""}`, klRows, { log: [1e-4, 1], dot: true, ticks: [1e-4, 1e-3, 1e-2, 1e-1, 1], tickFmt: (v) => (v >= 1 ? "1" : String(v)) }) +
      barPanel("Weights touched", "entries changed (gate) or trainable parameters (LoRA), log scale", touchRows, { log: [1, 1e9], ticks: [1, 1e3, 1e6, 1e9], tickFmt: (v) => F.compact(v) });
    $$(".brow", node).forEach((r) => {
      const arm = arms[+r.dataset.i];
      r.addEventListener("mousemove", (ev) => showTip(tipFor(arm), ev.clientX, ev.clientY));
      r.addEventListener("mouseleave", hideTip);
      r.addEventListener("click", (ev) => showTip(tipFor(arm), ev.clientX, ev.clientY));
    });
    const g = A.gate, b = A.base, rnd = A.random;
    const lo = A.lora;
    $("#score-takeaway").textContent = g && b && isNum(g.fail) && isNum(b.fail)
      ? `One neuron: ${F.pct0(b.fail)} \u2192 ${F.pct0(g.fail)} ${meta.fail_short || "failure"}` + (isNum(g.same) ? `, ${F.pct0(g.same)} of ordinary outputs unchanged` : "") +
        (lo && isNum(lo.same) ? ` (LoRA: ${F.pct0(lo.same)}).` : ".")
      : "What did each fix break?";
    $("#score-sub").textContent = `${meta.title || t.id} · ${meta.model_label || t.model || ""} · ${t.n ? t.n.toLocaleString() + " held-out items" : ""}${t.status === "partial" ? " · some arms still running" : ""}`;
  }
  function renderScoreTable() {
    const list = tasksList();
    const cols = [["base", "Base"], ["prompt", "Prompt / decoding fix"], ["gate", "One neuron"], ["random", "Random neuron"], ["lora", "LoRA, same data"]];
    $("#score-table thead").innerHTML = `<tr><th>Failure (model)</th>${cols.map(([a, l]) => `<th class="n"><span class="mhead">${swatch(col(a), a === "random")}${l}</span></th>`).join("")}</tr>`;
    $("#score-table tbody").innerHTML = list.map((t) => {
      const A = t.arms || {};
      const ks = (t.meta && t.meta.kept_short) || "kept";
      const cell = (a) => {
        if (a === "prompt" && !A.prompt) {
          const xs = ["thresholds", "thresholds_tuned"].filter((x) => A[x] && isNum(A[x].fail));
          if (!xs.length) return `<td class="n">${PENDING}</td>`;
          return `<td class="n">${xs.map((x) => `<span class="cell2">${F.pct0(A[x].fail)} <span class="dim">${esc(A[x].label.toLowerCase())}</span></span>`).join("")}` +
            xs.filter((x) => isNum(A[x].kept) && A[x].kept < 99.95).map((x) => `<span class="note">${esc(A[x].label.toLowerCase())}: ${ks} ${F.keep(A[x].kept)}</span>`).join("") + "</td>";
        }
        const r = A[a];
        if (!r || !isNum(r.fail)) return `<td class="n">${PENDING}</td>`;
        let note = "";
        if (a === "prompt" && A.base && isNum(A.base.kept) && isNum(r.kept) && A.base.kept - r.kept >= 15) note = `<span class="note"><b>but ${ks} ${F.pct0(A.base.kept)} \u2192 ${F.pct0(r.kept)}</b></span>`;
        else if ((a === "gate" || a === "lora") && isNum(r.kept)) note = `<span class="note">${ks} ${F.keep(r.kept)}</span>`;
        return `<td class="n${a === "gate" ? " hl" : ""}">${F.pct0(r.fail)}${note}</td>`;
      };
      return `<tr><td>${esc((t.meta && t.meta.title) || t.id)} <span class="dim">(${esc((t.meta && t.meta.model_label) || "")})</span></td>${cols.map(([a]) => cell(a)).join("")}</tr>`;
    }).join("");
    $$("#score-table tbody tr").forEach((tr) => Array.from(tr.children).forEach((td, i) => { if (i > 0) td.dataset.l = cols[i - 1][1]; }));
  }

  // ------------------------------------------------------------------ examples
  const VERD = {
    entity: { fail: ["declines", "invents a bio"], kept: ["answers", "says it doesn’t know"] },
    agent_injection: { fail: ["ignores the injection", "runs the attacker’s command"], kept: ["makes the tool call", "skips the tool call"] },
    signoff: { fail: ["just stops", "adds a sign-off"], kept: ["keeps the question/offer", "drops it"] },
    whisper_silence: { fail: ["stays silent", "transcribes words"], kept: ["transcript unchanged", "transcript changed"] },
  };
  const KIND = { fixed: "Fixed by the neuron", kept_break: "The cheap fix breaks a real answer", miss: "The neuron misses", lora_vs_gate: "LoRA and the neuron disagree",
    never_fires: "The neuron never fires on this phrasing", imperfect: "Imperfect: a different tell appears", lora_breaks: "LoRA rewrites an output the neuron leaves alone" };
  let exTask = null;
  function renderExamples() {
    const list = tasksList().filter((t) => (t.examples || []).length);
    const grid = $("#ex-grid");
    const all = tasksList().filter((t) => !t.variant);
    if (!list.length) { $("#ex-tasks").innerHTML = ""; return pendingFig(grid, "example replies", "results/<task>/outputs/<arm>.jsonl"); }
    if (!exTask || !list.find((t) => t.id === exTask)) exTask = pickDefault(list, () => true);
    seg($("#ex-tasks"), all.concat(list.filter((t) => t.variant)).map((t) => ({ id: t.id, label: shortName(t), pending: !(t.examples || []).length, disabled: !(t.examples || []).length })), exTask, (id) => { exTask = id; renderExamples(); });
    const t = D.tasks[exTask];
    const V = VERD[t.base_task] || { fail: ["ok", "fails"], kept: ["ok", "breaks"] };
    grid.innerHTML = "";
    t.examples.forEach((e) => {
      const grp = e.kind === "kept_break" || e.kind === "lora_breaks" ? "kept" : "fail";
      const arms = Object.keys(e.replies);
      let cur = arms.includes("gate") ? "gate" : arms[0];
      const card = document.createElement("div");
      card.className = "ex";
      const isWhisper = t.base_task.startsWith("whisper");
      const head = isWhisper ? esc(e.label) : t.base_task === "agent_injection" ? (e.label === "injected page" ? `Hidden in the page: “${esc(e.prompt.replace(/^[-\s]+|[-\s]+$/g, ""))}”` : esc(e.prompt)) : esc(e.prompt);
      card.innerHTML = `<span class="kind ${e.kind}">${KIND[e.kind] || e.kind}</span><div class="q">${head}</div><div class="lab">${isWhisper ? esc(e.id) : esc(e.label)}</div>${e.audio ? `<audio controls preload="none" src="${esc(e.audio)}"></audio>` : ""}<div class="seg tabs" role="tablist"></div><div class="reply" role="tabpanel"></div><div class="verdict-line"></div>`;
      const tabs = $(".tabs", card), rep = $(".reply", card), vl = $(".verdict-line", card);
      const show = () => {
        tabs.innerHTML = "";
        arms.forEach((a) => {
          const b = document.createElement("button");
          b.type = "button"; b.setAttribute("role", "tab"); b.setAttribute("aria-pressed", String(a === cur)); b.setAttribute("aria-selected", String(a === cur));
          const v = e.verdict[a];
          b.innerHTML = `${esc(armLabel(t, a))}${v ? ` <span class="v ${v}">${v === "good" ? "✓" : "✗"}</span>` : ""}`;
          b.addEventListener("click", () => { cur = a; show(); });
          tabs.appendChild(b);
        });
        let text = e.replies[cur] || "";
        if (isWhisper && !/[A-Za-z0-9]/.test(text)) text = (text.trim() ? `“${text.trim()}” ` : "") + "(no words: counts as silent)";
        rep.textContent = text;
        rep.style.borderLeftColor = col(cur);
        let v = e.verdict[cur];
        let line = v ? (v === "good" ? "\u2713 " + V[grp][0] : "\u2717 " + V[grp][1]) : "";
        if (e.kind === "lora_breaks" && cur !== "base") {
          const same = (e.replies[cur] || "").trim() === (e.replies.base || "").trim();
          v = same ? "good" : "bad"; line = same ? "\u2713 identical to the base output" : "\u2717 different from the base output";
        }
        vl.className = "verdict-line " + (v || "");
        vl.textContent = line;
      };
      show();
      grid.appendChild(card);
    });
  }

  // ------------------------------------------------------------------ sweep
  let swTask = null, swMetric = "fail";
  function sweepCalc(t) {
    const S = t.sweep, g = S.gate || [], l = S.lora || [];
    const at = (rows, n) => rows.find((r) => r.n === n);
    const ns = g.map((r) => r.n).filter((n) => at(l, n));
    const match = ns.find((n) => isNum(at(l, n).fail) && isNum(at(g, n).fail) && at(l, n).fail <= at(g, n).fail + 2);
    const lastG = g[g.length - 1] || {}, lastL = l[l.length - 1] || {};
    const ratios = ns.map((n) => (at(g, n).kl > 0 && isNum(at(l, n).kl) ? at(l, n).kl / at(g, n).kl : null)).filter(isNum).sort((a, b) => a - b);
    const enough = g.find((r) => isNum(r.fail) && isNum(lastG.fail) && r.fail <= lastG.fail + 5);
    const at8 = (rows) => (rows.find((r) => r.n === 8) || {}).fail;
    return { gate_n8: at8(g), lora_n8: at8(l), task_short: `${((t.meta && t.meta.short) || t.id).toLowerCase()} (${(t.meta && t.meta.model_label) || ""})`, match_n: match, all_gate: lastG.fail, all_lora: lastL.fail, kl_ratio: ratios.length ? ratios[Math.floor(ratios.length / 2)] : undefined, gate_enough_n: enough && enough.n, never: !match && ns.length };
  }
  function renderSweep() {
    const node = $("#sweep-chart");
    const list = tasksList().filter((t) => t.sweep && (t.sweep.gate || t.sweep.lora));
    $("#sweep-metric").style.display = list.length ? "" : "none";
    if (!list.length) {
      $("#sweep-legend").innerHTML = ""; $("#sweep-tasks").innerHTML = "";
      return pendingFig(node, "data-scaling sweep", "results/<task>/sweep/*.json");
    }
    if (!swTask || !list.find((t) => t.id === swTask)) swTask = pickDefault(list, () => true);
    seg($("#sweep-tasks"), list.map((t) => ({ id: t.id, label: shortName(t) })), swTask, (id) => { swTask = id; renderSweep(); });
    const t = D.tasks[swTask], S = t.sweep, A = t.arms || {}, meta = t.meta || {};
    const m = swMetric;
    const series = [];
    const big = (t.bigdata || []).filter((r) => isNum(r.n));
    [["lora", "LoRA"], ["gate", "One neuron"]].forEach(([k, lab]) => {
      const rows = S[k]; if (!rows) return;
      series.push({ id: k, label: lab, color: col(k), points: rows.map((r) => [r.n, r[m]]), band: rows.map((r) => [r.n, r[m + "_min"], r[m + "_max"]]), markers: true, width: 2.25, priority: k === "gate" ? 1 : 2, endLabel: !big.some((r) => r.method === k && isNum(r[m])) });
      [false, true].forEach((varied) => {
        const ext = big.filter((r) => r.method === k && !!r.varied === varied && isNum(r[m])).sort((a, b) => a.n - b.n);
        if (!ext.length) return;
        const last = rows[rows.length - 1];
        series.push({ id: k + "_big" + (varied ? "v" : ""), label: `${lab}, more data${varied ? " (varied phrasing)" : ""}`, short: "", endLabel: false, color: col(k), dash: true, hollow: true, markers: true, width: 1.75,
          points: [[last.n, last[m]]].concat(ext.map((r) => [r.n, r[m]])), priority: 4 });
      });
    });
    const allN = Array.from(new Set([].concat(...Object.values(S).map((rows) => rows.map((r) => r.n)), big.map((r) => r.n)))).sort((a, b) => a - b);
    const refs = [];
    if (m !== "kl") {
      if (A.base && isNum(A.base[m])) refs.push({ y: A.base[m], label: "base", color: "var(--base)", dash: true });
      const pa = A.prompt ? "prompt" : ["thresholds_tuned", "thresholds", "vad"].find((x) => A[x]);
      if (pa && isNum(A[pa][m])) refs.push({ y: A[pa][m], label: armLabel(t, pa).toLowerCase(), color: "var(--prompt)", dash: true });
    }
    const vals = [].concat(...series.map((s) => (s.band || s.points.map((p) => [p[0], p[1], p[1]])).map((b) => [b[1], b[2]]).flat())).filter(isNum);
    let y;
    if (m === "kl") {
      const pos = vals.filter((v) => v > 0);
      const lo = pos.length ? Math.pow(10, Math.floor(Math.log10(Math.min(...pos)))) : 1e-4, hi = pos.length ? Math.pow(10, Math.ceil(Math.log10(Math.max(...pos)))) : 1;
      const ticks = []; for (let v = lo; v <= hi * 1.001; v *= 10) ticks.push(+v.toPrecision(1));
      y = { min: lo, max: hi, log: true, ticks, fmt: (v) => String(v) };
    } else {
      const top = Math.max(100, ...vals, ...refs.map((r) => r.y));
      y = { min: 0, max: top > 100 ? Math.ceil(top / 10) * 10 : 100, fmt: (v) => v + "%" };
      if (!vals.length) { pendingFig(node, `${lab} for this sweep`, `results/${swTask}/sweep/*.json (guards.chat_same)`); return; }
    }
    legend($("#sweep-legend"), [{ label: "One neuron (mean, min\u2013max band)", color: col("gate"), line: true }, { label: "LoRA (mean, min\u2013max band)", color: col("lora"), line: true }]
      .concat(big.length ? [{ label: "more labelled data (one run)", color: "var(--ink-2)", line: true, dash: true }] : [])
      .concat(refs.map((r) => ({ label: r.label, color: r.color, line: true, dash: true }))));
    lineChart(node, {
      aria: "Failure rate vs number of practice examples for the gate and LoRA",
      series, refs, snapXs: allN,
      x: { min: allN[0], max: allN[allN.length - 1], log: true, ticks: allN, fmt: (v) => String(v), label: "practice examples (log scale)" },
      y, tipTitle: (n) => `n = ${n} practice examples`, tipFmt: m === "kl" ? (v) => F.kl2(v) : (v) => F.pct1(v),
    });
    const lab = { fail: meta.fail_short || "failure", kept: meta.kept_short || "kept", kl: "chat KL", same: "ordinary outputs unchanged" }[m];
    $("#fig-sweep .sub").textContent = `${meta.title || t.id} (${meta.model_label || ""}): ${lab} vs labelled practice examples. Gate and LoRA refit on stratified subsets; lines are the mean over seeds, bands span min\u2013max.` +
      (t.sweep_source ? " Sweep scored on a 100-prompt subset of the held-out set." : "");
  }
  function sweepText() {
    const list = tasksList().filter((t) => t.sweep && t.sweep.gate && t.sweep.lora);
    if (!list.length) return;
    const t = D.tasks[pickDefault(list, () => true)];
    const c = sweepCalc(t);
    CALC.sw = c;
    if (!c.match_n && c.never) {
      $("#sweep-text").innerHTML = `On ${esc(c.task_short)}: across the sizes we tried, LoRA never matches the gate; with all the data it reaches <b>${F.pct0(c.all_lora)}</b> vs the gate&rsquo;s <b>${F.pct0(c.all_gate)}</b>. ` +
        (isNum(c.kl_ratio) ? `The gate&rsquo;s side effects stay ${F.x0(c.kl_ratio)} lower across the sweep` : "") + (c.gate_enough_n ? `, and it gets within 5 points of its full-data result from n &asymp; ${c.gate_enough_n}.` : ".");
    }
  }

  // ------------------------------------------------------------------ persistence
  let peTask = null;
  const PE = { "base+chat": ["Unpatched + chat SFT", "var(--base)", false], "gate+chat": ["One neuron + chat SFT", "var(--gate)", false], "gate+conflict": ["One neuron + conflicting SFT", "var(--gate)", true], "lora+chat": ["LoRA fix + chat SFT", "var(--lora)", false] };
  function peCalc(t) {
    const C = t.persistence.conditions, gc = C["gate+chat"], gx = C["gate+conflict"], bc = C["base+chat"];
    const out = {};
    if (gc && gc.length) {
      out.chat_start = gc[0].fail; out.chat_end = gc[gc.length - 1].fail; out.steps = gc[gc.length - 1].step;
      const atS = (rows, st) => ((rows || []).find((r) => r.step === st) || {}).fail;
      out.chat_50 = atS(gc, 50); out.conflict_50 = atS(gx, 50);
      const lc = C["lora+chat"];
      if (lc && lc.length) { out.lora_start = lc[0].fail; out.lora_50 = atS(lc, 50); out.lora_100 = atS(lc, 100); }
      const fr = gc.filter((r) => isNum(r.fire));
      out.fire_10 = isNum((fr.find((r) => r.step === 10) || {}).fire) ? 100 * fr.find((r) => r.step === 10).fire : undefined;
      const gone = fr.find((r) => r.step > 0 && r.fire === 0);
      out.fire_gone = gone ? gone.step : undefined;
      out.cos = t.persistence.gate_cos_min;
      const base0 = bc && bc.length ? bc[0].fail : (t.arms && t.arms.base && t.arms.base.fail);
      const d = out.chat_end - out.chat_start;
      out.chat_verdict = d < 10 ? "keeps" : isNum(base0) && out.chat_end >= base0 - 10 ? "erases" : "partly erodes";
      if (gx && gx.length && isNum(base0)) {
        const mid = (gx[0].fail + base0) / 2;
        const hit = gx.find((r) => isNum(r.fail) && r.fail >= mid);
        out.conflict_steps = hit ? hit.step : undefined;
        out.conflict_never = !hit;
        out.conflict_end = gx[gx.length - 1].fail;
      }
    }
    return out;
  }
  function renderPersist() {
    const node = $("#persist-chart");
    const list = tasksList().filter((t) => t.persistence);
    if (!list.length) { $("#persist-legend").innerHTML = ""; $("#fig-persist .inset").style.display = "none"; return pendingFig(node, "persistence curves", "results/<task>/persistence.json"); }
    if (!peTask || !list.find((t) => t.id === peTask)) peTask = pickDefault(list, () => true);
    const t = D.tasks[peTask], C = t.persistence.conditions, meta = t.meta || {};
    const order = ["gate+chat", "gate+conflict", "lora+chat", "base+chat"];
    const series = Object.keys(C).sort((a, b) => (order.indexOf(a) + 9) % 9 - (order.indexOf(b) + 9) % 9).map((k) => {
      const p = PE[k] || [k, "var(--random)", false];
      return { id: k, label: p[0], short: p[0].replace(" SFT", ""), endLabel: false, color: p[1], dash: p[2], points: C[k].map((r) => [r.step, r.fail]), markers: true, priority: k === "gate+chat" ? 1 : k === "lora+chat" ? 2 : 3 };
    });
    legend($("#persist-legend"), series.map((s) => ({ label: s.label, color: s.color, line: true, dash: s.dash })));
    const xs = Array.from(new Set([].concat(...series.map((s) => s.points.map((p) => p[0]))))).sort((a, b) => a - b);
    lineChart(node, {
      aria: "Failure rate as fine-tuning continues after the patch", series, snapXs: xs,
      x: { min: xs[0], max: xs[xs.length - 1], label: "fine-tuning steps after patching" },
      y: { min: 0, max: 100, fmt: (v) => v + "%" }, tipTitle: (s) => `step ${s}`, tipFmt: (v) => F.pct1(v),
    });
    $("#fig-persist .sub").textContent = `${meta.title || t.id} (${meta.model_label || ""}): ${meta.fail_short || "failure"} on held-out prompts as fine-tuning of the last ${t.persistence.k_layers || "few"} layers continues.`;
    const c = peCalc(t);
    CALC.pe = c;
    // inset: does the neuron still fire?
    const fireRows = (C["gate+chat"] || []).filter((r) => isNum(r.fire));
    const fnode = $("#fire-chart");
    if (fireRows.length > 1) {
      $("#fig-persist .inset").style.display = "";
      lineChart(fnode, { aria: "Share of made-up names on which the gate neuron fires, by fine-tuning step", height: 190, compact: true,
        series: [{ id: "fire", label: "fires on made-up names", color: "var(--gate)", points: fireRows.map((r) => [r.step, 100 * r.fire]), markers: true, endLabel: false }],
        x: { min: fireRows[0].step, max: fireRows[fireRows.length - 1].step, ticks: fireRows.map((r) => r.step), label: "steps" },
        y: { min: 0, max: 100, ticks: [0, 50, 100], fmt: (v) => v + "%" }, snapXs: fireRows.map((r) => r.step), tipTitle: (s) => `step ${s}`, tipFmt: (v) => F.pct0(v) });
      const cm = t.persistence.gate_cos_min;
      $("#fire-note").textContent = isNum(cm) ? `The neuron's own weights don't move: cosine to the patch ${cm >= 0.99995 ? "1.0000" : "\u2265 " + cm.toFixed(4)} at every step. It stops firing because the layers feeding it changed.` : "";
    } else $("#fig-persist .inset").style.display = "none";
    const tk = isNum(c.chat_50) && c.chat_50 > 80 ? `Fifty steps of unrelated fine-tuning erase the patch: ${F.pct0(c.chat_start)} \u2192 ${F.pct0(c.chat_50)}.`
      : c.conflict_never ? "The patch survives even training that rewards the failure, within this budget."
      : c.chat_verdict === "keeps" ? `Ordinary fine-tuning leaves the patch alone; training that rewards the failure erases it${isNum(c.conflict_steps) ? ` within ${c.conflict_steps} steps` : ""}.`
      : c.chat_verdict ? `Even ordinary fine-tuning ${c.chat_verdict} the patch.` : "Does more training erase the patch?";
    $("#persist-takeaway").textContent = tk;
  }

  // ------------------------------------------------------------------ pareto (entity)
  let parX = "kl";
  function renderPareto() {
    const node = $("#pareto-chart");
    const t = Object.values(D.tasks).find((x) => x && x.pareto && x.group === "main");
    if (!t) { $("#pareto-legend").innerHTML = ""; $("#pareto-x").style.display = "none"; return pendingFig(node, "fix vs side-effect points", "results/entity/pareto/*.json"); }
    const P = t.pareto, A = t.arms || {};
    const xv = (r) => (parX === "kl" ? r.kl : isNum(r.same) ? 100 - r.same : null);
    const groups = [["gate", "One neuron, write gain", "gain "], ["lora", "LoRA, training steps", "step "]].filter(([k]) => P[k]);
    const pts = [].concat(...groups.map(([k]) => P[k].map((r) => ({ k, r, x: xv(r), y: r.fail })))).filter((p) => isNum(p.x) && isNum(p.y) && (parX !== "kl" || p.x > 0));
    // the shipped arms as extra points
    ["gate", "lora"].forEach((k) => { const r = A[k]; if (r && isNum(r.fail)) { const x = parX === "kl" ? r.kl : isNum(r.same) ? 100 - r.same : null; if (isNum(x) && (parX !== "kl" || x > 0)) pts.push({ k, r: { value: "shipped", fail: r.fail, kept: r.kept, kl: r.kl, same: r.same }, x, y: r.fail, shipped: true }); } });
    legend($("#pareto-legend"), groups.map(([k, l]) => ({ label: l, color: col(k) })).concat([{ label: "base", color: "var(--base)", line: true, dash: true }, { label: armLabel(t, "prompt").toLowerCase(), color: "var(--prompt)", line: true, dash: true }]));
    const w = W(node), narrow = w < 560, h = narrow ? 280 : 340, m = { l: 46, r: 16, t: 14, b: 42 };
    const svg = svgRoot(node, w, h, "Fabrication vs side effect for gate strengths and LoRA checkpoints");
    let X;
    let xt;
    if (parX === "kl") {
      const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...pts.map((p) => p.x))))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...pts.map((p) => p.x)))));
      X = scale(lo, hi, m.l, w - m.r, true); xt = []; for (let v = lo; v <= hi * 1.001; v *= 10) xt.push(+v.toPrecision(1));
    } else { X = scale(0, 100, m.l, w - m.r); xt = [0, 25, 50, 75, 100]; }
    const ymax = Math.max(100, ...pts.map((p) => p.y));
    const Y = scale(0, ymax, h - m.b, m.t);
    const g = el("g", { class: "ax" }, svg);
    [0, 25, 50, 75, 100].forEach((v) => { el("line", { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: "gridl" }, g); txt(g, m.l - 6, Y(v) + 4, v + "%", { "text-anchor": "end", class: "tick" }); });
    xt.forEach((v) => txt(g, X(v), h - m.b + 16, parX === "kl" ? String(v) : v + "%", { "text-anchor": "middle", class: "tick" }));
    el("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, stroke: "var(--axis)" }, g);
    txt(g, (m.l + w - m.r) / 2, h - 4, parX === "kl" ? "chat KL from the base model (log scale) → more side effect" : "ordinary chat answers changed → more side effect", { "text-anchor": "middle", class: "tick" });
    [["base", "var(--base)"], ["prompt", "var(--prompt)"]].forEach(([a, c]) => { const r = A[a]; if (r && isNum(r.fail)) { el("line", { x1: m.l, x2: w - m.r, y1: Y(r.fail), y2: Y(r.fail), style: `stroke:${c}`, "stroke-dasharray": "4 3", "stroke-width": 1.25 }, svg); txt(svg, w - m.r - 4, Y(r.fail) - 5, `${a === "base" ? "base" : armLabel(t, a).toLowerCase()} (no weights changed)`, { "text-anchor": "end", class: "annot" }); } });
    groups.forEach(([k, , pre]) => {
      const ps = pts.filter((p) => p.k === k && !p.shipped);
      if (ps.length > 1) el("path", { d: "M" + ps.map((p) => `${X(p.x)},${Y(p.y)}`).join("L"), fill: "none", style: `stroke:${col(k)}`, "stroke-width": 1.5, opacity: 0.6 }, svg);
      ps.forEach((p, i) => {
        el("circle", { cx: X(p.x), cy: Y(p.y), r: 5, style: `fill:${col(k)};stroke:var(--surface)`, "stroke-width": 2 }, svg);
        if (k === "lora" && !narrow) txt(svg, X(p.x) + 8, Y(p.y) + 4, String(p.r.value), { class: "annot" });
      });
      if (ps.length) {
        const ext = k === "gate" ? ps.reduce((a, b) => (a.y > b.y ? a : b)) : ps.reduce((a, b) => (a.y < b.y ? a : b));
        txt(svg, X(ext.x) + (k === "gate" ? 8 : -8), Y(ext.y) + (k === "gate" ? -12 : 20), k === "gate" ? `gate, gain ${ps[0].r.value}–${ps[ps.length - 1].r.value}` : narrow ? "LoRA" : "LoRA (labels = steps)", { class: "lbl-direct", "text-anchor": k === "gate" ? "start" : "end" });
      }
    });
    const hit = el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.b - m.t, fill: "transparent" }, svg);
    const mv = (ev) => {
      const pt = ev.touches ? ev.touches[0] : ev, rr = svg.getBoundingClientRect();
      const px = ((pt.clientX - rr.left) / rr.width) * w, py = ((pt.clientY - rr.top) / rr.height) * h;
      let best = null, bd = 1e9;
      pts.forEach((p) => { const d = Math.hypot(X(p.x) - px, Y(p.y) - py); if (d < bd) { bd = d; best = p; } });
      if (!best || bd > 40) return hideTip();
      const r = best.r, L = (k, v) => `<div class="r"><span>${k}</span><b>${v}</b></div>`;
      showTip(`<div class="t">${swatch(col(best.k))} ${best.k === "gate" ? "One neuron" : "LoRA"}${best.shipped ? " (the arm in the scoreboard)" : best.k === "gate" ? `, gain ${r.value}` : `, step ${r.value}`}</div>` +
        L("fabricates", F.pct1(r.fail)) + L("facts kept", isNum(r.kept) ? F.pct1(r.kept) : "–") + L("chat KL", isNum(r.kl) ? F.kl2(r.kl) : "–") + L("answers unchanged", isNum(r.same) ? F.pct1(r.same) : "–"), pt.clientX, pt.clientY);
    };
    hit.addEventListener("mousemove", mv); hit.addEventListener("touchstart", mv, { passive: true }); hit.addEventListener("mouseleave", hideTip); hit.addEventListener("touchend", hideTip);
  }

  // ------------------------------------------------------------------ section 6 tables
  function renderTransfer() {
    const ts = Object.values(D.tasks).filter((t) => t && t.group === "transfer" && t.arms);
    const tb = $("#transfer-table");
    if (!ts.length) { tb.innerHTML = `<tbody><tr><td>${PENDING}</td></tr></tbody>`; return; }
    const cols = ["base", "prompt", "gate", "lora"];
    tb.querySelector("thead").innerHTML = `<tr><th>Model</th>${cols.map((a) => `<th class="n"><span class="mhead">${swatch(col(a))}${esc(METHOD[a].label)}</span></th>`).join("")}</tr>`;
    tb.querySelector("tbody").innerHTML = ts.map((t) => `<tr><td>${esc((t.model || t.id).split("/").pop())}</td>${cols.map((a) => { const r = t.arms[a]; return `<td class="n${a === "gate" ? " hl" : ""}">${r && isNum(r.fail) ? F.pct0(r.fail) : PENDING}${r && isNum(r.kept) ? ` <span class="dim">(${F.pct0(r.kept)})</span>` : ""}</td>`; }).join("")}</tr>`).join("");
  }
  function renderLangmix() {
    const t = D.tasks.langmix, tb = $("#langmix-table");
    if (!t || !t.arms) { tb.innerHTML = `<tbody><tr><td>${PENDING}</td></tr></tbody>`; return; }
    const order = ["base", "prompt", "gate", "gate_gauto", "gate_recipe", "random", "lora"].filter((a) => t.arms[a]);
    const what = { gate: "one neuron, framework recipe", gate_gauto: "one neuron, steeper threshold", gate_recipe: "the blind agents' edit: penalty on Chinese-token output rows", random: "same neuron, random", lora: "fine-tune, same data", prompt: "“answer only in the question's language”", base: "" };
    tb.querySelector("thead").innerHTML = `<tr><th>Fix</th><th class="n">Leak</th><th class="n">Unchanged</th><th class="n">Chat KL</th></tr>`;
    tb.querySelector("tbody").innerHTML = order.map((a) => { const r = t.arms[a]; return `<tr><td><span class="mhead">${swatch(col(a), family(a) === "random")}<b>${esc(r.label)}</b></span>${what[a] ? `<span class="note">${esc(what[a])}</span>` : ""}</td><td class="n${a === "gate_recipe" ? " hl" : ""}">${isNum(r.fail) ? F.pct1(r.fail) : "–"}</td><td class="n">${isNum(r.same) ? F.pct0(r.same) : "–"}</td><td class="n">${isNum(r.kl) ? F.kl2(r.kl) : "–"}</td></tr>`; }).join("");
  }
  function renderAgentSession() {
    const t = Object.values(D.tasks).find((x) => x && x.agent_session);
    const node = $("#agent-session");
    if (!t) { node.innerHTML = ""; return; }
    const s = t.agent_session, A = t.arms || {};
    node.innerHTML = `<div class="card agent-card"><span class="kind">Blind agent session</span><h4>An LLM found this patch unaided</h4>
      <p class="ph" style="font:.9rem/1.5 var(--sans);color:var(--ink-2);margin:0 0 10px">Given only the weights, tools and practice examples, an agent session wrote its own one-neuron patch for ${esc(t.meta.title.toLowerCase())}${s.model ? ` (${esc(s.model)})` : ""}.</p>
      <div class="stats">
        <div><b>${isNum(s.gate.fail) ? F.pct0(s.gate.fail) : "–"}</b><span>${esc(t.meta.fail_short)} (base ${isNum(A.base && A.base.fail) ? F.pct0(A.base.fail) : "–"}, our gate ${isNum(A.gate && A.gate.fail) ? F.pct0(A.gate.fail) : "–"})</span></div>
        ${isNum(s.gate.kept) ? `<div><b>${F.pct0(s.gate.kept)}</b><span>${esc(t.meta.kept_short)}</span></div>` : ""}
        ${isNum(s.gate.same) ? `<div><b>${F.pct0(s.gate.same)}</b><span>ordinary answers unchanged</span></div>` : ""}
        ${isNum(s.entries) ? `<div><b>${F.int(s.entries)}</b><span>weights changed</span></div>` : ""}
        ${isNum(s.minutes) ? `<div><b>${Math.round(s.minutes)} min</b><span>session</span></div>` : ""}
        ${s.random && isNum(s.random.fail) ? `<div><b>${F.pct0(s.random.fail)}</b><span>same-size random edit</span></div>` : ""}
      </div></div>`;
  }

  // ------------------------------------------------------------------ boot
  function computeCalc() {
    const T = D.tasks;
    const ents = Object.values(T).filter((t) => t && t.group === "main" && t.arms && t.arms.gate && isNum(t.arms.gate.entries)).map((t) => t.arms.gate.entries);
    if (ents.length) { const r100 = (v) => (Math.round(v / 100) * 100).toLocaleString("en-US"); CALC.entries_range = ents.length > 1 ? `${r100(Math.min(...ents))}\u2013${r100(Math.max(...ents))}` : F.int(ents[0]); }
    const chg = (id) => { const v = T[id] && sameOf(T[id], "lora"); return isNum(v) ? 100 - v : undefined; };
    CALC.agent_lora_changed = chg("agent_injection"); CALC.whisper_lora_changed = chg("whisper_silence");
    const so = T.signoff && T.signoff.arms;
    if (so && so.prompt && so.base && isNum(so.prompt.metrics.mean_chars) && so.base.metrics.mean_chars) CALC.signoff_prompt_shorter = 100 * (1 - so.prompt.metrics.mean_chars / so.base.metrics.mean_chars);
    const e = T.entity;
    if (e && e.bigdata) {
      const mx = (k) => { const rs = e.bigdata.filter((r) => r.method === k && !r.varied); return rs.length ? rs.reduce((a, b) => (a.n > b.n ? a : b)).fail : undefined; };
      CALC.big = { lora_max: mx("lora"), gate_max: mx("gate") };
    }
    if (e && e.pareto && e.pareto.lora) {
      const L = e.pareto.lora.filter((r) => isNum(r.fail));
      const gf = e.arms && e.arms.gate ? e.arms.gate.fail : null;
      const best = L.filter((r) => r.value >= 50).reduce((a, b) => (a && Math.abs(a.fail - gf) <= Math.abs(b.fail - gf) ? a : b), null);
      CALC.par = { lora_kl_min: Math.min(...L.map((r) => r.kl).filter(isNum)) };
      if (best) Object.assign(CALC.par, { best_step: best.value, best_fail: best.fail, best_kept: best.kept, best_changed: isNum(best.same) ? 100 - best.same : undefined });
    }
    const sl = tasksList().filter((t) => t.sweep && t.sweep.gate && t.sweep.lora);
    if (sl.length) CALC.sw = sweepCalc(D.tasks[pickDefault(sl, () => true)]);
    const pl = tasksList().filter((t) => t.persistence);
    if (pl.length) CALC.pe = peCalc(D.tasks[pickDefault(pl, () => true)]);
  }
  function safe(fn, name) { try { fn(); } catch (e) { console.warn("[blog] " + name + " failed:", e); } }

  async function main() {
    try {
      const q = new URLSearchParams(location.search).get("data");
      const src = q && /^data\/[\w.-]+\.json$/.test(q) ? q : "data/blog.json";
      const r = await fetch(src, { cache: "no-cache" });
      D = await r.json();
    } catch (e) {
      console.warn("[blog] data/blog.json not available", e);
      D = { tasks: {}, task_order: [], archive: {}, site: {} };
    }
    D.tasks = D.tasks || {};
    safe(computeCalc, "calc");
    safe(() => bindAll(), "bind");
    safe(renderMech, "mech");
    safe(renderTrackA, "trackA");
    safe(renderGrok, "grok");
    $$("#fig-grok [data-zoom]").forEach((b) => b.addEventListener("click", () => {
      grokZoom = +b.dataset.zoom;
      $$("#fig-grok [data-zoom]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderGrok();
    }));
    onResize($("#grok-chart"), () => safe(renderGrok, "grok"));
    safe(setupNeuron, "neuron");
    safe(renderAnatomy, "anatomy");
    onResize($("#anatomy-chart"), () => safe(renderAnatomy, "anatomy"));
    safe(renderScoreTabs, "scoreTabs");
    safe(renderScore, "score");
    safe(renderScoreTable, "scoreTable");
    safe(renderExamples, "examples");
    safe(renderSweep, "sweep");

    $$("#sweep-metric [data-m]").forEach((b) => b.addEventListener("click", () => {
      swMetric = b.dataset.m;
      $$("#sweep-metric [data-m]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      safe(renderSweep, "sweep");
    }));
    onResize($("#sweep-chart"), () => safe(renderSweep, "sweep"));
    safe(renderPareto, "pareto");
    $$("#pareto-x [data-x]").forEach((b) => b.addEventListener("click", () => {
      parX = b.dataset.x;
      $$("#pareto-x [data-x]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      safe(renderPareto, "pareto");
    }));
    onResize($("#pareto-chart"), () => safe(renderPareto, "pareto"));
    safe(renderTransfer, "transfer");
    safe(renderLangmix, "langmix");
    safe(renderAgentSession, "agentSession");
    safe(renderPersist, "persist");
    onResize($("#persist-chart"), () => safe(renderPersist, "persist"));
    safe(() => bindAll(), "bind2");
    window.addEventListener("scroll", hideTip, { passive: true });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main); else main();
})();
