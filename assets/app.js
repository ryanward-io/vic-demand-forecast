// Victorian demand forecast page: fetches the latest forecast from the API and the backtest from this repo, draws two charts.
// For local work, serve the page on localhost and add ?api=http://localhost:8002 to point it at the dev server.
// Everything from the network is written with textContent or DOM calls, never as HTML.

const PROD_API = "https://ryanward-io--vic-demand-forecast.us-east.modal.run";
const LOCAL_HOST = /^(localhost|127\.0\.0\.1)$/;
const LOCAL_API = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;
const API = (() => {
  const asked = new URLSearchParams(location.search).get("api");
  return asked && LOCAL_HOST.test(location.hostname) && LOCAL_API.test(asked) ? asked : PROD_API;
})();

const SVG = "http://www.w3.org/2000/svg";
const MARGIN = { top: 20, right: 12, bottom: 22, left: 48 };
// Colour follows the model, never its rank or which models are shown. AEMO's forecast is the reference, so it gets a neutral grey.
// Every forecast also has its own line pattern, so no two are told apart by colour alone.
const COLOURS = { aemo: "--reference", ensemble: "--series-1", lightgbm: "--series-2", chronos2: "--series-3", timestral: "--series-7",
  lstm: "--series-4", mstl: "--series-5", mean_4w: "--series-6", naive_day: "--series-6", naive_week: "--series-6" };
const DASHES = { aemo: null, ensemble: null, lightgbm: "7 3", chronos2: "2 3", timestral: "10 3 2 3", lstm: "14 4", mstl: "4 2 1 2 1 2",
  mean_4w: "5 4", naive_day: "1.5 3", naive_week: "9 3 1.5 3" };
const WIDTHS = { ensemble: 3, aemo: 2.5 };
const SHOWN_BY_DEFAULT = ["aemo", "ensemble", "lightgbm", "chronos2", "timestral"];
const BEST = "ensemble";  // the best of my models in the backtest, used for the headline numbers
const RANGE_MODELS = { chronos2: "Chronos-2", timestral: "Timestral" };
const STALE_HOURS = 30;
const LATE_HOURS = 3;
const MIN_LIVE_DAYS = 7;
const STORE = "vdf:chart:v1";

const $ = (id) => document.getElementById(id);
const parse = (s) => new Date(s + ":00Z");  // NEM time, kept as "UTC" so the browser's own zone never shifts it
const fmtTime = (d) => d.toLocaleString("en-AU", { timeZone: "UTC", weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
const fmtClock = (d) => d.toLocaleTimeString("en-AU", { timeZone: "UTC", hour: "numeric", minute: "2-digit" });
const fmtWeekday = (d) => d.toLocaleDateString("en-AU", { timeZone: "UTC", weekday: "short" });
const fmtDay = (d) => d.toLocaleDateString("en-AU", { timeZone: "UTC", weekday: "short", day: "numeric" });
const fmtDate = (s) => new Date(s + "T00:00Z").toLocaleDateString("en-AU", { timeZone: "UTC", day: "numeric", month: "short", year: "numeric" });
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const fmtMW = (v) => isNum(v) ? `${Math.round(v).toLocaleString("en-AU")} MW` : "–";
const round5 = (v) => Math.round(v / 5) * 5;

let backtest = null;  // set once results/backtest.json arrives; the chart's range labels use it

// ---- small DOM helpers ----
function h(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function lineKey(s) {
  const svg = el("svg", { class: "key", viewBox: "0 0 22 6", "aria-hidden": "true" });
  el("line", { x1: 0, y1: 3, x2: 22, y2: 3, stroke: `var(${s.colorVar})`, "stroke-width": s.width || 2, "stroke-dasharray": s.dash }, svg);
  return svg;
}

const load = () => { try { return JSON.parse(localStorage.getItem(STORE)) || null; } catch { return null; } };
const save = (v) => { try { v ? localStorage.setItem(STORE, JSON.stringify(v)) : localStorage.removeItem(STORE); } catch { /* private mode */ } };

async function fetchJSON(url, waits = [0]) {
  let last;
  for (const wait of waits) {
    if (wait) await new Promise((r) => setTimeout(r, wait));
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) { last = e; }
  }
  throw last;
}

function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(v);
  return ticks;
}

// ---- charts ----
// series: [{name, label, colorVar, points: [[Date, value]], dash, width}]; bands: [{label, colorVar, points: [[Date, low, high]]}]
function lineChart(host, series, { x0, x1, now, unit, fmt, bands = [], label }) {
  host.replaceChildren();
  host.tabIndex = 0;
  host.setAttribute("role", "img");
  host.setAttribute("aria-label", `${label}. Use the left and right arrow keys to read values.`);
  const W = host.clientWidth, H = host.clientHeight;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, host);
  const inView = (p) => p[0] >= x0 && p[0] <= x1;
  const vals = series.flatMap((s) => s.points.filter(inView).map((p) => p[1]))
    .concat(bands.flatMap((b) => b.points.filter(inView).flatMap((p) => [p[1], p[2]]))).filter(isNum);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const ticks = niceTicks(lo, hi);
  const x = (d) => MARGIN.left + ((d - x0) / (x1 - x0)) * (W - MARGIN.left - MARGIN.right);
  const y = (v) => H - MARGIN.bottom - ((v - lo) / (hi - lo)) * (H - MARGIN.top - MARGIN.bottom);

  el("rect", { class: "future", x: x(now), y: MARGIN.top, width: x(x1) - x(now), height: H - MARGIN.top - MARGIN.bottom }, svg);
  el("text", { class: "axis-label", x: MARGIN.left - 6, y: 12, "text-anchor": "end" }, svg).textContent = unit;
  el("text", { class: "axis-label", x: x(now) + 6, y: 12 }, svg).textContent = "Forecast →";
  for (const t of ticks) {
    el("line", { class: "grid", x1: MARGIN.left, x2: W - MARGIN.right, y1: y(t), y2: y(t) }, svg);
    el("text", { x: MARGIN.left - 6, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = t.toLocaleString("en-AU");
  }
  const dayMs = 864e5, every = (W - MARGIN.left) / ((x1 - x0) / dayMs) < 60 ? 2 : 1;
  for (let d = new Date(Math.ceil(x0 / dayMs) * dayMs), i = 0; d <= x1; d = new Date(+d + dayMs), i++) {
    if (i % every) continue;
    el("text", { x: x(d), y: H - 6, "text-anchor": "middle" }, svg).textContent = fmtDay(d);
  }
  el("line", { class: "now", x1: x(now), x2: x(now), y1: MARGIN.top, y2: H - MARGIN.bottom }, svg);

  for (const band of bands) {  // a soft wash between the low and high edges, under the lines
    const pts = band.points.filter(inView);
    const top = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[2]).toFixed(1)}`).join("");
    const bottom = [...pts].reverse().map((p) => `L${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    el("path", { class: "band", d: `${top}${bottom}Z`, fill: `var(${band.colorVar})` }, svg);
  }
  for (const s of series) {
    const d = s.points.filter((p) => isNum(p[1]) && inView(p)).map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    el("path", { class: "line", d, stroke: `var(${s.colorVar})`, "stroke-width": s.width || 2, "stroke-dasharray": s.dash }, svg);
  }

  // Reading values: a crosshair, dots and a tooltip at one half-hour, driven by mouse, touch or the arrow keys.
  const cross = el("line", { class: "cross", y1: MARGIN.top, y2: H - MARGIN.bottom, visibility: "hidden" }, svg);
  const dots = series.map((s) => el("circle", { class: "dot", r: 4, fill: `var(${s.colorVar})`, visibility: "hidden" }, svg));
  const hit = el("rect", { class: "hit", x: MARGIN.left, y: 0, width: W - MARGIN.left - MARGIN.right, height: H, fill: "transparent" }, svg);
  const tip = $("tooltip");
  let current = null;
  const hide = () => {
    current = null; tip.hidden = true; cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  };
  const timeAt = (clientX) => {
    const box = svg.getBoundingClientRect();
    const px = (clientX - box.left) * (W / box.width);
    return new Date(Math.round((+x0 + ((px - MARGIN.left) / (W - MARGIN.left - MARGIN.right)) * (x1 - x0)) / 18e5) * 18e5);
  };
  const showAt = (t) => {
    t = new Date(Math.min(Math.max(+t, +x0), +x1));
    const rows = [];
    series.forEach((s, i) => {
      const p = s.points.find((q) => +q[0] === +t);
      if (!p || !isNum(p[1])) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", x(p[0])); dots[i].setAttribute("cy", y(p[1])); dots[i].setAttribute("visibility", "visible");
      rows.push([s.colorVar, s.label, fmt(p[1]), false]);
    });
    for (const band of bands) {
      const b = band.points.find((q) => +q[0] === +t);
      if (b) rows.push([band.colorVar, band.label, `${fmtMW(b[1])} – ${fmtMW(b[2])}`, true]);
    }
    if (!rows.length) return hide();
    current = t;
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(h("strong", { text: `${fmtTime(t)} AEST` }), ...rows.map(([c, name, v, isBand]) => {
      const key = h("span", { class: isBand ? "key band-key" : "key" });
      key.style.background = `var(${c})`;
      return h("div", { class: "row" }, key, name, h("span", { class: "val", text: v }));
    }));
    tip.hidden = false;
    // Sit beside the point, on whichever side has more room, so the tooltip never covers the lines being read.
    const box = svg.getBoundingClientRect(), panel = tip.parentElement.getBoundingClientRect();
    const cx = box.left - panel.left + x(t) * (box.width / W);
    const left = cx > panel.width / 2 ? cx - tip.offsetWidth - 14 : cx + 14;
    tip.style.left = `${Math.max(8, Math.min(left, panel.width - tip.offsetWidth - 8))}px`;
    tip.style.top = `${box.top - panel.top + 8}px`;
  };
  hit.addEventListener("pointermove", (ev) => showAt(timeAt(ev.clientX)));
  hit.addEventListener("pointerdown", (ev) => showAt(timeAt(ev.clientX)));  // a tap on a phone
  hit.addEventListener("pointerleave", (ev) => { if (ev.pointerType === "mouse") hide(); });  // touch keeps it until a tap elsewhere
  host.addEventListener("keydown", (ev) => {
    const step = { ArrowLeft: -18e5, ArrowRight: 18e5 }[ev.key];
    if (step) { ev.preventDefault(); showAt(new Date(+(current ?? now) + (current ? step : 18e5))); }
    else if (ev.key === "Home") { ev.preventDefault(); showAt(x0); }
    else if (ev.key === "End") { ev.preventDefault(); showAt(x1); }
    else if (ev.key === "Escape") hide();
  });
  host.addEventListener("blur", hide);
  return { hide, owns: (node) => host.contains(node) };
}

// ---- score tables ----
function scoreTable(table, scores, models, { relative = {}, minDays = 0, reference = "aemo", caption }) {
  const label = Object.fromEntries(models.map((m) => [m.name, m.label]));
  const leads = ["today", "tomorrow"];
  const by = {};
  for (const s of scores) (by[s.model] ??= {})[s.lead] = s;
  const total = (m) => isNum(by[m]?.today?.mae) && isNum(by[m]?.tomorrow?.mae) ? by[m].today.mae + by[m].tomorrow.mae : Infinity;
  const days = (m) => by[m]?.today?.days ?? 0;
  const enough = (m) => days(m) >= minDays;
  const names = models.map((m) => m.name).filter((m) => by[m]);
  const ours = names.filter((m) => m !== reference && enough(m)).sort((a, b) => total(a) - total(b));
  const few = names.filter((m) => m !== reference && !enough(m)).sort((a, b) => total(a) - total(b));
  const order = [...(by[reference] ? [reference] : []), ...ours, ...few];
  const best = ours[0];
  // Fallback when the data has no paired ratio: total miss relative to "same day last week".
  const lazy = total("naive_week");
  const vsWeek = (m) => isNum(relative[m]) ? relative[m] : (isNum(lazy) && isFinite(total(m)) ? (100 * total(m)) / lazy : null);

  table.replaceChildren(
    h("caption", { class: "sr-only", text: caption }),
    h("thead", {}, h("tr", {},
      h("th", { scope: "col", text: "Model" }),
      ...leads.flatMap((l) => [h("th", { scope: "col", text: `Average miss ${l}` }), h("th", { scope: "col", class: "pct", text: `% ${l}` })]),
      h("th", { scope: "col", text: "vs last week" }),
      h("th", { scope: "col", text: "Days" }))));
  const body = h("tbody");
  for (const m of order) {
    const r = by[m];
    const tags = [];
    if (m === reference) tags.push(h("span", { class: "tag", text: "reference" }));
    if (m === best) tags.push(h("span", { class: "tag best", text: "best of mine" }));
    if (m !== reference && !enough(m)) tags.push(h("span", { class: "tag", text: "too few days" }));
    const rel = vsWeek(m);
    body.append(h("tr", { class: [m === best && "best", m === reference && "ref", !enough(m) && "few"].filter(Boolean).join(" ") || null },
      h("th", { scope: "row" }, label[m] ?? m, ...tags),
      ...leads.flatMap((l) => [h("td", { text: fmtMW(r[l]?.mae) }), h("td", { class: "pct", text: isNum(r[l]?.mape) ? `${r[l].mape.toFixed(1)}%` : "–" })]),
      h("td", { text: isNum(rel) ? String(Math.round(rel)) : "–" }),
      h("td", { text: String(days(m) || "–") })));
  }
  table.append(body);
  return by;
}

function updateScrollHints() {
  for (const wrap of document.querySelectorAll(".table-wrap")) {
    const scrolls = wrap.scrollWidth > wrap.clientWidth + 1;
    wrap.classList.toggle("scrolls", scrolls);
    const hint = wrap.nextElementSibling;
    if (hint?.classList.contains("scroll-hint")) hint.hidden = !scrolls;
  }
}

// ---- backtest panel ----
function fillFigures(bt) {
  const score = (m, l) => bt.scores.find((s) => s.model === m && s.lead === l);
  const cmp = (a, b) => (bt.comparisons || []).find((c) => c.a === a && c.b === b);
  const range = (m) => (bt.ranges || []).filter((r) => r.model === m);
  const benchmark = Math.min(...["naive_day", "naive_week", "mean_4w"].map((m) => (score(m, "today")?.mae ?? Infinity) + (score(m, "tomorrow")?.mae ?? Infinity)));
  const figure = {
    mape: (m, l) => l ? score(m, l)?.mape?.toFixed(1) : (isNum(score(m, "today")?.mape) ? String(Math.round((score(m, "today").mape + score(m, "tomorrow").mape) / 2)) : null),
    cmp: (a, b, what) => { const c = cmp(a, b); return !c ? null : what === "pct" ? String(round5(c.a_better_pct)) : String(Math.round(c.mw)); },
    range: (m) => { const r = range(m); return r.length ? String(Math.round(r.reduce((t, x) => t + x.inside_pct, 0) / r.length)) : null; },
    cut: (m) => { const t = (score(m, "today")?.mae ?? NaN) + (score(m, "tomorrow")?.mae ?? NaN); return isNum(t) ? `${round5(100 * (1 - t / benchmark))}%` : null; },
  };
  for (const span of document.querySelectorAll("[data-fig]")) {
    const [kind, ...args] = span.dataset.fig.split(":");
    const v = figure[kind]?.(...args);
    if (v != null) span.textContent = v;
  }
}

function renderBacktest(bt) {
  backtest = bt;
  $("bt-start").textContent = fmtDate(bt.start);
  $("bt-end").textContent = fmtDate(bt.end);
  const by = scoreTable($("bt-table"), bt.scores, bt.models, { relative: bt.relative, caption: "Backtest: average miss by model, lower is better" });
  // Say so when a model was scored on fewer half-hours than the rest (AEMO has a few unpublished runs).
  const n = (m) => (by[m]?.today?.n ?? 0) + (by[m]?.tomorrow?.n ?? 0);
  const most = Math.max(...Object.keys(by).map(n));
  const label = Object.fromEntries(bt.models.map((m) => [m.name, m.label]));
  const gaps = Object.keys(by).filter((m) => n(m) && n(m) < most)
    .map((m) => `${label[m]} is missing ${(most - n(m)).toLocaleString("en-AU")} of ${most.toLocaleString("en-AU")} half-hours, so it's scored on the rest.`);
  $("bt-gaps").textContent = gaps.join(" ");
  fillFigures(bt);
  const notes = Object.entries(RANGE_MODELS).map(([m, short]) => {
    const r = Object.fromEntries((bt.ranges || []).filter((x) => x.model === m).map((x) => [x.lead, x]));
    return r.today && r.tomorrow ? `${short}'s held the actual demand ${r.today.inside_pct.toFixed(0)}% of the time today and `
      + `${r.tomorrow.inside_pct.toFixed(0)}% tomorrow, and was ${fmtMW(r.today.width_mw)} wide on average today, ${fmtMW(r.tomorrow.width_mw)} tomorrow.` : null;
  }).filter(Boolean);
  if (notes.length) $("range-note").textContent = `Likely ranges: a model's 80% range should hold the actual demand about 80% of the time. ${notes.join(" ")}`;
  if (bt.run) $("bt-run").textContent = `Run on ${new Date(bt.run.at).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" })} from commit ${bt.run.commit} with: ${bt.run.command}`;
  updateRangeLabels();
  updateScrollHints();
}

function rangeHeld(m) {
  const r = (backtest?.ranges || []).filter((x) => x.model === m);
  return r.length ? Math.round(r.reduce((t, x) => t + x.inside_pct, 0) / r.length) : null;
}

function updateRangeLabels() {
  for (const span of document.querySelectorAll("[data-range-label]")) {
    const m = span.dataset.rangeLabel, held = rangeHeld(m);
    span.textContent = held == null ? RANGE_MODELS[m] : `${RANGE_MODELS[m]} (held ${held}% in the backtest)`;
  }
}

// ---- forecast panel ----
function peaks(points, origin) {
  // Highest half-hour in each of the two forecast days. A half-hour belongs to the day it starts in (labels are period-ending).
  const out = [];
  for (let d = 0; d < 2; d++) {
    const start = +origin + d * 864e5, end = start + 864e5;
    const day = points.filter((p) => isNum(p[1]) && +p[0] > start && +p[0] <= end);
    if (day.length) out.push(day.reduce((a, b) => (b[1] > a[1] ? b : a)));
  }
  return out;
}

function summary(series, now) {
  const best = series.find((s) => s.name === BEST) || series.find((s) => s.name === "lightgbm");
  const aemo = series.find((s) => s.name === "aemo");
  const past = series[0].points.filter((p) => isNum(p[1]));
  if (!best) return null;
  const mine = peaks(best.points, now);
  const theirs = aemo ? peaks(aemo.points, now) : [];
  const p = $("summary");
  const at = (pk) => [h("strong", { text: fmtMW(pk[1]) }), ` on ${fmtWeekday(pk[0])} around ${fmtClock(pk[0])}`];
  p.replaceChildren(
    "Expected peak, from the best of my models: ", ...at(mine[0]), ...(mine[1] ? [", and ", ...at(mine[1])] : []), ".",
    theirs.length === mine.length ? ` AEMO expects ${theirs.map((t) => fmtMW(t[1])).join(" and ")}.` : "",
    past.length ? ` For scale, the past week ran from ${fmtMW(Math.min(...past.map((q) => q[1])))} to ${fmtMW(Math.max(...past.map((q) => q[1])))}.` : "");
  p.hidden = false;
  return p.textContent;
}

function metaLine(latest, now) {
  const issued = new Date(latest.issued);
  const made = issued.toLocaleString("en-AU", { timeZone: "Australia/Brisbane", weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
  const upTo = now.getUTCHours() === 0 && now.getUTCMinutes() === 0 ? `midnight at the start of ${fmtWeekday(now)} ${now.getUTCDate()}` : fmtTime(now);
  const parts = [`Forecast made ${made} AEST, using data up to ${upTo}.`];
  const offsetH = (issued - (now - 10 * 36e5)) / 36e5;  // issued minus origin, with the origin's NEM time turned into real UTC
  if (latest.late || offsetH > LATE_HOURS) parts.push("It was made late, from older data, so it isn't part of the track record.");
  const ageH = (Date.now() - issued) / 36e5;
  $("meta").classList.toggle("warn", ageH > STALE_HOURS);
  if (ageH > STALE_HOURS) parts.push(`That's ${Math.floor(ageH / 24)} day${ageH >= 48 ? "s" : ""} ago: the nightly run may have failed, so this isn't today's forecast.`);
  $("meta").textContent = parts.join(" ");
}

function forecastTable(series, now) {
  const forecasts = series.slice(1);
  const temp = series.temp;
  const times = (forecasts[0]?.points || []).map((p) => p[0]).filter((t) => t > now && t.getUTCMinutes() === 0);
  const val = (s, t) => s.points.find((p) => +p[0] === +t)?.[1];
  $("forecast-table").replaceChildren(
    h("caption", { class: "sr-only", text: "Hourly forecast of demand by model, with temperature" }),
    h("thead", {}, h("tr", {}, h("th", { scope: "col", text: "Time (AEST)" }), ...forecasts.map((s) => h("th", { scope: "col", text: s.label })),
      h("th", { scope: "col", text: "Temperature" }))),
    h("tbody", {}, ...times.map((t) => h("tr", {}, h("th", { scope: "row", text: fmtTime(t) }),
      ...forecasts.map((s) => h("td", { text: fmtMW(val(s, t)) })),
      h("td", { text: isNum(val(temp, t)) ? `${val(temp, t).toFixed(1)} °C` : "–" })))));
}

function renderForecast(latest) {
  const toPts = (a) => a.map(([t, v]) => [parse(t), v]);
  const now = parse(latest.origin);
  const labels = Object.fromEntries(latest.models.map((m) => [m.name, m.label]));
  const series = [{ name: "actual", label: "Actual", colorVar: "--actual", points: toPts(latest.actual), width: 2 }];
  for (const name of Object.keys(COLOURS).filter((n) => latest.forecasts[n])) {
    series.push({ name, label: labels[name] ?? name, colorVar: COLOURS[name], points: toPts(latest.forecasts[name]), dash: DASHES[name], width: WIDTHS[name] });
  }
  const temp = { name: "temp", label: "Temperature", colorVar: "--temp", points: toPts(latest.temp) };
  const x1 = series[1].points.at(-1)[0];
  const bands = Object.entries(RANGE_MODELS).filter(([n]) => latest.ranges?.[n]).map(([n, short]) => ({
    name: n, short, label: `${short} 80% range`, colorVar: COLOURS[n], points: latest.ranges[n].map(([t, lo, hi]) => [parse(t), lo, hi]) }));

  const saved = load();
  const state = {
    shown: new Set(saved?.shown?.filter((n) => series.some((s) => s.name === n)) ?? SHOWN_BY_DEFAULT),
    bands: new Set(saved?.bands ?? []),
    more: false,
  };
  const persist = () => save({ shown: [...state.shown], bands: [...state.bands] });

  metaLine(latest, now);
  const takeaway = summary(series, now);
  $("chart-status")?.remove();
  for (const id of ["howto", "controls", "temp-block", "forecast-details"]) $(id).hidden = false;

  // The legend doubles as the show/hide control for each model; the less useful ones sit behind "More models".
  const legend = $("legend");
  const buttons = series.slice(1).map((s) => {
    const b = h("button", { type: "button", "aria-pressed": String(state.shown.has(s.name)) }, h("span", { class: "tick", "aria-hidden": "true" }), lineKey(s), s.label);
    b.addEventListener("click", () => {
      state.shown.has(s.name) ? state.shown.delete(s.name) : state.shown.add(s.name);
      persist(); sync(); draw();
    });
    return [s, h("li", { class: SHOWN_BY_DEFAULT.includes(s.name) ? null : "extra" }, b), b];
  });
  legend.replaceChildren(h("li", {}, h("span", { class: "static" }, lineKey(series[0]), "Actual")), ...buttons.map(([, li]) => li));
  const more = $("more");
  more.hidden = !buttons.some(([s]) => !SHOWN_BY_DEFAULT.includes(s.name));
  more.onclick = () => { state.more = !state.more; sync(); };
  $("reset").onclick = () => {
    state.shown = new Set(SHOWN_BY_DEFAULT); state.bands.clear(); state.more = false;
    save(null); sync(); draw();
  };
  function sync() {
    for (const [s, li, b] of buttons) {
      b.setAttribute("aria-pressed", String(state.shown.has(s.name)));
      li.hidden = !state.more && !SHOWN_BY_DEFAULT.includes(s.name) && !state.shown.has(s.name);
    }
    more.textContent = state.more ? "Fewer models" : "More models";
    more.setAttribute("aria-expanded", String(state.more));
    for (const input of $("range-toggle").querySelectorAll("input")) input.checked = state.bands.has(input.dataset.band);
  }

  // Optional likely ranges (the foundation models' 10-90% quantiles), off by default.
  const toggle = $("range-toggle");
  toggle.querySelectorAll("label").forEach((l) => l.remove());
  if (bands.length) {
    toggle.hidden = false;
    for (const b of bands) {
      const input = h("input", { type: "checkbox", "data-band": b.name });
      input.addEventListener("change", () => { input.checked ? state.bands.add(b.name) : state.bands.delete(b.name); persist(); draw(); });
      toggle.append(h("label", { class: "check" }, input, h("span", { "data-range-label": b.name, text: b.short })));
    }
    updateRangeLabels();
  }
  sync();

  let charts = [];
  const draw = () => {
    // On a phone, show three days of history so the forecast gets more of the width.
    const narrow = $("demand-chart").clientWidth < 500;
    const x0 = narrow ? new Date(+now - 3 * 864e5) : series[0].points[0][0];
    const shown = series.filter((s, i) => i === 0 || state.shown.has(s.name));
    charts = [
      lineChart($("demand-chart"), shown, { x0, x1, now, unit: "MW", fmt: fmtMW, bands: bands.filter((b) => state.bands.has(b.name)),
        label: takeaway ? `Demand chart. ${takeaway}` : $("demand-chart").dataset.label }),
      lineChart($("temp-chart"), [temp], { x0, x1, now, unit: "°C", fmt: (v) => `${v.toFixed(1)} °C`, label: $("temp-chart").dataset.label }),
    ];
    const withTemp = Object.assign(shown.slice(), { temp });
    forecastTable(withTemp, now);
    updateScrollHints();
  };
  draw();
  // A tap anywhere outside the charts closes the tooltip (on touch screens it otherwise stays put).
  document.addEventListener("pointerdown", (ev) => { if (!charts.some((c) => c.owns(ev.target))) charts.forEach((c) => c.hide()); });
  let pending;
  addEventListener("resize", () => { clearTimeout(pending); pending = setTimeout(draw, 150); });

  if (latest.live_score?.length) {
    scoreTable($("live-table"), latest.live_score, latest.models, { relative: latest.live_relative, minDays: MIN_LIVE_DAYS,
      caption: "Live track record: average miss by model, lower is better" });
    updateScrollHints();
  } else {
    $("live-table").closest(".table-wrap").hidden = true;
    $("live-status").textContent = "Nothing to score yet: the first forecasts are still in the future.";
  }
}

function showError(e, retry) {
  $("meta").textContent = "Couldn't load the latest forecast.";
  const status = $("chart-status") || $("demand-chart").appendChild(h("p", { class: "placeholder", id: "chart-status" }));
  const button = h("button", { type: "button", class: "retry", text: "Try again" });
  button.addEventListener("click", retry);
  status.replaceChildren(`The forecast server didn't answer (${e.message}). It may be waking up or down for a moment. `, button);
  $("live-status").textContent = "The live track record comes from the same server, so it's unavailable too.";
}

async function loadForecast() {
  $("meta").textContent = "Loading the latest forecast…";
  const status = $("chart-status");
  if (status) status.textContent = "Loading…";
  let latest;
  try {
    // A sleeping server can take a few seconds to wake, so try three times before giving up.
    latest = await fetchJSON(`${API}/forecast`, [0, 2000, 5000]);
  } catch (e) {
    return showError(e, loadForecast);
  }
  renderForecast(latest);
}

// The backtest is a file in this repo, so it shows even when the forecast server is down.
fetchJSON("results/backtest.json").then(renderBacktest).catch(() => {
  $("bt-table").closest(".table-wrap").hidden = true;
  $("bt-gaps").textContent = "The backtest results didn't load; try refreshing.";
});
loadForecast();
addEventListener("resize", updateScrollHints);
