// Victorian demand forecast page: fetches the latest forecast from the API and the backtest from this repo, draws two charts.
// ?api=http://localhost:8002 points the page at a local dev server.

const API = new URLSearchParams(location.search).get("api") || "https://ryanward-io--vic-demand-forecast.us-east.modal.run";
const SVG = "http://www.w3.org/2000/svg";
const MARGIN = { top: 8, right: 12, bottom: 22, left: 48 };
// Colour follows the model, never its rank or which models are shown. The first three are shown by default.
const COLOURS = { ensemble: "--series-1", lightgbm: "--series-2", chronos2: "--series-3", lstm: "--series-4", mstl: "--series-5",
  mean_4w: "--series-6", naive_day: "--series-7", naive_week: "--series-8" };
const SHOWN_BY_DEFAULT = new Set(["ensemble", "lightgbm", "chronos2"]);

const $ = (id) => document.getElementById(id);
const parse = (s) => new Date(s + ":00Z");  // NEM time, kept as "UTC" so the browser's own zone never shifts it
const fmtTime = (d) => d.toLocaleString("en-AU", { timeZone: "UTC", weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
const fmtDay = (d) => d.toLocaleDateString("en-AU", { timeZone: "UTC", weekday: "short", day: "numeric" });
const fmtMW = (v) => v == null ? "–" : `${Math.round(v).toLocaleString("en-AU")} MW`;

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(lo, hi, n = 4) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(v);
  return ticks;
}

// series: [{name, label, colorVar, points: [[Date, value]], dashed}]
function lineChart(host, series, { x0, x1, now, unit, fmt }) {
  host.innerHTML = "";
  const W = host.clientWidth, H = host.clientHeight;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": host.dataset.label || "" }, host);
  const vals = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => v != null);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const ticks = niceTicks(lo, hi);
  const x = (d) => MARGIN.left + ((d - x0) / (x1 - x0)) * (W - MARGIN.left - MARGIN.right);
  const y = (v) => H - MARGIN.bottom - ((v - lo) / (hi - lo)) * (H - MARGIN.top - MARGIN.bottom);

  el("rect", { class: "future", x: x(now), y: MARGIN.top, width: x(x1) - x(now), height: H - MARGIN.top - MARGIN.bottom }, svg);
  for (const t of ticks) {
    el("line", { class: "grid", x1: MARGIN.left, x2: W - MARGIN.right, y1: y(t), y2: y(t) }, svg);
    el("text", { x: MARGIN.left - 6, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = unit === "MW" ? (t >= 1000 ? `${t / 1000}K` : t) : t;
  }
  const dayMs = 864e5, every = W < 500 ? 2 : 1;
  for (let d = new Date(Math.ceil(x0 / dayMs) * dayMs), i = 0; d <= x1; d = new Date(+d + dayMs), i++) {
    if (i % every) continue;
    el("text", { x: x(d), y: H - 6, "text-anchor": "middle" }, svg).textContent = fmtDay(d);
  }
  el("line", { class: "now", x1: x(now), x2: x(now), y1: MARGIN.top, y2: H - MARGIN.bottom }, svg);

  for (const s of series) {
    const d = s.points.filter((p) => p[1] != null).map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    el("path", { class: "line", d, stroke: `var(${s.colorVar})`, ...(s.dashed ? { "stroke-dasharray": "5 4" } : {}) }, svg);
  }

  // Hover layer: crosshair + dots + tooltip at the nearest half-hour.
  const cross = el("line", { class: "cross", y1: MARGIN.top, y2: H - MARGIN.bottom, visibility: "hidden" }, svg);
  const dots = series.map((s) => el("circle", { class: "dot", r: 4, fill: `var(${s.colorVar})`, visibility: "hidden" }, svg));
  const hit = el("rect", { x: MARGIN.left, y: 0, width: W - MARGIN.left - MARGIN.right, height: H, fill: "transparent" }, svg);
  const tip = $("tooltip");
  const hide = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); };
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("pointermove", (ev) => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    const t = new Date(+x0 + ((px - MARGIN.left) / (W - MARGIN.left - MARGIN.right)) * (x1 - x0));
    const snapped = new Date(Math.round(t / 18e5) * 18e5);
    const rows = [];
    series.forEach((s, i) => {
      const p = s.points.find((q) => +q[0] === +snapped);
      if (!p || p[1] == null) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", x(p[0])); dots[i].setAttribute("cy", y(p[1])); dots[i].setAttribute("visibility", "visible");
      rows.push(`<div class="row"><span class="key" style="background:var(${s.colorVar})"></span>${s.label}<span class="val">${fmt(p[1])}</span></div>`);
    });
    if (!rows.length) return hide();
    cross.setAttribute("x1", x(snapped)); cross.setAttribute("x2", x(snapped)); cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<strong>${fmtTime(snapped)}</strong>${rows.join("")}`;
    tip.hidden = false;
    const panel = tip.parentElement.getBoundingClientRect();
    // Sit beside the cursor, on whichever side has more room, so the tooltip never covers the lines being read.
    const cx = ev.clientX - panel.left;
    const left = cx > panel.width / 2 ? cx - tip.offsetWidth - 14 : cx + 14;
    tip.style.left = `${Math.max(8, Math.min(left, panel.width - tip.offsetWidth - 8))}px`;
    tip.style.top = `${box.top - panel.top + 8}px`;
  });
}

function scoreTable(table, scores, models) {
  const label = Object.fromEntries(models.map((m) => [m.name, m.label]));
  const leads = ["today", "tomorrow"];
  const by = {};
  for (const s of scores) (by[s.model] ??= {})[s.lead] = s;
  const avg = (m) => by[m.name] ? (by[m.name].today?.mae ?? 0) + (by[m.name].tomorrow?.mae ?? 0) : Infinity;
  const sorted = [...models].sort((a, b) => avg(a) - avg(b));  // best first
  const best = Math.min(...Object.values(by).map((m) => m.tomorrow?.mae ?? Infinity));
  table.innerHTML = `<thead><tr><th>Model</th>${leads.map((l) => `<th>MAE ${l}</th><th>MAPE ${l}</th>`).join("")}<th>Days</th></tr></thead>`;
  const body = document.createElement("tbody");
  for (const m of sorted) {
    const r = by[m.name];
    if (!r) continue;
    const tr = document.createElement("tr");
    if (r.tomorrow?.mae === best) tr.className = "best";
    tr.innerHTML = `<td>${label[m.name]}</td>` + leads.map((l) => `<td>${fmtMW(r[l]?.mae)}</td><td>${r[l] ? r[l].mape.toFixed(1) + "%" : "–"}</td>`).join("")
      + `<td>${r.today?.days ?? "–"}</td>`;
    body.appendChild(tr);
  }
  table.appendChild(body);
}

async function main() {
  const [latest, bt] = await Promise.all([
    fetch(`${API}/forecast`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    fetch("results/backtest.json").then((r) => r.ok ? r.json() : null).catch(() => null),
  ]);

  const toPts = (a) => a.map(([t, v]) => [parse(t), v]);
  const now = parse(latest.origin);
  const series = [{ name: "actual", label: "Actual", colorVar: "--actual", points: toPts(latest.actual), shown: true }];
  const order = Object.keys(COLOURS).filter((n) => latest.forecasts[n]);
  for (const name of order) {
    const m = latest.models.find((x) => x.name === name);
    series.push({ name, label: m.label, colorVar: COLOURS[name], points: toPts(latest.forecasts[name]), dashed: true, shown: SHOWN_BY_DEFAULT.has(name) });
  }
  const x0 = series[0].points[0][0], x1 = series[1].points.at(-1)[0];

  $("meta").textContent = `Issued ${new Date(latest.issued).toLocaleString("en-AU", { dateStyle: "medium", timeStyle: "short" })} (your time), from data up to ${fmtTime(now)} NEM time.`;
  // The legend doubles as the show/hide control for each model.
  $("legend").innerHTML = series.map((s, i) => {
    const key = `<span class="key${s.dashed ? " dashed" : ""}" style="border-color:var(${s.colorVar})"></span>${s.label}`;
    return i === 0 ? `<li><span class="static">${key}</span></li>` : `<li><button type="button" data-i="${i}" aria-pressed="${s.shown}">${key}</button></li>`;
  }).join("");
  $("legend").addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    const s = series[b.dataset.i];
    s.shown = !s.shown;
    b.setAttribute("aria-pressed", s.shown);
    draw();
  });

  const draw = () => {
    lineChart($("demand-chart"), series.filter((s) => s.shown), { x0, x1, now, unit: "MW", fmt: fmtMW });
    lineChart($("temp-chart"), [{ name: "temp", label: "Temperature", colorVar: "--temp", points: toPts(latest.temp) }],
      { x0, x1, now, unit: "C", fmt: (v) => `${v.toFixed(1)} °C` });
  };
  $("demand-chart").dataset.label = "Line chart of Victorian electricity demand for the past week and each model's forecast for the next two days";
  $("temp-chart").dataset.label = "Line chart of Melbourne temperature, observed then forecast";
  draw();
  let pending;
  addEventListener("resize", () => { clearTimeout(pending); pending = setTimeout(draw, 150); });

  if (bt) {
    $("bt-meta").textContent = `${bt.start} to ${bt.end}, one forecast per day.`;
    scoreTable($("bt-table"), bt.scores, bt.models);
  }
  if (latest.live_score.length) scoreTable($("live-table"), latest.live_score, latest.models);
  else $("live-table").outerHTML = `<p class="meta">Nothing to score yet: the first forecasts are still in the future.</p>`;
}

main().catch((e) => { $("meta").textContent = `Couldn't load the forecast (${e.message}). The server may be waking up; try refreshing.`; });
