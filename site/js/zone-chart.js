/* ════════════════════════════════════════════════════════════════════════
   zone-chart.js — shared 14-wedge "2K-style" hot-zone shot chart.

   Used by shot-charts.html and the player-card modals (player-stats.html,
   impact.html).  Renders filled, colour-graded SVG polygons; colour is FG%
   relative to the NCAA average for each zone.

   Boundaries MUST stay identical to build_site.py classify_shot_zones().
   Zone index order (matches build_site.ZONE_NAMES):
     0 Restricted Area
     1-3  Close Mid Right / Center / Left
     4-8  Mid Right / Right-Center / Center / Left-Center / Left
     9-13 3PT Right / Right-Center / Center / Left-Center / Left

   API:
     ZoneChart.classifyZone(px, py)               → 0..13 or -1   (px=-coord_y, py=47-|coord_x|)
     ZoneChart.computeZones(shots, keepFn)         → [[zi, makes, atts], …]
     ZoneChart.render(containerEl, zoneData, opts) → draws SVG
        opts = { zones, baselines, title, subtitle, minAtt, compact }
   ════════════════════════════════════════════════════════════════════════ */
const ZoneChart = (function () {
  "use strict";

  const HOOP_Y = 5.25;
  const ZONE_NAMES = [
    "Restricted Area",
    "Close Mid - Right", "Close Mid - Center", "Close Mid - Left",
    "Mid - Right", "Mid - Right Center", "Mid - Center",
    "Mid - Left Center", "Mid - Left",
    "3PT - Right", "3PT - Right Center", "3PT - Center",
    "3PT - Left Center", "3PT - Left",
  ];

  // ── classification (mirrors build_site.classify_shot_zones) ──────────────
  function classifyZone(px, py) {
    const lat = px, out = py - HOOP_Y;
    const dist = Math.hypot(lat, out);
    if (dist >= 40) return -1;
    let ang = Math.atan2(out, lat) * 180 / Math.PI;
    if (ang < -90) ang += 360;
    const RA = 4.0, CLOSE = 11.0, THREE = 22.146, CX = 21.65;
    const YMEET = Math.sqrt(THREE * THREE - CX * CX);
    if (dist < RA) return 0;
    const is3 = dist >= THREE || (Math.abs(lat) >= CX && out <= YMEET);
    if (is3) {
      if (ang < 36) return 9; if (ang < 72) return 10; if (ang < 108) return 11;
      if (ang < 144) return 12; return 13;
    }
    if (dist < CLOSE) { if (ang < 60) return 1; if (ang < 120) return 2; return 3; }
    if (ang < 36) return 4; if (ang < 72) return 5; if (ang < 108) return 6;
    if (ang < 144) return 7; return 8;
  }

  function computeZones(shots, keepFn) {
    const makes = new Array(14).fill(0), atts = new Array(14).fill(0);
    for (let i = 0; i < shots.length; i += 5) {
      if (keepFn && !keepFn(shots[i], shots[i + 4])) continue;
      const zi = classifyZone(shots[i + 1] / 10, shots[i + 2] / 10);
      if (zi < 0) continue;
      atts[zi]++; if (shots[i + 3]) makes[zi]++;
    }
    const out = [];
    for (let zi = 0; zi < 14; zi++) if (atts[zi] > 0) out.push([zi, makes[zi], atts[zi]]);
    return out;
  }

  // ── colour: FG% relative to the zone's NCAA baseline ─────────────────────
  function familyBaseline(name) {
    if (name === "Restricted Area") return 0.62;
    if (name.startsWith("Close Mid")) return 0.42;
    if (name.startsWith("3PT")) return 0.345;
    return 0.36;
  }
  function baselineFor(name, zones, baselines) {
    if (zones && baselines) {
      const i = zones.indexOf(name);
      if (i >= 0 && baselines[i] != null && !isNaN(baselines[i])) return baselines[i];
    }
    return familyBaseline(name);
  }
  const SPAN = 0.15;
  function colorRel(pct, base) {
    let d = Math.max(-SPAN, Math.min(SPAN, pct - base));
    const t = (d + SPAN) / (2 * SPAN);
    const stops = [[0, [122, 0, 0]], [0.25, [192, 57, 43]], [0.5, [232, 216, 184]], [0.75, [39, 174, 96]], [1, [10, 74, 37]]];
    for (let i = 0; i < stops.length - 1; i++) {
      if (t <= stops[i + 1][0]) {
        const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]), a = stops[i][1], b = stops[i + 1][1];
        return `rgb(${Math.round(a[0] + f * (b[0] - a[0]))},${Math.round(a[1] + f * (b[1] - a[1]))},${Math.round(a[2] + f * (b[2] - a[2]))})`;
      }
    }
    return `rgb(${stops[stops.length - 1][1].join(",")})`;
  }

  // ── SVG geometry (tenths-of-feet, hoop at polar origin) ──────────────────
  const NS = "http://www.w3.org/2000/svg";
  const Y_MIN = -58, Y_MAX = 250, PAD = 10;
  const VB_W = 500 + 2 * PAD, VB_H = (Y_MAX - Y_MIN) + 2 * PAD;
  const cx = x => x + 250 + PAD, cy = y => (Y_MAX - y) + PAD;
  const pt = (x, y) => `${cx(x).toFixed(2)},${cy(y).toFixed(2)}`;
  const R_RA = 40, R_CLOSE = 110, R_3 = 221.46, CORNER = 216.5, Y_BASE = -52.5;
  const Y_MEET = Math.sqrt(R_3 * R_3 - CORNER * CORNER);
  const polar = (r, d) => [r * Math.cos(d * Math.PI / 180), r * Math.sin(d * Math.PI / 180)];
  const arc = (r, a1, a2) => {
    const sweep = a2 > a1 ? 0 : 1, large = Math.abs(a2 - a1) > 180 ? 1 : 0;
    const x2 = r * Math.cos(a2 * Math.PI / 180), y2 = r * Math.sin(a2 * Math.PI / 180);
    return `A${r.toFixed(2)},${r.toFixed(2)} 0 ${large},${sweep} ${pt(x2, y2)}`;
  };
  const Mv = (x, y) => `M${pt(x, y)}`, Lv = (x, y) => `L${pt(x, y)}`;

  const ZGEO = {
    "Restricted Area": () => Mv(-R_RA, Y_BASE) + Lv(-R_RA, 0) + arc(R_RA, 180, 0) + Lv(R_RA, Y_BASE) + " Z",
    "Close Mid - Right": () => { let d = Mv(R_RA, Y_BASE) + Lv(R_CLOSE, Y_BASE) + Lv(R_CLOSE, 0) + arc(R_CLOSE, 0, 60); const p = polar(R_RA, 60); return d + Lv(p[0], p[1]) + arc(R_RA, 60, 0) + " Z"; },
    "Close Mid - Center": () => { const p1 = polar(R_CLOSE, 60), p2 = polar(R_RA, 120), s = polar(R_RA, 60); return Mv(s[0], s[1]) + Lv(p1[0], p1[1]) + arc(R_CLOSE, 60, 120) + Lv(p2[0], p2[1]) + arc(R_RA, 120, 60) + " Z"; },
    "Close Mid - Left": () => { const p1 = polar(R_CLOSE, 120), p2 = polar(R_RA, 120); return Mv(p2[0], p2[1]) + Lv(p1[0], p1[1]) + arc(R_CLOSE, 120, 180) + Lv(-R_CLOSE, Y_BASE) + Lv(-R_RA, Y_BASE) + Lv(-R_RA, 0) + arc(R_RA, 180, 120) + " Z"; },
    "Mid - Right": () => { const am = Math.atan2(Y_MEET, CORNER) * 180 / Math.PI, p36 = polar(R_CLOSE, 36); return Mv(R_CLOSE, Y_BASE) + Lv(CORNER, Y_BASE) + Lv(CORNER, Y_MEET) + arc(R_3, am, 36) + Lv(p36[0], p36[1]) + arc(R_CLOSE, 36, 0) + " Z"; },
    "Mid - Right Center": () => { const p1 = polar(R_CLOSE, 36), p2 = polar(R_3, 36), p72 = polar(R_CLOSE, 72); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R_3, 36, 72) + Lv(p72[0], p72[1]) + arc(R_CLOSE, 72, 36) + " Z"; },
    "Mid - Center": () => { const p1 = polar(R_CLOSE, 72), p2 = polar(R_3, 72), p108 = polar(R_CLOSE, 108); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R_3, 72, 108) + Lv(p108[0], p108[1]) + arc(R_CLOSE, 108, 72) + " Z"; },
    "Mid - Left Center": () => { const p1 = polar(R_CLOSE, 108), p2 = polar(R_3, 108), p144 = polar(R_CLOSE, 144); return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R_3, 108, 144) + Lv(p144[0], p144[1]) + arc(R_CLOSE, 144, 108) + " Z"; },
    "Mid - Left": () => { const p1 = polar(R_CLOSE, 144), p2 = polar(R_3, 144), am = Math.atan2(Y_MEET, -CORNER) * 180 / Math.PI; return Mv(p1[0], p1[1]) + Lv(p2[0], p2[1]) + arc(R_3, 144, am) + Lv(-CORNER, Y_MEET) + Lv(-CORNER, Y_BASE) + Lv(-R_CLOSE, Y_BASE) + Lv(-R_CLOSE, 0) + arc(R_CLOSE, 180, 144) + " Z"; },
    "3PT - Right": () => { const p36 = polar(500, 36), in36 = polar(R_3, 36), am = Math.atan2(Y_MEET, CORNER) * 180 / Math.PI; return Mv(CORNER, Y_BASE) + Lv(500, Y_BASE) + Lv(500, 500) + Lv(p36[0], p36[1]) + Lv(in36[0], in36[1]) + arc(R_3, 36, am) + Lv(CORNER, Y_MEET) + " Z"; },
    "3PT - Right Center": () => { const p1 = polar(R_3, 36), o = polar(500, 36), p2 = polar(R_3, 72); return Mv(p1[0], p1[1]) + Lv(o[0], o[1]) + arc(500, 36, 72) + Lv(p2[0], p2[1]) + arc(R_3, 72, 36) + " Z"; },
    "3PT - Center": () => { const p1 = polar(R_3, 72), o = polar(500, 72), p2 = polar(R_3, 108); return Mv(p1[0], p1[1]) + Lv(o[0], o[1]) + arc(500, 72, 108) + Lv(p2[0], p2[1]) + arc(R_3, 108, 72) + " Z"; },
    "3PT - Left Center": () => { const p1 = polar(R_3, 108), o = polar(500, 108), p2 = polar(R_3, 144); return Mv(p1[0], p1[1]) + Lv(o[0], o[1]) + arc(500, 108, 144) + Lv(p2[0], p2[1]) + arc(R_3, 144, 108) + " Z"; },
    "3PT - Left": () => { const am = Math.atan2(Y_MEET, -CORNER) * 180 / Math.PI, o = polar(500, 144); return Mv(-CORNER, Y_BASE) + Lv(-CORNER, Y_MEET) + arc(R_3, am, 144) + Lv(o[0], o[1]) + Lv(-500, 500) + Lv(-500, Y_BASE) + " Z"; },
  };

  const LABEL = {
    "Restricted Area": [0, 5], "Close Mid - Right": [72, 22], "Close Mid - Center": [0, 78],
    "Close Mid - Left": [-72, 22], "Mid - Right": [150, 18], "Mid - Right Center": [100, 130],
    "Mid - Center": [0, 158], "Mid - Left Center": [-100, 130], "Mid - Left": [-150, 18],
    "3PT - Right": [228, 5], "3PT - Right Center": [150, 195], "3PT - Center": [0, 232],
    "3PT - Left Center": [-150, 195], "3PT - Left": [-228, 5],
  };

  function courtLines() {
    const s = "rgba(255,255,255,0.16)", w = 1.2;
    const tm = Math.atan2(Y_MEET, CORNER) * 180 / Math.PI;
    const a = polar(R_3, tm), b = polar(R_3, 180 - tm);
    return [
      `<line x1="${cx(-250)}" y1="${cy(Y_BASE)}" x2="${cx(250)}" y2="${cy(Y_BASE)}" stroke="${s}" stroke-width="${w}"/>`,
      `<line x1="${cx(-250)}" y1="${cy(Y_BASE)}" x2="${cx(-250)}" y2="${cy(Y_MAX - 4)}" stroke="${s}" stroke-width="${w}"/>`,
      `<line x1="${cx(250)}" y1="${cy(Y_BASE)}" x2="${cx(250)}" y2="${cy(Y_MAX - 4)}" stroke="${s}" stroke-width="${w}"/>`,
      `<line x1="${cx(-CORNER)}" y1="${cy(Y_BASE)}" x2="${cx(-CORNER)}" y2="${cy(Y_MEET)}" stroke="${s}" stroke-width="${w}"/>`,
      `<line x1="${cx(CORNER)}" y1="${cy(Y_BASE)}" x2="${cx(CORNER)}" y2="${cy(Y_MEET)}" stroke="${s}" stroke-width="${w}"/>`,
      `<path d="M${cx(a[0]).toFixed(2)},${cy(a[1]).toFixed(2)} A${R_3},${R_3} 0 0,0 ${cx(b[0]).toFixed(2)},${cy(b[1]).toFixed(2)}" stroke="${s}" stroke-width="${w}" fill="none"/>`,
    ].join("");
  }

  // ── render ───────────────────────────────────────────────────────────────
  function render(container, zoneData, opts) {
    opts = opts || {};
    const zones = opts.zones || ZONE_NAMES;
    const baselines = opts.baselines || null;
    const minAtt = opts.minAtt || 5;

    const stat = {};
    for (const [zi, m, a] of zoneData) stat[zones[zi]] = { m, a, pct: a > 0 ? m / a : 0 };

    let polys = "", labels = "";
    for (const [name, fn] of Object.entries(ZGEO)) {
      const z = stat[name], base = baselineFor(name, zones, baselines);
      let fill = "#161b27", low = false, stroke = "rgba(11,14,20,0.9)", sw = 1.4;
      if (z && z.a > 0) {
        if (z.a < minAtt) { fill = "url(#zhatch)"; low = true; stroke = colorRel(z.pct, base); sw = 2; }
        else fill = colorRel(z.pct, base);
      }
      polys += `<path class="zone-poly" d="${fn()}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" data-zone="${name}"></path>`;
      if (z && z.a > 0) {
        const lp = LABEL[name];
        let col = "#fff";
        if (!low) { const mm = fill.match(/\d+/g); if (mm) col = ((0.299 * mm[0] + 0.587 * mm[1] + 0.114 * mm[2]) / 255) > 0.6 ? "#15181f" : "#fff"; }
        labels += `<text x="${cx(lp[0]).toFixed(2)}" y="${cy(lp[1]).toFixed(2)}" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="700" fill="${col}" pointer-events="none">${(z.pct * 100).toFixed(0)}%</text>`;
      }
    }

    const totA = zoneData.reduce((s, [, , a]) => s + a, 0);
    const head = opts.title
      ? `<div class="zone-head"><div class="zone-title">${opts.title}</div><div class="zone-sub">${opts.subtitle || ""}</div></div>`
      : "";
    container.innerHTML = `
      <div class="zone-card${opts.compact ? " zone-compact" : ""}">
        ${head}
        <div class="zone-area">
          <svg viewBox="0 0 ${VB_W} ${VB_H}" xmlns="${NS}">
            <defs><pattern id="zhatch" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">
              <rect width="7" height="7" fill="#1c2230"/><line x1="0" y1="0" x2="0" y2="7" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
            </pattern></defs>
            <rect x="0" y="0" width="${VB_W}" height="${VB_H}" fill="#0B0E14"/>
            ${polys}${labels}${courtLines()}
          </svg>
          <div class="zone-tip"></div>
        </div>
        <div class="zone-legend">
          <div class="cap">FG% vs NCAA average · ${totA} FGA · hatched = under ${minAtt} FGA</div>
          <div class="bar"></div>
          <div class="lab"><span>−15%</span><span>−7%</span><span>avg</span><span>+7%</span><span>+15%</span></div>
        </div>
      </div>`;

    const area = container.querySelector(".zone-area");
    const tip = container.querySelector(".zone-tip");
    area.querySelectorAll(".zone-poly").forEach(path => {
      const name = path.getAttribute("data-zone"), z = stat[name];
      if (!z || z.a === 0) return;
      const base = baselineFor(name, zones, baselines), rel = (z.pct - base) * 100;
      path.addEventListener("mousemove", e => {
        const r = area.getBoundingClientRect();
        tip.innerHTML =
          `<div class="t">${name}</div>` +
          `<div class="s">${z.m}/${z.a} FG</div>` +
          `<div class="p" style="color:${colorRel(z.pct, base)}">${(z.pct * 100).toFixed(1)}% FG</div>` +
          `<div class="r" style="color:${rel >= 0 ? "#7fd8a0" : "#e08a82"}">${rel >= 0 ? "+" : ""}${rel.toFixed(1)}% vs NCAA avg (${(base * 100).toFixed(0)}%)</div>` +
          (z.a < minAtt ? `<div class="r" style="color:#c8b27a">⚠ Low sample</div>` : "");
        tip.style.left = Math.min(e.clientX - r.left + 12, r.width - 150) + "px";
        tip.style.top = Math.max(e.clientY - r.top - 12, 0) + "px";
        tip.style.display = "block";
      });
      path.addEventListener("mouseleave", () => tip.style.display = "none");
    });
  }

  return { classifyZone, computeZones, render, ZONE_NAMES };
})();
