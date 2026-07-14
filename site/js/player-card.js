/* player-card.js — shared player-card modal used by Player Stats and Scout.
 *
 * Self-contained: injects its own CSS + modal markup on load and exposes a tiny
 * API. Both pages hand it a fully-formed player row (the same shape used on the
 * Player Stats table) so the card renders identically wherever it is opened.
 *
 *   PlayerCard.open(row, {season, curSeason, hasOnOff, resolveRapm})
 *   PlayerCard.close()
 *
 * Dependencies (already loaded on both pages): common.js (loadJSON, logoUrl,
 * playerUrl), zone-chart.js (ZoneChart). Plotly is optional — the shot-density
 * chart is skipped gracefully if window.Plotly is absent.
 */
(function () {
  "use strict";

  // ── one-time CSS injection ────────────────────────────────────────────────
  const CSS = `
  .pc-overlay { display:none; position:fixed; inset:0; z-index:200;
    background:rgba(0,0,0,.72); backdrop-filter:blur(4px);
    align-items:center; justify-content:center; }
  .pc-overlay.open { display:flex; }
  .pc-modal { background:var(--card); border:1px solid var(--line); border-radius:20px;
    width:min(680px,95vw); max-height:90vh; overflow-y:auto; position:relative; padding:0; }
  .pc-close { position:absolute; top:14px; right:16px;
    background:var(--card2); border:1px solid var(--line); border-radius:50%;
    width:32px; height:32px; font-size:1.1rem; line-height:32px; text-align:center;
    cursor:pointer; color:var(--muted); z-index:10; }
  .pc-close:hover { color:var(--ink); }
  .pc-header { display:flex; gap:20px; align-items:flex-start; padding:28px 28px 20px;
    border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5;
    background:var(--card); border-top-left-radius:20px; border-top-right-radius:20px; }
  .pc-headshot { width:100px; height:100px; border-radius:14px; object-fit:cover;
    object-position:top; background:var(--card2); flex-shrink:0; }
  .pc-identity { flex:1; min-width:0; }
  .pc-name { font-size:1.35rem; font-weight:800; line-height:1.2; margin-bottom:4px; }
  .pc-team { display:flex; align-items:center; gap:8px; font-size:.9rem; color:var(--muted); margin-bottom:10px; }
  .pc-team img { width:22px; height:22px; object-fit:contain; }
  .pc-chips { display:flex; gap:7px; flex-wrap:wrap; }
  .pc-chip { display:inline-block; font-size:.72rem; font-weight:700; padding:3px 10px;
    border-radius:20px; background:var(--card2); color:var(--muted); letter-spacing:.03em; }
  .pc-chip.highlight { background:rgba(250,100,50,.15); color:var(--t1); }
  .pc-bio { display:flex; gap:0; border-bottom:1px solid var(--line); }
  .pc-bio-item { flex:1; text-align:center; padding:14px 10px; border-right:1px solid var(--line); }
  .pc-bio-item:last-child { border-right:none; }
  .pc-bio-val { font-size:1.05rem; font-weight:800; }
  .pc-bio-lbl { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-top:2px; }
  .pc-stats { display:grid; grid-template-columns:1fr 1fr; gap:0; padding:20px 28px 24px; }
  .pc-stat-col { display:flex; flex-direction:column; gap:4px; }
  .pc-stat-col:first-child { padding-right:20px; border-right:1px solid var(--line); }
  .pc-stat-col:last-child  { padding-left:20px; }
  .pc-stat-row { display:flex; justify-content:space-between; align-items:center;
    padding:5px 0; border-bottom:1px solid #1a2235; }
  .pc-stat-row:last-child { border-bottom:none; }
  .pc-stat-lbl { font-size:.8rem; color:var(--muted); }
  .pc-stat-val { font-size:.95rem; font-weight:700; font-variant-numeric:tabular-nums; }
  .pc-stat-val.good { color:#16A34A; }
  .pc-stat-val.bad  { color:#DC2626; }
  .pc-col-hdr { font-size:.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.06em; color:var(--muted); margin-bottom:6px; }
  #pc-chart .js-plotly-plot { border-radius:10px; overflow:hidden; }
  `;

  const HTML = `
  <div class="pc-overlay" id="pc-overlay">
    <div class="pc-modal" id="pc-modal">
      <div class="pc-close" id="pc-close-btn">✕</div>
      <div class="pc-header">
        <img id="pc-img" class="pc-headshot" src="" alt="">
        <div class="pc-identity">
          <div class="pc-name" id="pc-name"></div>
          <div class="pc-team">
            <img id="pc-logo" src="" alt="" onerror="this.style.opacity=0">
            <span id="pc-team"></span>
            <span style="color:var(--line)">·</span>
            <span id="pc-conf" style="font-size:.82rem"></span>
          </div>
          <div class="pc-chips" id="pc-chips"></div>
        </div>
      </div>
      <div class="pc-bio" id="pc-bio"></div>
      <div class="pc-stats">
        <div class="pc-stat-col">
          <div class="pc-col-hdr">Box Score</div>
          <div id="pc-box"></div>
        </div>
        <div class="pc-stat-col">
          <div class="pc-col-hdr">Advanced</div>
          <div id="pc-adv"></div>
        </div>
      </div>
      <div id="pc-rapm-section" style="border-top:1px solid var(--line); padding:18px 28px 4px; display:none">
        <div class="pc-col-hdr" style="margin-bottom:8px">RAPM / 100 possessions</div>
        <div class="pc-stats" style="padding:0 0 18px">
          <div class="pc-stat-col" id="pc-rapm-left"></div>
          <div class="pc-stat-col" id="pc-rapm-right"></div>
        </div>
      </div>
      <div id="pc-chart-section" style="border-top:1px solid var(--line); padding:20px 28px 24px">
        <div id="pc-chart-status" style="color:var(--muted);font-size:.85rem;min-height:24px"></div>
        <div id="pc-charts" style="display:none">
          <div class="pc-col-hdr" style="margin-bottom:6px">Zone Efficiency</div>
          <div id="pc-chart-zone"></div>
          <div class="pc-col-hdr" style="margin:16px 0 6px">Shot Density</div>
          <div id="pc-chart-density"></div>
        </div>
      </div>
    </div>
  </div>`;

  function injectOnce() {
    if (document.getElementById("pc-overlay")) return;
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    const holder = document.createElement("div");
    holder.innerHTML = HTML;
    document.body.appendChild(holder.firstElementChild);
    document.getElementById("pc-close-btn").addEventListener("click", close);
    document.getElementById("pc-overlay").addEventListener("click", (e) => {
      if (e.target === document.getElementById("pc-overlay")) close();
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  }

  // ── helpers ───────────────────────────────────────────────────────────────
  function posGroup(pos) {
    if (!pos) return "";
    const p = pos.toLowerCase();
    if (p.includes("center")) return "C";
    if (p.includes("forward")) return "F";
    return "G";
  }

  // Court geometry (X_RANGE/Y_RANGE, courtTraces, shotsXY, densityTraces) comes
  // from js/court.js, loaded before this module on both host pages.
  const SC_LAYOUT = {
    paper_bgcolor: "#0B0E14", plot_bgcolor: "#0B0E14",
    height: 400, margin: { l: 10, r: 10, t: 10, b: 10 },
    showlegend: false, dragmode: false,
    xaxis: { range: X_RANGE, visible: false, scaleanchor: "y", scaleratio: 1, constrain: "domain" },
    yaxis: { range: Y_RANGE, visible: false, constrain: "domain" },
  };
  const SC_CFG = { displaylogo: false, staticPlot: true };

  let PZB = null;   // {zones, baselines} — lazy-loaded once
  async function ensureBaselines() { if (!PZB) PZB = await loadJSON("data/zone-baselines.json"); return PZB; }
  function renderZone(divEl, zoneData) { ZoneChart.render(divEl, zoneData, { zones: PZB?.zones, baselines: PZB?.baselines, minAtt: 5 }); }
  const getXY = (shots, piIdx) => shotsXY(shots, (z, pi) => pi === piIdx);
  function renderDensity(divEl, mx, my, hx, hy) {
    const traces = densityTraces(mx, my, hx, hy);
    // Plotly loads on demand (see ensurePlotly in common.js); degrade quietly if blocked.
    ensurePlotly()
      .then(Plotly => Plotly.react(divEl, traces, { ...SC_LAYOUT, plot_bgcolor: "#F0E8D5" }, SC_CFG))
      .catch(() => {});
  }
  const TEAM_SHOT_CACHE = {};
  async function loadTeamShots(teamName) {
    if (TEAM_SHOT_CACHE[teamName]) return TEAM_SHOT_CACHE[teamName];
    const data = await loadJSON(`data/shots/${slugify(teamName)}.json`);
    if (data) TEAM_SHOT_CACHE[teamName] = data;
    return data;
  }

  // ── open / close ──────────────────────────────────────────────────────────
  async function open(p, opts = {}) {
    injectOnce();
    const { season, curSeason, hasOnOff = false, resolveRapm = null } = opts;
    const pg = posGroup(p.pos);

    document.getElementById("pc-img").src = playerUrl(p.id);
    document.getElementById("pc-name").textContent = p.n ?? "—";
    document.getElementById("pc-logo").src = logoUrl(p.tid);
    document.getElementById("pc-team").textContent = p.t ?? "—";
    document.getElementById("pc-conf").textContent = p.conf ?? "";

    const chips = [];
    if (p.jn)  chips.push(`<span class="pc-chip">#${p.jn}</span>`);
    if (p.pos) chips.push(`<span class="pc-chip pos-${pg}">${p.pos}</span>`);
    if (p.exp) chips.push(`<span class="pc-chip highlight">${p.exp}</span>`);
    document.getElementById("pc-chips").innerHTML = chips.join("");

    const bioItems = [
      { val: p.ht ?? "—", lbl: "Height" },
      { val: p.wt ?? "—", lbl: "Weight" },
      { val: p.hw ?? "—", lbl: "Hometown" },
      { val: p.gp ?? "—", lbl: "Games" },
    ];
    document.getElementById("pc-bio").innerHTML = bioItems.map((b) =>
      `<div class="pc-bio-item"><div class="pc-bio-val">${b.val}</div><div class="pc-bio-lbl">${b.lbl}</div></div>`
    ).join("");

    const boxStats = [
      { lbl: "PPG", val: p.ppg?.toFixed(1) ?? "—" },
      { lbl: "RPG", val: p.rpg?.toFixed(1) ?? "—" },
      { lbl: "APG", val: p.apg?.toFixed(1) ?? "—" },
      { lbl: "SPG", val: p.spg?.toFixed(1) ?? "—" },
      { lbl: "BPG", val: p.bpg?.toFixed(1) ?? "—" },
      { lbl: "TOV", val: p.tpg?.toFixed(1) ?? "—" },
      { lbl: "MPG", val: p.mpg?.toFixed(1) ?? "—" },
    ];
    const pc1 = (val) => val != null ? val.toFixed(1) : "—";
    const advStats = [
      { lbl: "TS%",    val: p.ts   != null ? (p.ts * 100).toFixed(1) + "%"  : "—" },
      { lbl: "eFG%",   val: p.efg  != null ? (p.efg * 100).toFixed(1) + "%" : "—" },
      { lbl: "USG%",   val: pc1(p.usg) },
      { lbl: "Resp%",  val: p.resp != null ? (p.resp * 100).toFixed(1) + "%" : "—" },
      { lbl: "AST%",   val: pc1(p.astp) },
      { lbl: "TOV%",   val: pc1(p.tovp) },
      { lbl: "ORB%",   val: pc1(p.orbp) },
      { lbl: "DRB%",   val: pc1(p.drbp) },
      { lbl: "TRB%",   val: pc1(p.trbp) },
      { lbl: "STL%",   val: pc1(p.stlp) },
      { lbl: "BLK%",   val: pc1(p.blkp) },
      { lbl: "Ast'd%", val: pc1(p.astdp) },
      { lbl: "FG%",    val: p.fg   != null ? (p.fg * 100).toFixed(1) + "%"  : "—" },
      { lbl: "3P%",    val: p.fg3  != null ? (p.fg3 * 100).toFixed(1) + "%" : "—" },
      { lbl: "FT%",    val: p.ft   != null ? (p.ft * 100).toFixed(1) + "%"  : "—" },
      { lbl: "Rim%",   val: p.rimr  != null ? p.rimr.toFixed(1) + "%"  : "—" },
      { lbl: "Mid%",   val: p.midr  != null ? p.midr.toFixed(1) + "%"  : "—" },
      { lbl: "3PA%",   val: p.thr3r != null ? p.thr3r.toFixed(1) + "%" : "—" },
    ];
    if (hasOnOff) {
      advStats.push({ lbl: "On/Off", val: p.on_off != null
          ? (p.on_off > 0 ? "+" : "") + p.on_off.toFixed(1) : "—",
        cls: p.on_off == null ? "" : p.on_off > 0 ? "good" : "bad" });
    }

    const statRow = (s) =>
      `<div class="pc-stat-row"><span class="pc-stat-lbl">${s.lbl}</span>
       <span class="pc-stat-val ${s.cls ?? ""}" ${s.bold ? 'style="font-weight:900"' : ""}>${s.val}</span></div>`;

    document.getElementById("pc-box").innerHTML = boxStats.map(statRow).join("");
    document.getElementById("pc-adv").innerHTML = advStats.map(statRow).join("");

    document.getElementById("pc-overlay").classList.add("open");
    document.body.style.overflow = "hidden";

    // Shot charts — only the live season has shot-coordinate data.
    const chartSection = document.getElementById("pc-chart-section");
    const chartStatus  = document.getElementById("pc-chart-status");
    const chartsWrap   = document.getElementById("pc-charts");
    if (season != null && curSeason != null && season !== curSeason) {
      chartSection.style.display = "none";
    } else {
      chartSection.style.display = "block";
      chartStatus.textContent = "Loading shot data…";
      chartsWrap.style.display = "none";
      loadTeamShots(p.t).then(async (td) => {
        if (!td) { chartStatus.textContent = "Shot data not available."; return; }
        const piIdx = td.players.indexOf(p.n);
        if (piIdx === -1) { chartStatus.textContent = "No shot data for this player."; return; }
        const zoneData = ZoneChart.computeZones(td.shots, (g, idx) => idx === piIdx);
        if (!zoneData.length) { chartStatus.textContent = "No shot attempts found."; return; }
        const { mx, my, hx, hy } = getXY(td.shots, piIdx);
        await ensureBaselines();
        chartStatus.textContent = "";
        chartsWrap.style.display = "block";
        renderZone(document.getElementById("pc-chart-zone"), zoneData);
        renderDensity(document.getElementById("pc-chart-density"), mx, my, hx, hy);
      }).catch(() => { chartStatus.textContent = "Shot data not available."; });
    }

    // RAPM block — read from the row, or resolve it lazily if a provider is given.
    const rapmSection = document.getElementById("pc-rapm-section");
    const r = resolveRapm ? (await resolveRapm(p)) || p : p;
    if (r.rapm != null || r.rapm_bp != null) {
      const sg = (val) => val == null ? "—" : (val >= 0 ? "+" : "−") + Math.abs(val).toFixed(2);
      const sc = (val) => val == null ? "" : val > 0 ? "good" : val < 0 ? "bad" : "";
      const rRow = (lbl, val, bold) =>
        `<div class="pc-stat-row"><span class="pc-stat-lbl">${lbl}</span>
         <span class="pc-stat-val ${sc(val)}" ${bold ? 'style="font-weight:900"' : ""}>${sg(val)}</span></div>`;
      document.getElementById("pc-rapm-left").innerHTML =
        `<div class="pc-col-hdr">RAPM</div>` +
        rRow("Offense", r.orapm) + rRow("Defense", r.drapm) + rRow("Total", r.rapm, true);
      document.getElementById("pc-rapm-right").innerHTML =
        `<div class="pc-col-hdr">Box-Prior</div>` +
        rRow("Offense", r.orapm_bp) + rRow("Defense", r.drapm_bp) + rRow("Total", r.rapm_bp, true);
      rapmSection.style.display = "block";
    } else {
      rapmSection.style.display = "none";
    }
  }

  function close() {
    const overlay = document.getElementById("pc-overlay");
    if (!overlay) return;
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    const z = document.getElementById("pc-chart-zone");
    const d = document.getElementById("pc-chart-density");
    if (window.Plotly && d) Plotly.purge(d);
    if (z) z.innerHTML = "";
  }

  window.PlayerCard = { open, close };
})();
