/* games.js — shared game data + box-score modal for the player and team cards.
 *
 *   Games.teamFile(season, teamId)   → {t, games:[{g, d, ha, o, oid, ts, os, w, players, totals}]}
 *   Games.openBoxScore({season, teamId, gameId, highlightId, chart})
 *   Games.shortDate(d) · Games.opponentLabel(game) · Games.shootLine(made, att)
 *
 * Reads data/{season}/games/teams/{team_id}.json from hoglib/gamelogs.py, which
 * stores names once per file and each line as an array in "cols" order; teamFile
 * hands back plain objects. A box score is the team's game plus the same game in
 * the opponent's file.
 *
 * chart ({kind: "player"|"team", name}) adds that game's makes and misses from
 * data/shots/{team}.json — live season only, so callers pass it only then.
 * Needs common.js (loadJSON, logoUrl, slugify, ensurePlotly) and court.js.
 */
(function () {
  "use strict";

  const CSS = `
  .gm-overlay { display:none; position:fixed; inset:0; z-index:210; background:rgba(0,0,0,.6);
    align-items:center; justify-content:center; }
  .gm-overlay.open { display:flex; }
  .gm-modal { background:var(--card); border:1px solid var(--line); border-radius:20px;
    width:min(860px,95vw); max-height:90vh; overflow-y:auto; position:relative; padding:24px 28px; }
  .gm-close { position:absolute; top:14px; right:16px; background:var(--card2); border:1px solid var(--line);
    border-radius:50%; width:32px; height:32px; font-size:1.1rem; line-height:32px; text-align:center;
    cursor:pointer; color:var(--muted); z-index:10; }
  .gm-close:hover { color:var(--ink); }
  .gm-status { color:var(--muted); font-size:.85rem; }
  .gm-score { display:flex; flex-direction:column; gap:6px; margin-bottom:4px; padding-right:40px; }
  .gm-team-line { display:flex; align-items:center; gap:10px; font-size:1.05rem; font-weight:700; color:var(--muted); }
  .gm-team-line.won { color:var(--ink); font-weight:800; }
  .gm-team-line img { width:24px; height:24px; object-fit:contain; }
  .gm-team-line .gm-pts { margin-left:auto; font-variant-numeric:tabular-nums; }
  .gm-date { color:var(--muted); font-size:.8rem; margin-bottom:18px; }
  .gm-hdr { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em;
    color:var(--muted); margin-bottom:6px; }
  .gm-section { margin-top:18px; }
  .gm-wrap { overflow-x:auto; }
  .gm-tbl { width:100%; border-collapse:collapse; font-size:.8rem; font-variant-numeric:tabular-nums; }
  .gm-tbl th { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
    color:var(--muted); text-align:right; padding:0 6px 6px; white-space:nowrap; }
  .gm-tbl td { text-align:right; padding:6px; border-top:1px solid #1a2235; white-space:nowrap; }
  .gm-tbl th:nth-child(-n+2), .gm-tbl td:nth-child(-n+2) { text-align:left; }
  .gm-tbl tr.gm-row { cursor:pointer; }
  .gm-tbl tr.gm-row:hover td { background:var(--card2); }
  .gm-tbl tr.gm-highlight td { font-weight:800; color:var(--t1); }
  .gm-tbl tr.gm-totals td { font-weight:800; border-top:2px solid var(--line); }
  .gm-tbl td.gm-dnp { color:var(--muted); text-align:left; }
  .gm-win { color:#16A34A; font-weight:700; }
  .gm-loss { color:#DC2626; font-weight:700; }
  #gm-chart .js-plotly-plot { border-radius:10px; overflow:hidden; }
  `;

  const HTML = `
  <div class="gm-overlay" id="gm-overlay">
    <div class="gm-modal">
      <div class="gm-close" id="gm-close">✕</div>
      <div id="gm-body"></div>
    </div>
  </div>`;

  // Styles go in at load: the cards' game logs use the table classes before any box score opens.
  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);

  function injectOnce() {
    if (document.getElementById("gm-overlay")) return;
    const holder = document.createElement("div");
    holder.innerHTML = HTML;
    document.body.appendChild(holder.firstElementChild);
    document.getElementById("gm-close").addEventListener("click", closeBoxScore);
    document.getElementById("gm-overlay").addEventListener("click", (e) => {
      if (e.target === document.getElementById("gm-overlay")) closeBoxScore();
    });
    // Capture phase so Escape closes the box score before the card underneath.
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isOpen()) { e.stopImmediatePropagation(); closeBoxScore(); }
    }, true);
  }

  // ── data ──────────────────────────────────────────────────────────────────
  const FILE_CACHE = {};

  function decode(raw) {
    const line = (arr, offset) => Object.fromEntries(raw.cols.map((c, i) => [c, arr[i + offset]]));
    const who = (id) => ({ id, n: raw.roster[id]?.[0] ?? "—", pos: raw.roster[id]?.[1] ?? "" });
    return {
      t: raw.t,
      games: raw.games.map((g) => ({
        ...g, w: !!g.w,
        players: [
          ...g.p.map((arr) => ({ ...who(arr[0]), st: !!arr[1], dnp: false, ...line(arr, 2) })),
          ...g.dnp.map((id) => ({ ...who(id), st: false, dnp: true })),
        ],
        totals: line(g.tot, 0),
      })),
    };
  }

  async function teamFile(season, teamId) {
    const key = `${season}/${teamId}`;
    if (!(key in FILE_CACHE)) {
      const raw = await loadJSON(`data/${season}/games/teams/${teamId}.json`);
      FILE_CACHE[key] = raw ? decode(raw) : null;
    }
    return FILE_CACHE[key];
  }

  // ── formatting ────────────────────────────────────────────────────────────
  const shootLine = (made, att) => (att ? `${made}-${att}` : "—");
  const shortDate = (d) => new Date(`${d}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const opponentLabel = (g) => `${g.ha === "away" ? "@" : "vs"} ${g.o ?? "—"}`;
  const v = (x) => x ?? "—";

  // ── box score ─────────────────────────────────────────────────────────────
  const BOX_COLS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF", "FG", "3P", "FT"];
  const statCells = (s) => `<td>${v(s.pts)}</td><td>${v(s.reb)}</td><td>${v(s.ast)}</td><td>${v(s.stl)}</td>
    <td>${v(s.blk)}</td><td>${v(s.tov)}</td><td>${v(s.pf)}</td>
    <td>${shootLine(s.fgm, s.fga)}</td><td>${shootLine(s.tpm, s.tpa)}</td><td>${shootLine(s.ftm, s.fta)}</td>`;

  function teamTable(side, highlightId) {
    const rows = side.game.players.map((p) => p.dnp
      ? `<tr><td>${p.n}</td><td>${p.pos}</td><td class="gm-dnp" colspan="${BOX_COLS.length}">DNP</td></tr>`
      : `<tr class="${p.id === highlightId ? "gm-highlight" : ""}"><td>${p.n}</td><td>${p.pos}</td>
           <td>${v(p.min)}</td>${statCells(p)}</tr>`).join("");
    return `<div class="gm-section"><div class="gm-hdr">${side.name}</div>
      <div class="gm-wrap"><table class="gm-tbl">
        <tr><th>Player</th><th>Pos</th>${BOX_COLS.map((c) => `<th>${c}</th>`).join("")}</tr>
        ${rows}
        <tr class="gm-totals"><td>Totals</td><td></td><td></td>${statCells(side.game.totals)}</tr>
      </table></div></div>`;
  }

  async function renderChart(teamName, gameId, chart) {
    const el = document.getElementById("gm-chart");
    if (!el) return;
    const shots = await loadJSON(`data/shots/${slugify(teamName)}.json`);
    const gameIndex = shots?.games?.findIndex((g) => g.id === gameId) ?? -1;
    if (gameIndex < 0) { el.closest(".gm-section").remove(); return; }
    const playerIndex = chart.kind === "player" ? shots.players.indexOf(chart.name) : null;
    const { mx, my, hx, hy } = shotsXY(shots.shots, (g, pi) =>
      g === gameIndex && (playerIndex === null || pi === playerIndex));
    if (!mx.length && !hx.length) { el.closest(".gm-section").remove(); return; }
    const marker = (color, symbol) => ({ color, size: 9, symbol, line: { color: "white", width: 0.6 }, opacity: 0.85 });
    ensurePlotly().then((Plotly) => Plotly.react(el, [
      ...courtTraces(),
      { type: "scatter", x: hx, y: hy, mode: "markers", name: `Miss (${hx.length})`, marker: marker("#EF4444", "x"), hoverinfo: "skip" },
      { type: "scatter", x: mx, y: my, mode: "markers", name: `Make (${mx.length})`, marker: marker("#22C55E", "circle"), hoverinfo: "skip" },
    ], {
      paper_bgcolor: "#0B0E14", plot_bgcolor: "#0B0E14", height: 400,
      margin: { l: 10, r: 10, t: 10, b: 10 }, dragmode: false, showlegend: true,
      legend: { orientation: "h", x: 0.5, xanchor: "center", y: 0.02, bgcolor: "rgba(17,21,31,0.7)", font: { color: "#E6E9EF" } },
      xaxis: { range: X_RANGE, visible: false, scaleanchor: "y", scaleratio: 1, constrain: "domain" },
      yaxis: { range: Y_RANGE, visible: false, constrain: "domain" },
    }, { displaylogo: false, staticPlot: true })).catch(() => {});
  }

  let boxRequest = 0;
  async function openBoxScore({ season, teamId, gameId, highlightId = null, chart = null }) {
    injectOnce();
    const request = ++boxRequest;
    const body = document.getElementById("gm-body");
    body.innerHTML = `<div class="gm-status">Loading…</div>`;
    document.getElementById("gm-overlay").classList.add("open");

    const ownFile = await teamFile(season, teamId);
    const own = ownFile?.games.find((g) => g.g === gameId);
    if (request !== boxRequest) return;
    if (!own) { body.innerHTML = `<div class="gm-status">Box score not available.</div>`; return; }
    const oppFile = await teamFile(season, own.oid);
    const opp = oppFile?.games.find((g) => g.g === gameId);
    if (request !== boxRequest) return;

    const sides = [{ name: ownFile.t, tid: teamId, game: own, score: own.ts, won: own.w, ha: own.ha }];
    if (opp) sides.push({ name: oppFile.t, tid: own.oid, game: opp, score: opp.ts, won: opp.w, ha: opp.ha });
    sides.sort((a, b) => (a.ha === "away" ? -1 : 1) - (b.ha === "away" ? -1 : 1));   // away first

    body.innerHTML = `<div class="gm-score">${sides.map((s) => `
        <div class="gm-team-line ${s.won ? "won" : ""}">
          <img src="${logoUrl(s.tid)}" alt="" onerror="this.style.opacity=0">
          <span>${s.name}</span><span class="gm-pts">${s.score}</span></div>`).join("")}</div>
      <div class="gm-date">${shortDate(own.d)}</div>
      ${chart ? `<div class="gm-section"><div class="gm-hdr">${chart.name}</div><div id="gm-chart"></div></div>` : ""}
      ${sides.map((s) => teamTable(s, highlightId)).join("")}`;
    if (chart) renderChart(ownFile.t, gameId, chart);
  }

  function isOpen() {
    return document.getElementById("gm-overlay")?.classList.contains("open");
  }

  function closeBoxScore() {
    const overlay = document.getElementById("gm-overlay");
    if (!overlay) return;
    overlay.classList.remove("open");
    const chartEl = document.getElementById("gm-chart");
    if (window.Plotly && chartEl) Plotly.purge(chartEl);
  }

  window.Games = { teamFile, openBoxScore, closeBoxScore, shortDate, opponentLabel, shootLine };
})();
