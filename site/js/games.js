/* games.js — shared game data + box-score modal for the player and team cards.
 *
 *   Games.teamFile(season, teamId)   → {t, colors, games:[{g, d, ha, o, oid, ts, os, w, players, totals}]}
 *   Games.openBoxScore({season, teamId, gameId, highlightId, chart})
 *   Games.renderBoxScore(el, {season, teamId, gameId, highlightId, live, onPlayer})  — inline, no modal
 *   Games.shortDate(d) · Games.opponentLabel(game) · Games.shootLine(made, att)
 *
 * Reads data/{season}/games/teams/{team_id}.json from hoglib/gamelogs.py, which
 * stores names once per file and each line as an array in "cols" order; teamFile
 * hands back plain objects. A box score is the team's game plus the same game in
 * the opponent's file.
 *
 * chart ({kind: "player"|"team", name}) adds that game's makes and misses from
 * data/shots/{team}.json — live season only, so callers pass it only then.
 * Needs common.js (loadJSON, logoUrl, slugify, ensurePlotly, the *Link/*Href helpers) and court.js.
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
    const who = (id) => ({ id, n: raw.roster[id]?.[0] ?? "—", pos: raw.roster[id]?.[1] ?? "", jn: raw.roster[id]?.[2] ?? null });
    return {
      t: raw.t,
      colors: raw.c ?? [null, null],          // [primary, alternate] hex without "#"
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
  // Names link to the player and team pages. A player's row opens his shot chart
  // for this game (live season) or his page (past seasons); ctx.onPlayer, when
  // given, takes the row/chart click instead (the game chart page switches in place).
  const BOX_COLS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF", "FG", "3P", "FT"];
  const statCells = (s) => `<td>${v(s.pts)}</td><td>${v(s.reb)}</td><td>${v(s.ast)}</td><td>${v(s.stl)}</td>
    <td>${v(s.blk)}</td><td>${v(s.tov)}</td><td>${v(s.pf)}</td>
    <td>${shootLine(s.fgm, s.fga)}</td><td>${shootLine(s.tpm, s.tpa)}</td><td>${shootLine(s.ftm, s.fta)}</td>`;

  function teamTable(side, ctx) {
    const { season, live, gameId, highlightId } = ctx;
    const chartCol = live ? 1 : 0;
    const rows = side.game.players.map((p) => p.dnp
      ? `<tr><td>${playerLink(p.id, p.n, season)}</td><td>${p.pos}</td><td class="gm-dnp" colspan="${BOX_COLS.length + chartCol}">DNP</td></tr>`
      : `<tr class="gm-row ${p.id === highlightId ? "gm-highlight" : ""}" data-pid="${p.id}" data-tid="${side.tid}">
           <td>${playerLink(p.id, p.n, season)}</td><td>${p.pos}</td>
           <td>${v(p.min)}</td>${statCells(p)}${live ? `<td>${chartLink(gameChartHref(p.id, gameId))}</td>` : ""}</tr>`).join("");
    const teamChart = live ? ` ${chartLink(teamGameChartHref(side.tid, gameId), "Team shot chart")}` : "";
    return `<div class="gm-section"><div class="gm-hdr">${teamLink(side.tid, side.name, season)}${teamChart}</div>
      <div class="gm-wrap"><table class="gm-tbl">
        <tr><th>Player</th><th>Pos</th>${BOX_COLS.map((c) => `<th>${c}</th>`).join("")}${live ? "<th></th>" : ""}</tr>
        ${rows}
        <tr class="gm-totals"><td>Totals</td><td></td><td></td>${statCells(side.game.totals)}${live ? "<td></td>" : ""}</tr>
      </table></div></div>`;
  }

  // Both teams' lines for one game: the team's own file + the same game in the opponent's.
  async function loadSides(season, teamId, gameId) {
    const ownFile = await teamFile(season, teamId);
    const own = ownFile?.games.find((g) => g.g === gameId);
    if (!own) return null;
    const oppFile = await teamFile(season, own.oid);
    const opp = oppFile?.games.find((g) => g.g === gameId);
    const sides = [{ name: ownFile.t, tid: teamId, game: own, score: own.ts, won: own.w, ha: own.ha }];
    if (opp) sides.push({ name: oppFile.t, tid: own.oid, game: opp, score: opp.ts, won: opp.w, ha: opp.ha });
    else sides.push({ name: own.o, tid: own.oid, game: null, score: own.os, won: !own.w, ha: own.ha === "away" ? "home" : "away" });
    sides.sort((a, b) => (a.ha === "away" ? -1 : 1) - (b.ha === "away" ? -1 : 1));   // away first
    return { ownFile, own, sides };
  }

  function scoreHTML(sides, own, season) {
    return `<div class="gm-score">${sides.map((s) => `
        <div class="gm-team-line ${s.won ? "won" : ""}">
          <img src="${logoUrl(s.tid)}" alt="" onerror="this.style.opacity=0">
          <span>${teamLink(s.tid, s.name, season)}</span><span class="gm-pts">${s.score}</span></div>`).join("")}</div>
      <div class="gm-date">${shortDate(own.d)}</div>`;
  }

  const tablesHTML = (sides, ctx) => sides.filter((s) => s.game).map((s) => teamTable(s, ctx)).join("");

  // Row + chart-icon clicks. Capture phase, so it runs before the icon's own link.
  function wireRows(el, ctx) {
    if (el._gmRows) el.removeEventListener("click", el._gmRows, true);
    el._gmRows = (e) => {
      if (e.target.closest("a.team-link")) return;              // a name: follow its link
      const row = e.target.closest("tr.gm-row[data-pid]");
      if (!row) return;
      const pid = Number(row.dataset.pid), tid = Number(row.dataset.tid);
      if (ctx.onPlayer && ctx.live) {
        e.preventDefault(); e.stopPropagation();
        ctx.onPlayer(pid, tid, ctx.gameId);
      } else if (!e.target.closest("a")) {
        goTo(ctx.live ? gameChartHref(pid, ctx.gameId) : playerHref(pid, ctx.season));
      }
    };
    el.addEventListener("click", el._gmRows, true);
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

    const loaded = await loadSides(season, teamId, gameId);
    if (request !== boxRequest) return;
    if (!loaded) { body.innerHTML = `<div class="gm-status">Box score not available.</div>`; return; }
    const { ownFile, own, sides } = loaded;
    const ctx = { season, live: !!chart, gameId, highlightId };   // callers pass chart for the live season only

    body.innerHTML = scoreHTML(sides, own, season) +
      (chart ? `<div class="gm-section"><div class="gm-hdr"><a class="team-link" href="${chart.kind === "player" && highlightId != null
          ? gameChartHref(highlightId, gameId)
          : teamGameChartHref(teamId, gameId)}">${escHtml(chart.name)} <span class="team-link-arrow">→</span></a></div><div id="gm-chart"></div></div>` : "") +
      tablesHTML(sides, ctx);
    wireRows(body, ctx);
    if (chart) renderChart(ownFile.t, gameId, chart);
  }

  // The same box score drawn into a page (the game chart pages) instead of the modal.
  // onPlayer(playerId, teamId, gameId) takes a player's row/chart click.
  async function renderBoxScore(el, { season, teamId, gameId, highlightId = null, live = false, onPlayer = null }) {
    const request = (el._gmRequest = (el._gmRequest ?? 0) + 1);
    const loaded = await loadSides(season, teamId, gameId);
    if (request !== el._gmRequest) return;
    if (!loaded) { el.innerHTML = ""; return; }
    const ctx = { season, live, gameId, highlightId, onPlayer };
    el.innerHTML = scoreHTML(loaded.sides, loaded.own, season) + tablesHTML(loaded.sides, ctx);
    wireRows(el, ctx);
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

  window.Games = { teamFile, openBoxScore, renderBoxScore, closeBoxScore, shortDate, opponentLabel, shootLine };
})();
