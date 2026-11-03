/* player-card.js — shared player-card modal used by Player Stats and Scout.
 *
 * Self-contained: injects its own CSS + modal markup on load and exposes a tiny
 * API. Both pages hand it a fully-formed player row (the same shape used on the
 * Player Stats table) so the card renders identically wherever it is opened.
 *
 *   PlayerCard.open(row, {season, curSeason, hasOnOff})
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
  .pc-career { border-top:1px solid var(--line); padding:20px 28px 24px; }
  .pc-career-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
  .pc-career-switch { display:flex; gap:6px; }
  .pc-career-switch button { font:inherit; font-size:.72rem; font-weight:700; padding:3px 10px;
    border-radius:20px; border:1px solid var(--line); background:var(--card2); color:var(--muted); cursor:pointer; }
  .pc-career-switch button.active { background:rgba(250,100,50,.15); color:var(--t1); border-color:transparent; }
  .pc-career-wrap { overflow-x:auto; }
  .pc-career-tbl { width:100%; border-collapse:collapse; font-size:.8rem; font-variant-numeric:tabular-nums; }
  .pc-career-tbl th { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
    color:var(--muted); text-align:right; padding:0 6px 6px; white-space:nowrap; }
  .pc-career-tbl td { text-align:right; padding:6px; border-top:1px solid #1a2235; white-space:nowrap; }
  .pc-career-tbl th:nth-child(-n+2), .pc-career-tbl td:nth-child(-n+2) { text-align:left; }
  .pc-career-tbl tr.current td { font-weight:800; }
  .pc-career-school { display:flex; align-items:center; gap:6px; max-width:180px; }
  .pc-career-school img { width:16px; height:16px; object-fit:contain; flex-shrink:0; }
  .pc-career-school span { overflow:hidden; text-overflow:ellipsis; }
  .pc-tabs { display:flex; gap:4px; padding:12px 28px 0; border-bottom:1px solid var(--line); }
  .pc-tabs button { font:inherit; font-size:.85rem; font-weight:700; padding:8px 14px; background:none;
    border:none; border-bottom:2px solid transparent; color:var(--muted); cursor:pointer; margin-bottom:-1px; }
  .pc-tabs button.active { color:var(--ink); border-bottom-color:var(--t1); }
  .pc-tabs button[hidden] { display:none; }
  .pc-tab-career .pc-career { border-top:none; }
  .pc-games { padding:20px 28px 24px; }
  .pc-games-status { color:var(--muted); font-size:.85rem; }
  .pc-career-tbl tr.pc-game { cursor:pointer; }
  .pc-career-tbl tr.pc-game:hover td { background:var(--card2); }
  .pc-win { color:#16A34A; font-weight:700; }
  .pc-loss { color:#DC2626; font-weight:700; }
  .pc-box-overlay { display:none; position:fixed; inset:0; z-index:210; background:rgba(0,0,0,.6);
    align-items:center; justify-content:center; }
  .pc-box-overlay.open { display:flex; }
  .pc-box-modal { background:var(--card); border:1px solid var(--line); border-radius:20px;
    width:min(860px,95vw); max-height:90vh; overflow-y:auto; position:relative; padding:24px 28px; }
  .pc-box-score { display:flex; flex-direction:column; gap:6px; margin-bottom:4px; padding-right:40px; }
  .pc-box-team-line { display:flex; align-items:center; gap:10px; font-size:1.05rem; font-weight:700; color:var(--muted); }
  .pc-box-team-line.won { color:var(--ink); font-weight:800; }
  .pc-box-team-line img { width:24px; height:24px; object-fit:contain; }
  .pc-box-team-line .pc-box-pts { margin-left:auto; font-variant-numeric:tabular-nums; }
  .pc-box-date { color:var(--muted); font-size:.8rem; margin-bottom:18px; }
  .pc-box-team { margin-top:18px; }
  .pc-career-tbl tr.pc-box-player td { font-weight:800; color:var(--t1); }
  .pc-career-tbl tr.pc-box-totals td { font-weight:800; border-top:2px solid var(--line); }
  .pc-career-tbl td.pc-dnp { color:var(--muted); text-align:left; }
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
      <div class="pc-tabs" id="pc-tabs">
        <button data-tab="season" class="active">Season</button>
        <button data-tab="career">Career</button>
        <button data-tab="games">Games</button>
      </div>
      <div class="pc-tab-season" id="pc-tab-season">
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
      </div>
      <div class="pc-tab-career" id="pc-tab-career" hidden>
      <div class="pc-career" id="pc-career" style="display:none">
        <div class="pc-career-head">
          <div></div>
          <div class="pc-career-switch" id="pc-career-switch">
            <button data-view="box" class="active">Box Score</button>
            <button data-view="adv">Advanced</button>
          </div>
        </div>
        <div class="pc-career-wrap"><table class="pc-career-tbl" id="pc-career-tbl"></table></div>
      </div>
      </div>
      <div class="pc-games" id="pc-tab-games" hidden>
        <div class="pc-games-status" id="pc-games-status"></div>
        <div class="pc-career-wrap"><table class="pc-career-tbl" id="pc-games-tbl"></table></div>
      </div>
      <div class="pc-tab-season-charts" id="pc-tab-season-charts">
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
    </div>
  </div>
  <div class="pc-box-overlay" id="pc-box-overlay">
    <div class="pc-box-modal">
      <div class="pc-close" id="pc-box-close">✕</div>
      <div id="pc-box-body"></div>
    </div>
  </div>`;

  function injectOnce() {
    if (document.getElementById("pc-overlay")) return;
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    const holder = document.createElement("div");
    holder.innerHTML = HTML;
    while (holder.firstElementChild) document.body.appendChild(holder.firstElementChild);
    document.getElementById("pc-close-btn").addEventListener("click", close);
    document.getElementById("pc-overlay").addEventListener("click", (e) => {
      if (e.target === document.getElementById("pc-overlay")) close();
    });
    document.getElementById("pc-box-close").addEventListener("click", closeBoxScore);
    document.getElementById("pc-box-overlay").addEventListener("click", (e) => {
      if (e.target === document.getElementById("pc-box-overlay")) closeBoxScore();
    });
    // Escape closes the box score first, then the card.
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (document.getElementById("pc-box-overlay").classList.contains("open")) closeBoxScore();
      else close();
    });
    document.getElementById("pc-tabs").addEventListener("click", (e) => {
      if (e.target.dataset?.tab) showTab(e.target.dataset.tab);
    });
    document.getElementById("pc-games-tbl").addEventListener("click", (e) => {
      const gameId = e.target.closest("tr.pc-game")?.dataset.game;
      if (gameId) openBoxScore(Number(gameId));
    });
    document.getElementById("pc-career-switch").addEventListener("click", (e) => {
      const view = e.target.dataset?.view;
      if (!view || view === careerView) return;
      careerView = view;
      renderCareer();
    });
  }

  // ── career table ──────────────────────────────────────────────────────────
  // Every season the player appears in, with the school he played for that year.
  // careers/{id % CAREER_SHARDS}.json comes from hoglib/careers.py.
  const CAREER_SHARDS = 64;   // MUST match CAREER_SHARDS in hoglib/careers.py
  const pct  = (v) => v != null ? (v * 100).toFixed(1) : "—";
  const dec1 = (v) => v != null ? v.toFixed(1) : "—";
  const CAREER_COLS = {
    box: [["GP", (r) => r.gp ?? "—"], ["MPG", (r) => dec1(r.mpg)], ["PPG", (r) => dec1(r.ppg)],
          ["RPG", (r) => dec1(r.rpg)], ["APG", (r) => dec1(r.apg)], ["FG%", (r) => pct(r.fg)],
          ["3P%", (r) => pct(r.fg3)], ["FT%", (r) => pct(r.ft)], ["TS%", (r) => pct(r.ts)]],
    adv: [["USG%", (r) => dec1(r.usg)], ["AST%", (r) => dec1(r.astp)], ["TOV%", (r) => dec1(r.tovp)],
          ["TRB%", (r) => dec1(r.trbp)], ["STL%", (r) => dec1(r.stlp)], ["BLK%", (r) => dec1(r.blkp)],
          ["TS%", (r) => pct(r.ts)], ["eFG%", (r) => pct(r.efg)]],
  };
  const CAREER_SHARD_CACHE = {};
  let careerView = "box";
  let career = null;         // {rows, season} for the card on screen
  let careerRequest = 0;     // ignores a slow fetch for a card that's already been replaced

  async function loadCareer(playerId, season) {
    const request = ++careerRequest;
    career = null;
    document.getElementById("pc-career").style.display = "none";
    if (playerId == null) return;
    const shardIndex = Number(playerId) % CAREER_SHARDS;
    if (!(shardIndex in CAREER_SHARD_CACHE)) {
      CAREER_SHARD_CACHE[shardIndex] = await loadJSON(`data/careers/${shardIndex}.json`);
    }
    const rows = CAREER_SHARD_CACHE[shardIndex]?.[String(playerId)];
    if (request !== careerRequest || !rows?.length) return;
    career = { rows: [...rows].sort((a, b) => a.s - b.s), season };
    renderCareer();
  }

  function renderCareer() {
    if (!career) return;
    document.querySelectorAll("#pc-career-switch button").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === careerView));
    const cols = CAREER_COLS[careerView];
    const head = `<tr><th>Season</th><th>School</th>${cols.map(([lbl]) => `<th>${lbl}</th>`).join("")}</tr>`;
    const body = career.rows.map((r) => `
      <tr class="${r.s === career.season ? "current" : ""}">
        <td>${seasonLabel(r.s)}</td>
        <td><div class="pc-career-school" title="${r.t ?? ""}">
          <img src="${logoUrl(r.tid)}" alt="" onerror="this.style.opacity=0"><span>${r.t ?? "—"}</span></div></td>
        ${cols.map(([, fmt]) => `<td>${fmt(r)}</td>`).join("")}
      </tr>`).join("");
    document.getElementById("pc-career-tbl").innerHTML = head + body;
    document.getElementById("pc-career").style.display = "block";
  }

  // ── tabs ──────────────────────────────────────────────────────────────────
  function showTab(tab) {
    document.querySelectorAll("#pc-tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === tab));
    document.getElementById("pc-tab-season").hidden = tab !== "season";
    document.getElementById("pc-tab-season-charts").hidden = tab !== "season";
    document.getElementById("pc-tab-career").hidden = tab !== "career";
    document.getElementById("pc-tab-games").hidden = tab !== "games";
    if (tab === "games") loadGames();
  }

  // ── games: game log + box scores ──────────────────────────────────────────
  // data/{season}/games/teams/{team_id}.json (hoglib/gamelogs.py) holds every game
  // a team played with every player's line, so the log is this player's lines
  // and a box score is his team's game plus the same game in the opponent's file.
  const TEAM_GAMES_CACHE = {};
  let gamesFor = null;       // {playerId, teamId, season} for the card on screen
  let gamesRequest = 0;

  async function teamGames(season, teamId) {
    const key = `${season}/${teamId}`;
    if (!(key in TEAM_GAMES_CACHE)) {
      TEAM_GAMES_CACHE[key] = await loadJSON(`data/${season}/games/teams/${teamId}.json`);
    }
    return TEAM_GAMES_CACHE[key];
  }

  const shootLine = (made, att) => (att ? `${made}-${att}` : "—");
  const shortDate = (d) => new Date(`${d}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const opponentLabel = (g) => `${g.ha === "away" ? "@" : "vs"} ${g.o ?? "—"}`;

  async function loadGames() {
    if (!gamesFor) return;
    const { playerId, teamId, season } = gamesFor;
    const request = ++gamesRequest;
    const status = document.getElementById("pc-games-status");
    const table = document.getElementById("pc-games-tbl");
    status.textContent = "Loading…";
    table.innerHTML = "";
    const team = await teamGames(season, teamId);
    if (request !== gamesRequest) return;
    const games = (team?.games ?? [])
      .map((g) => ({ g, line: g.players.find((p) => p.id === Number(playerId) && !p.dnp) }))
      .filter((x) => x.line)
      .reverse();                                   // newest first
    if (!games.length) { status.textContent = "No games yet."; return; }
    status.textContent = "";
    const cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG", "3P", "FT"];
    table.innerHTML = `<tr><th>Date</th><th>Opp</th><th>Result</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>` +
      games.map(({ g, line: l }) => `
        <tr class="pc-game" data-game="${g.g}">
          <td>${shortDate(g.d)}</td><td>${opponentLabel(g)}</td>
          <td><span class="${g.w ? "pc-win" : "pc-loss"}">${g.w ? "W" : "L"}</span> ${g.ts}–${g.os}</td>
          <td>${l.min ?? "—"}</td><td>${l.pts ?? "—"}</td><td>${l.reb ?? "—"}</td><td>${l.ast ?? "—"}</td>
          <td>${l.stl ?? "—"}</td><td>${l.blk ?? "—"}</td><td>${l.tov ?? "—"}</td>
          <td>${shootLine(l.fgm, l.fga)}</td><td>${shootLine(l.tpm, l.tpa)}</td><td>${shootLine(l.ftm, l.fta)}</td>
        </tr>`).join("");
  }

  function boxTeamTable(side, highlightId) {
    const cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF", "FG", "3P", "FT"];
    const statCells = (s) => `<td>${s.pts ?? "—"}</td><td>${s.reb ?? "—"}</td><td>${s.ast ?? "—"}</td>
      <td>${s.stl ?? "—"}</td><td>${s.blk ?? "—"}</td><td>${s.tov ?? "—"}</td><td>${s.pf ?? "—"}</td>
      <td>${shootLine(s.fgm, s.fga)}</td><td>${shootLine(s.tpm, s.tpa)}</td><td>${shootLine(s.ftm, s.fta)}</td>`;
    const rows = side.game.players.map((p) => p.dnp
      ? `<tr><td>${p.n}</td><td>${p.pos ?? ""}</td><td class="pc-dnp" colspan="${cols.length}">DNP</td></tr>`
      : `<tr class="${p.id === highlightId ? "pc-box-player" : ""}"><td>${p.n}</td><td>${p.pos ?? ""}</td>
           <td>${p.min ?? "—"}</td>${statCells(p)}</tr>`).join("");
    return `<div class="pc-box-team">
      <div class="pc-col-hdr">${side.name}</div>
      <div class="pc-career-wrap"><table class="pc-career-tbl">
        <tr><th>Player</th><th>Pos</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>
        ${rows}
        <tr class="pc-box-totals"><td>Totals</td><td></td><td></td>${statCells(side.game.totals)}</tr>
      </table></div></div>`;
  }

  async function openBoxScore(gameId) {
    if (!gamesFor) return;
    const { playerId, teamId, season } = gamesFor;
    const body = document.getElementById("pc-box-body");
    body.innerHTML = `<div class="pc-games-status">Loading…</div>`;
    document.getElementById("pc-box-overlay").classList.add("open");
    const own = (await teamGames(season, teamId))?.games.find((g) => g.g === gameId);
    if (!own) { body.innerHTML = `<div class="pc-games-status">Box score not available.</div>`; return; }
    const oppFile = await teamGames(season, own.oid);
    const opp = oppFile?.games.find((g) => g.g === gameId);
    const sides = [{ name: (await teamGames(season, teamId)).t, tid: teamId, game: own, score: own.ts, won: own.w, ha: own.ha }];
    if (opp) sides.push({ name: oppFile.t, tid: own.oid, game: opp, score: opp.ts, won: opp.w, ha: opp.ha });
    sides.sort((a, b) => (a.ha === "away" ? -1 : 1) - (b.ha === "away" ? -1 : 1));   // away first
    const scoreLines = sides.map((s) => `
      <div class="pc-box-team-line ${s.won ? "won" : ""}">
        <img src="${logoUrl(s.tid)}" alt="" onerror="this.style.opacity=0">
        <span>${s.name}</span><span class="pc-box-pts">${s.score}</span></div>`).join("");
    body.innerHTML = `<div class="pc-box-score">${scoreLines}</div>
      <div class="pc-box-date">${shortDate(own.d)}</div>
      ${sides.map((s) => boxTeamTable(s, Number(playerId))).join("")}`;
  }

  function closeBoxScore() {
    document.getElementById("pc-box-overlay")?.classList.remove("open");
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
    const { season, curSeason, hasOnOff = false } = opts;
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
    loadCareer(p.id, season ?? curSeason);

    // Game logs exist for the live season only.
    const isLiveSeason = season == null || curSeason == null || season === curSeason;
    gamesFor = isLiveSeason && p.tid != null ? { playerId: p.id, teamId: p.tid, season: season ?? curSeason } : null;
    document.querySelector('#pc-tabs [data-tab="games"]').hidden = !gamesFor;
    showTab("season");

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
  }

  function close() {
    const overlay = document.getElementById("pc-overlay");
    if (!overlay) return;
    closeBoxScore();
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    const z = document.getElementById("pc-chart-zone");
    const d = document.getElementById("pc-chart-density");
    if (window.Plotly && d) Plotly.purge(d);
    if (z) z.innerHTML = "";
  }

  window.PlayerCard = { open, close };
})();
