/* player-card.js — shared player-card modal used by Player Stats and Scout.
 *
 * Self-contained: injects its own CSS + modal markup on load and exposes a tiny
 * API. Both pages hand it a fully-formed player row (the same shape used on the
 * Player Stats table) so the card renders identically wherever it is opened.
 *
 *   PlayerCard.open(row, {season, curSeason, hasOnOff})
 *   PlayerCard.open(row, {..., inline: containerEl, tab, onTab})   — player page: render
 *     into containerEl instead of a modal, start on `tab`, report tab changes
 *   PlayerCard.close()
 *
 * Dependencies (already loaded on both pages): common.js (loadJSON, logoUrl,
 * playerUrl), zone-chart.js (ZoneChart), court.js, games.js (Games tab + box
 * scores). Plotly is optional — the shot-density chart is skipped gracefully if
 * window.Plotly is absent.
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
  /* Inline: the card is the player page's content (opts.inline), not a modal. */
  .pc-inline .pc-overlay { position:static; display:block; background:none; backdrop-filter:none; z-index:auto; }
  .pc-inline .pc-modal { width:100%; max-height:none; overflow:visible; }
  .pc-inline .pc-close { display:none; }
  .pc-inline .pc-header { position:static; }
  .pc-tabs { display:flex; gap:4px; padding:12px 28px 0; border-bottom:1px solid var(--line); }
  .pc-tabs button { font:inherit; font-size:.85rem; font-weight:700; padding:8px 14px; background:none;
    border:none; border-bottom:2px solid transparent; color:var(--muted); cursor:pointer; margin-bottom:-1px; }
  .pc-tabs button.active { color:var(--ink); border-bottom-color:var(--t1); }
  .pc-tabs button[hidden] { display:none; }
  .pc-tab-career .pc-career { border-top:none; }
  .pc-games { padding:20px 28px 24px; }
  .pc-games-status { color:var(--muted); font-size:.85rem; }
  #pc-games-tbl { font-size:.76rem; }
  #pc-games-tbl th, #pc-games-tbl td { padding-left:4px; padding-right:4px; }
  .pc-career-tbl tr.pc-game { cursor:pointer; }
  .pc-career-tbl tr.pc-game:hover td { background:var(--card2); }
  .pc-win { color:#16A34A; font-weight:700; }
  .pc-loss { color:#DC2626; font-weight:700; }
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
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });   // games.js closes its box score first
    document.getElementById("pc-tabs").addEventListener("click", (e) => {
      if (e.target.dataset?.tab) showTab(e.target.dataset.tab);
    });
    document.getElementById("pc-games-tbl").addEventListener("click", (e) => {
      const gameId = e.target.closest("tr.pc-game")?.dataset.game;
      if (gameId && gamesFor) Games.openBoxScore({
        season: gamesFor.season, teamId: gamesFor.teamId, gameId: Number(gameId), highlightId: Number(gamesFor.playerId),
        chart: gamesFor.live ? { kind: "player", name: gamesFor.name } : null,   // shot locations: live season only
      });
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
  let onTabChange = null;     // player page: keeps ?tab= in the URL
  let inline = false;

  function showTab(tab) {
    onTabChange?.(tab);
    document.querySelectorAll("#pc-tabs button").forEach((b) =>
      b.classList.toggle("active", b.dataset.tab === tab));
    document.getElementById("pc-tab-season").hidden = tab !== "season";
    document.getElementById("pc-tab-season-charts").hidden = tab !== "season";
    document.getElementById("pc-tab-career").hidden = tab !== "career";
    document.getElementById("pc-tab-games").hidden = tab !== "games";
    if (tab === "games") loadGames();
  }

  // ── games: game log ───────────────────────────────────────────────────────
  // His lines from his team's games file (see games.js); a row opens the box score.
  let gamesFor = null;       // {playerId, teamId, season, live, name} for the card on screen
  let gamesRequest = 0;

  async function loadGames() {
    if (!gamesFor) return;
    const { playerId, teamId, season } = gamesFor;
    const request = ++gamesRequest;
    const status = document.getElementById("pc-games-status");
    const table = document.getElementById("pc-games-tbl");
    status.textContent = "Loading…";
    table.innerHTML = "";
    const team = await Games.teamFile(season, teamId);
    if (request !== gamesRequest) return;
    const games = (team?.games ?? [])
      .map((g) => ({ g, line: g.players.find((p) => p.id === Number(playerId) && !p.dnp) }))
      .filter((x) => x.line)
      .reverse();                                   // newest first
    if (!games.length) { status.textContent = "No games yet."; return; }
    status.textContent = "";
    const { shortDate, opponentLabel, shootLine } = Games;
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
    // Embedded in the team page: a modal inside the frame would sit mid-way down
    // a tall iframe, so the outer page shows the card.
    if (window.parent !== window && window.parent.PlayerCard) return window.parent.PlayerCard.open(p, opts);
    injectOnce();
    const { season, curSeason, hasOnOff = false } = opts;
    const pg = posGroup(p.pos);
    inline = !!opts.inline;
    onTabChange = opts.onTab ?? null;
    if (inline) {
      document.body.classList.add("pc-inline");
      opts.inline.appendChild(document.getElementById("pc-overlay"));
    }

    document.getElementById("pc-img").src = playerUrl(p.id);
    if (inline) {
      document.getElementById("pc-name").textContent = p.n ?? "—";
    } else {
      const playerLink = document.createElement("a");    // to the player page
      playerLink.className = "team-link";
      playerLink.href = `player.html?id=${p.id}&season=${season ?? curSeason ?? ""}`;
      playerLink.textContent = p.n ?? "—";
      playerLink.insertAdjacentHTML("beforeend", ' <span class="team-link-arrow">→</span>');
      document.getElementById("pc-name").replaceChildren(playerLink);
    }
    document.getElementById("pc-logo").src = logoUrl(p.tid);
    const teamLink = document.createElement("a");     // to the team page
    teamLink.className = "team-link";
    teamLink.href = `team.html?id=${p.tid}&season=${season ?? curSeason ?? ""}`;
    teamLink.textContent = p.t ?? "—";
    document.getElementById("pc-team").replaceChildren(teamLink);
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

    const cardSeason = season ?? curSeason;
    gamesFor = p.tid != null && cardSeason != null
      ? { playerId: p.id, teamId: p.tid, season: cardSeason, live: cardSeason === curSeason, name: p.n } : null;
    document.querySelector('#pc-tabs [data-tab="games"]').hidden = !gamesFor;
    showTab(opts.tab && (opts.tab !== "games" || gamesFor) ? opts.tab : "season");

    document.getElementById("pc-overlay").classList.add("open");
    if (!inline) document.body.style.overflow = "hidden";

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
    if (!overlay || inline) return;      // the player page's card never closes
    window.Games?.closeBoxScore();
    overlay.classList.remove("open");
    document.body.style.overflow = "";
    const z = document.getElementById("pc-chart-zone");
    const d = document.getElementById("pc-chart-density");
    if (window.Plotly && d) Plotly.purge(d);
    if (z) z.innerHTML = "";
  }

  window.PlayerCard = { open, close };
})();
