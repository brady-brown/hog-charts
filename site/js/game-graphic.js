/* game-graphic.js — the 1080×1350 (Instagram portrait) game graphic shared by
 * game-chart.html (one player) and team-game-chart.html (one team).
 *
 *   GameGraphic.draw(canvas, spec, style, isCurrent) → Promise<bool drawn>
 *   GameGraphic.save(canvas, filename)
 *   GameGraphic.STYLES  ("dark" | "wood")
 *
 * spec = {
 *   image:  {src, fit: "cover"|"contain"}, fallback: src (shown when image fails)
 *   jersey: {color, alt, number} instead of image — a jersey in the team's colors
 *           (hex, "#" optional; missing → neutral grey) with the number on it
 *   title:  big name line
 *   sub:    {logo: src|null, text}            second line (team logo + name, or conference)
 *   chips:  [text, …]
 *   game:   a Games.teamFile game ({ha, o, oid, d, w, ts, os})
 *   line:   box line with pts reb ast fgm fga tpm tpa ftm fta
 *   shots:  {mx, my, hx, hy} from shotsXY (screen feet: overhead, basket at the bottom)
 * }
 * isCurrent() is checked after the images load, so a newer pick wins the race.
 * Everything is drawn on the canvas, so the PNG is exactly what's on screen.
 * Needs common.js (logoUrl), court.js (X_RANGE, Y_RANGE, courtTraces), games.js.
 */
(function () {
  "use strict";

  const W = 1080, H = 1350, PAD = 60;
  // The site charts stop at Y_RANGE[1] (31 ft); about 4% of shots land deeper.
  // The graphic shows 5 more feet, which leaves out only heaves (~0.2%).
  const Y_TOP = Y_RANGE[1] + 5;

  const STYLES = {
    dark: { bg:"#0E1320", ink:"#F2F4F8", muted:"#9AA4B2", line:"#262F40", card:"#141925",
            floor:"#0B0E14", court:"rgba(255,255,255,0.32)", make:"#22C55E", miss:"#EF4444", ring:"#FFFFFF" },
    wood: { bg:"#0E1320", ink:"#F2F4F8", muted:"#9AA4B2", line:"#262F40", card:"#141925",
            floor:"#D2A679", court:"rgba(255,255,255,0.95)", make:"#2E7D32", miss:"#D32F2F", ring:"#FFFFFF" },
  };

  function loadImage(src) {
    if (!src) return Promise.resolve(null);
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = "anonymous";        // ESPN's CDN sends ACAO:*, so the canvas stays exportable
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // Largest font size (≤ max) at which text fits in maxW.
  function fitFont(ctx, text, weight, max, maxW) {
    let size = max;
    do { ctx.font = `${weight} ${size}px Inter`; } while (ctx.measureText(text).width > maxW && --size > 12);
    return size;
  }

  // "Cover"-crop into a box, anchored to the top (headshots).
  function drawCover(ctx, img, x, y, w, h) {
    const s = Math.max(w / img.width, h / img.height);
    const sw = w / s, sh = h / s;
    ctx.drawImage(img, (img.width - sw) / 2, 0, sw, sh, x, y, w, h);
  }

  function drawContain(ctx, img, x, y, size) {
    const s = Math.min(size / img.width, size / img.height);
    const w = img.width * s, h = img.height * s;
    ctx.drawImage(img, x + (size - w) / 2, y + (size - h) / 2, w, h);
  }

  // ── jersey (stands in for a player photo) ─────────────────────────────────
  const hex = (c) => (c ? `#${String(c).replace(/^#/, "")}` : null);
  function luminance(c) {
    const n = parseInt(c.slice(1), 16), ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
      v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  }
  const contrast = (a, b) => { const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

  // A basketball tank top filling a size×size box at (x, y): body in the
  // primary color, neck and armhole trim plus the number in the alternate
  // color — or white/black when the alternate is missing or too close to read.
  function drawJersey(ctx, x, y, size, jersey) {
    const body = hex(jersey.color) ?? "#3B4252";
    let trim = hex(jersey.alt);
    if (!trim || contrast(body, trim) < 2.5) trim = luminance(body) > 0.4 ? "#111111" : "#FFFFFF";
    const k = size / 200, X = (v) => x + v * k, Y = (v) => y + v * k;
    const neck = () => { ctx.moveTo(X(66), Y(26)); ctx.quadraticCurveTo(X(100), Y(80), X(134), Y(26)); };
    const armL = () => { ctx.moveTo(X(46), Y(26)); ctx.quadraticCurveTo(X(50), Y(82), X(24), Y(96)); };
    const armR = () => { ctx.moveTo(X(154), Y(26)); ctx.quadraticCurveTo(X(150), Y(82), X(176), Y(96)); };

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(X(46), Y(26)); ctx.lineTo(X(66), Y(26));
    ctx.quadraticCurveTo(X(100), Y(80), X(134), Y(26));
    ctx.lineTo(X(154), Y(26));
    ctx.quadraticCurveTo(X(150), Y(82), X(176), Y(96));
    ctx.lineTo(X(172), Y(184));
    ctx.quadraticCurveTo(X(100), Y(192), X(28), Y(184));
    ctx.lineTo(X(24), Y(96));
    ctx.quadraticCurveTo(X(50), Y(82), X(46), Y(26));
    ctx.closePath();
    ctx.fillStyle = body; ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.lineWidth = 2 * k; ctx.stroke();

    ctx.strokeStyle = trim; ctx.lineWidth = 7 * k; ctx.lineCap = "round";
    for (const edge of [neck, armL, armR]) { ctx.beginPath(); edge(); ctx.stroke(); }

    const number = jersey.number != null ? String(jersey.number) : "";
    if (number) {
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = trim;
      fitFont(ctx, number, 900, 74, 104 * k);
      ctx.fillText(number, X(100), Y(132));
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    }
    ctx.restore();
  }

  const fmtDate = (d) => new Date(`${d}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  const pct = (m, a) => (a ? `${Math.round((m / a) * 100)}%` : "");

  async function draw(canvas, spec, style = "dark", isCurrent = () => true) {
    const P = STYLES[style] ?? STYLES.dark;
    const { game, line, shots } = spec;
    const ctx = canvas.getContext("2d");

    await Promise.all(["900 56px Inter", "800 40px Inter", "700 26px Inter", "600 26px Inter"].map((f) => document.fonts.load(f)));
    let [image, subLogo, oppLogo, brand] = await Promise.all([
      loadImage(spec.image?.src), loadImage(spec.sub?.logo),
      game.oid ? loadImage(logoUrl(game.oid)) : null, loadImage("img/logo.png"),
    ]);
    let fit = spec.image?.fit ?? "cover";
    if (!image && spec.fallback) { image = await loadImage(spec.fallback); fit = "contain"; }
    if (!isCurrent()) return false;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = P.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.textBaseline = "alphabetic";

    // Header: picture + title / sub line / chips.
    const HS = 200;
    roundRect(ctx, PAD, PAD, HS, HS, 22);
    ctx.save(); ctx.clip();
    ctx.fillStyle = P.card; ctx.fillRect(PAD, PAD, HS, HS);
    if (spec.jersey) drawJersey(ctx, PAD + 10, PAD + 8, HS - 20, spec.jersey);
    else if (image && fit === "cover") drawCover(ctx, image, PAD - 40, PAD, HS + 80, HS);
    else if (image) drawContain(ctx, image, PAD + 24, PAD + 24, HS - 48);
    ctx.restore();

    const tx = PAD + HS + 36, textW = W - tx - PAD;
    ctx.fillStyle = P.ink;
    fitFont(ctx, spec.title, 900, 60, textW);
    ctx.fillText(spec.title, tx, PAD + 64);

    let subX = tx;
    if (subLogo) { drawContain(ctx, subLogo, tx, PAD + 88, 44); subX += 58; }
    if (spec.sub?.text) {
      ctx.fillStyle = P.muted;
      fitFont(ctx, spec.sub.text, 700, 30, textW - (subX - tx));
      ctx.fillText(spec.sub.text, subX, PAD + 121);
    }

    ctx.font = "700 24px Inter";
    let cx = tx;
    for (const chip of (spec.chips ?? []).filter(Boolean)) {
      const w = ctx.measureText(chip).width + 32;
      roundRect(ctx, cx, PAD + 146, w, 44, 22);
      ctx.fillStyle = P.card; ctx.fill();
      ctx.fillStyle = P.muted; ctx.fillText(chip, cx + 16, PAD + 176);
      cx += w + 10;
    }

    // Game strip: opponent + date, result on the right.
    const gy = PAD + HS + 36;
    roundRect(ctx, PAD, gy, W - 2 * PAD, 84, 18);
    ctx.fillStyle = P.card; ctx.fill();
    let gx = PAD + 26;
    ctx.fillStyle = P.muted; ctx.font = "700 28px Inter";
    const at = game.ha === "away" ? "@" : "vs";
    ctx.fillText(at, gx, gy + 53);
    gx += ctx.measureText(at).width + 14;
    if (oppLogo) { drawContain(ctx, oppLogo, gx, gy + 20, 44); gx += 56; }
    const result = `${game.w ? "W" : "L"} ${game.ts}–${game.os}`;
    ctx.font = "800 30px Inter";
    const resultW = ctx.measureText(result).width;
    const oppText = `${game.o ?? "—"}  ·  ${fmtDate(game.d)}`;
    ctx.fillStyle = P.ink;
    fitFont(ctx, oppText, 700, 28, W - PAD - 26 - resultW - 30 - gx);
    ctx.fillText(oppText, gx, gy + 53);
    ctx.font = "800 30px Inter";
    ctx.fillStyle = game.w ? "#16A34A" : "#DC2626";
    ctx.textAlign = "right";
    ctx.fillText(result, W - PAD - 26, gy + 54);
    ctx.textAlign = "left";

    // Stat line.
    const sy = gy + 84 + 26;
    const stats = [
      { v: line.pts ?? "—", l: "PTS" }, { v: line.reb ?? "—", l: "REB" }, { v: line.ast ?? "—", l: "AST" },
      { v: Games.shootLine(line.fgm, line.fga), l: "FG", p: pct(line.fgm, line.fga) },
      { v: Games.shootLine(line.tpm, line.tpa), l: "3P", p: pct(line.tpm, line.tpa) },
      { v: Games.shootLine(line.ftm, line.fta), l: "FT", p: pct(line.ftm, line.fta) },
    ];
    const cellW = (W - 2 * PAD) / stats.length;
    ctx.textAlign = "center";
    stats.forEach((s, i) => {
      const mid = PAD + cellW * i + cellW / 2;
      ctx.fillStyle = P.ink;
      fitFont(ctx, String(s.v), 900, 54, cellW - 14);
      ctx.fillText(String(s.v), mid, sy + 58);
      ctx.fillStyle = P.muted; ctx.font = "700 22px Inter";
      ctx.fillText(s.p ? `${s.l} · ${s.p}` : s.l, mid, sy + 94);
      if (i) { ctx.fillStyle = P.line; ctx.fillRect(PAD + cellW * i, sy + 12, 2, 86); }
    });
    ctx.textAlign = "left";

    // Court + shots. Same feet-based geometry and orientation as the site charts (court.js).
    const top = sy + 120, courtW = W - 2 * PAD;
    const scale = courtW / (X_RANGE[1] - X_RANGE[0]);
    const courtH = (Y_TOP - Y_RANGE[0]) * scale;
    const px = (x) => PAD + (x - X_RANGE[0]) * scale;
    const py = (y) => top + (Y_TOP - y) * scale;
    roundRect(ctx, PAD, top, courtW, courtH, 18);
    ctx.save(); ctx.clip();
    ctx.fillStyle = P.floor; ctx.fillRect(PAD, top, courtW, courtH);
    ctx.strokeStyle = P.court; ctx.lineWidth = 3; ctx.lineJoin = "round"; ctx.lineCap = "round";
    for (const seg of courtTraces()) {
      ctx.beginPath();
      seg.x.forEach((x, i) => (i ? ctx.lineTo(px(x), py(seg.y[i])) : ctx.moveTo(px(x), py(seg.y[i]))));
      ctx.stroke();
    }
    for (const x of [-25, 25]) {                        // sidelines out to the deeper edge
      ctx.beginPath(); ctx.moveTo(px(x), py(Y_RANGE[1])); ctx.lineTo(px(x), py(Y_TOP)); ctx.stroke();
    }
    // A whole team's game is ~60 attempts, so its markers run a little smaller.
    const R = shots.mx.length + shots.hx.length > 30 ? 11 : 13;
    const cross = (x, y) => {
      ctx.beginPath(); ctx.moveTo(x - R, y - R); ctx.lineTo(x + R, y + R); ctx.moveTo(x + R, y - R); ctx.lineTo(x - R, y + R); ctx.stroke();
    };
    for (let i = 0; i < shots.hx.length; i++) {           // misses under makes
      const x = px(shots.hx[i]), y = py(shots.hy[i]);
      ctx.strokeStyle = P.ring; ctx.lineWidth = 8; cross(x, y);
      ctx.strokeStyle = P.miss; ctx.lineWidth = 5; cross(x, y);
    }
    for (let i = 0; i < shots.mx.length; i++) {
      ctx.beginPath(); ctx.arc(px(shots.mx[i]), py(shots.my[i]), R + 1, 0, 2 * Math.PI);
      ctx.fillStyle = P.make; ctx.fill();
      ctx.strokeStyle = P.ring; ctx.lineWidth = 3; ctx.stroke();
    }
    ctx.restore();

    // Legend + brand footer.
    const fy = H - PAD + 6;
    ctx.font = "700 26px Inter";
    let lx = PAD + 4;
    ctx.beginPath(); ctx.arc(lx + 12, fy - 9, 12, 0, 2 * Math.PI);
    ctx.fillStyle = P.make; ctx.fill();
    ctx.fillStyle = P.muted; ctx.fillText(`Make (${shots.mx.length})`, lx + 34, fy);
    lx += 34 + ctx.measureText(`Make (${shots.mx.length})`).width + 30;
    ctx.strokeStyle = P.miss; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(lx + 2, fy - 19); ctx.lineTo(lx + 22, fy + 1); ctx.moveTo(lx + 22, fy - 19); ctx.lineTo(lx + 2, fy + 1); ctx.stroke();
    ctx.fillStyle = P.muted; ctx.fillText(`Miss (${shots.hx.length})`, lx + 34, fy);

    ctx.textAlign = "right";
    ctx.fillStyle = P.ink; ctx.font = "800 28px Inter";
    const handle = "@hogcharts";
    ctx.fillText(handle, W - PAD, fy);
    const handleW = ctx.measureText(handle).width;
    if (brand) drawContain(ctx, brand, W - PAD - handleW - 56, fy - 36, 46);
    ctx.textAlign = "left";
    return true;
  }

  async function save(canvas, filename) {
    const blob = await new Promise((r) => canvas.toBlob(r, "image/png"));
    if (!blob) return;
    const file = new File([blob], filename, { type: "image/png" });
    // Phones: the share sheet (Save Image / Instagram). Desktop: a plain download.
    if (navigator.canShare?.({ files: [file] }) && matchMedia("(pointer:coarse)").matches) {
      try { await navigator.share({ files: [file] }); return; } catch (e) { if (e.name === "AbortError") return; }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  window.GameGraphic = { draw, save, STYLES };
})();
