// court.js — shared half-court geometry + shot-density trace builder.
//
// Used by shot-charts.html, net-ratings.html and js/player-card.js (loaded
// before each). Only per-page concerns stay on the page: Plotly layout objects
// (chart height / title / static-vs-interactive) and the scatter renderer.
//
// ⚠️ The geometry MUST stay identical to build_site.classify_shot_zones (the
// Python that assigns shots to zones) and to the mirror in js/zone-chart.js.
// Coordinates are in feet, hoop-relative; the raw shot stream packs 5 numbers
// per shot: [zoneIdx, x*10, y*10, made(0/1), playerIndex].

const HOOP_Y   = 5.25;
const THREE_R  = 22.146;
const CORNER_X = 21.65;
const X_RANGE  = [-26, 26];
const Y_RANGE  = [-2.5, 31];

// Points along an arc (degrees t0→t1) as a Plotly {x, y} line segment.
function arcPts(cx, cy, r, t0, t1, n = 80) {
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const t = (t0 + (t1 - t0) * i / (n - 1)) * Math.PI / 180;
    xs.push(cx + r * Math.cos(t));
    ys.push(cy + r * Math.sin(t));
  }
  return { x: xs, y: ys };
}

// The court outline as an array of Plotly line traces.
function courtTraces(lineColor = "#3C4456") {
  const S = [];
  const ln  = (x0, y0, x1, y1) => S.push({ x: [x0, x1], y: [y0, y1] });
  const arc = (cx, cy, r, t0, t1) => S.push(arcPts(cx, cy, r, t0, t1));
  ln(-25, 0, 25, 0); ln(-25, 0, -25, 31); ln(25, 0, 25, 31);
  ln(-3, 4, 3, 4); arc(0, HOOP_Y, 0.75, 0, 360); ln(0, 4, 0, 4.5);
  arc(0, HOOP_Y, 4, 0, 180);
  ln(-6, 0, -6, 19); ln(6, 0, 6, 19); ln(-6, 19, 6, 19);
  arc(0, 19, 6, 0, 180); arc(0, 19, 6, 180, 360);
  const junc = HOOP_Y + Math.sqrt(THREE_R ** 2 - CORNER_X ** 2);
  const ang  = Math.atan2(junc - HOOP_Y, CORNER_X) * 180 / Math.PI;
  ln(-CORNER_X, 0, -CORNER_X, junc); ln(CORNER_X, 0, CORNER_X, junc);
  arc(0, HOOP_Y, THREE_R, ang, 180 - ang);
  return S.map(s => ({
    type: "scatter", x: s.x, y: s.y, mode: "lines",
    line: { color: lineColor, width: 1.6 }, hoverinfo: "skip", showlegend: false,
  }));
}

// Split a raw shot stream into make/miss x,y arrays (feet). filterFn(zoneIdx,
// playerIndex) selects which shots to include — defaults to all. This subsumes
// the three former variants: whole-team (no filter), single player (by index),
// and shot-charts' zone/player filter function.
function shotsXY(shots, filterFn = () => true) {
  const mx = [], my = [], hx = [], hy = [];
  for (let i = 0; i < shots.length; i += 5) {
    if (!filterFn(shots[i], shots[i + 4])) continue;
    const px = shots[i + 1] / 10, py = shots[i + 2] / 10;
    if (shots[i + 3]) { mx.push(px); my.push(py); }
    else              { hx.push(px); hy.push(py); }
  }
  return { mx, my, hx, hy };
}

// The shot-density Plotly traces: a filled 2-D histogram contour under the
// court outline. Layout/config stay per-page.
function densityTraces(mx, my, hx, hy) {
  const allX = [...mx, ...hx], allY = [...my, ...hy];
  return [
    { type: "histogram2dcontour", x: allX, y: allY, colorscale: "YlOrRd", showscale: false, ncontours: 20,
      opacity: 1.0, hoverinfo: "skip", contours: { coloring: "fill", showlines: false },
      xbins: { start: -26, end: 26, size: 2.5 }, ybins: { start: -2.5, end: 31, size: 2.5 }, zmin: 0 },
    ...courtTraces("rgba(255,255,255,0.85)"),
  ];
}
