// percentile.js — shared percentile-rank machinery for Player Stats + Team Stats.
//
// Both pages rank a row's stat against the FULL qualified pool for the current
// scope (not the filtered view), so a percentile holds regardless of the
// conference/team/search filters — the way Cleaning the Glass does it. The
// pool construction differs per page (which rows qualify, which keys), so that
// stays on the page; this module owns the reusable parts:
//   buildPools(rows, keys, qualifies) → { key: ascending value array }
//   rank(poolArray, val, higherBetter) → 0–100 (100 = best) or null
//   color(pct) → { bg, fg } diverging red→amber→green band
const Percentile = (function () {
  "use strict";

  // first index i with arr[i] >= x  (count of values strictly < x)
  function lowerBound(arr, x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
    return lo;
  }
  // first index i with arr[i] > x
  function upperBound(arr, x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
    return lo;
  }

  // Build an ascending-sorted value array per key from the rows that pass the
  // optional qualifies(row) predicate (defaults to every row).
  function buildPools(rows, keys, qualifies) {
    const q = qualifies || (() => true);
    const sortedByKey = {};
    keys.forEach(k => {
      const vals = [];
      for (const r of (rows || [])) {
        if (!q(r)) continue;
        const x = r[k];
        if (x != null && !isNaN(x)) vals.push(+x);
      }
      vals.sort((a, b) => a - b);
      sortedByKey[k] = vals;
    });
    return sortedByKey;
  }

  // Percentile 0–100 where 100 = best. higherBetter flips for "lower is better"
  // stats (defense, turnovers). Leader → 100, worst → 0.
  function rank(arr, val, higherBetter) {
    if (!arr || arr.length < 2 || val == null || isNaN(val)) return null;
    const n = arr.length;
    const below = lowerBound(arr, val);        // strictly less
    const above = n - upperBound(arr, val);    // strictly greater
    return 100 * (higherBetter ? below : above) / (n - 1);
  }

  // Diverging colour band: red (0, worst) → amber (50) → green (100, best).
  function color(p) {
    const hue = (p / 100) * 130;   // 0 = red, 130 = green
    return { bg: `hsla(${hue},70%,42%,0.20)`, fg: `hsl(${hue},75%,72%)` };
  }

  return { buildPools, rank, color };
})();
