/* ════════════════════════════════════════════════════════════════════════
   lineup-engine.js — compute ANY on/off split in the browser.

   Consumes the per-team payload written by build_site.py (section 4b) from the
   parquet tables build_lineups.py emits:

     roster : [{id, n}]                        index-addressable
     stints : flat ints, 11 per stint —
              [p0..p4, PF, Poss_Off×10, PA, Poss_Def×10, secs, scope_bits]
              pN are roster indices of OUR five on the floor. Possessions are
              stored ×10 as ints because 0.44·FTA makes them fractional.
     shots  : flat ints, 2 per shot — [stint_index, code]
              code = zone_index×4 + made×2 + is_our_shot
              is_our_shot=0 means the OPPONENT took it, i.e. our defense was on
              the floor for it.

   Nothing here is precomputed: pick any set of players and get that filter's
   ratings and shot profile with no possession threshold. Scope (reg / post /
   conf / nonconf / all) is a bitmask over the same payload, so one file per
   team serves every toggle.

   The stints are the same ones that build the Lineup Stats combos, so a
   one-player split here equals that player's row on the 1-Man tab.
   ════════════════════════════════════════════════════════════════════════ */
const LineupEngine = (function () {
  "use strict";

  const STRIDE = 11, SHOT_STRIDE = 2, N_ZONES = 14;

  // Scope bitfield — MUST match SCOPE_BIT in build_lineups.py.
  const SCOPE_MASK = { reg: 1, post: 2, conf: 4, nonconf: 8, all: 1 | 2 };

  /** Bitmask per stint for fast membership tests. Rosters run well under 32
      players in practice; beyond that fall back to a Set per stint. */
  function index(payload) {
    const n = payload.stints.length / STRIDE;
    const masks = new Uint32Array(n);
    const wide = payload.roster.length > 31 ? [] : null;
    for (let s = 0; s < n; s++) {
      const o = s * STRIDE;
      let m = 0;
      const set = wide ? new Set() : null;
      for (let k = 0; k < 5; k++) {
        const pi = payload.stints[o + k];
        if (wide) set.add(pi); else m |= (1 << pi);
      }
      if (wide) wide.push(set); else masks[s] = m;
    }
    return { n, masks, wide, payload };
  }

  const _has = (ix, s, pi) => ix.wide ? ix.wide[s].has(pi) : !!(ix.masks[s] & (1 << pi));

  /** Minutes played per roster index, for ordering the player pool. */
  function minutes(ix, scope) {
    const S = ix.payload.stints, mask = SCOPE_MASK[scope] ?? SCOPE_MASK.all;
    const out = new Array(ix.payload.roster.length).fill(0);
    for (let s = 0; s < ix.n; s++) {
      const o = s * STRIDE;
      if (!(S[o + 10] & mask)) continue;
      const mins = S[o + 9] / 60;
      for (let k = 0; k < 5; k++) out[S[o + k]] += mins;
    }
    return out;
  }

  /** Stints where every player in `onIdxs` is ON the floor and every player in
      `offIdxs` is OFF it, within one scope — the two-bucket Builder filter. */
  function selectStints(ix, onIdxs, offIdxs, scope) {
    const S = ix.payload.stints, mask = SCOPE_MASK[scope] ?? SCOPE_MASK.all;
    const out = [];
    for (let s = 0; s < ix.n; s++) {
      if (!(S[s * STRIDE + 10] & mask)) continue;
      let ok = true;
      for (const pi of onIdxs) if (!_has(ix, s, pi)) { ok = false; break; }
      if (ok) for (const pi of offIdxs) if (_has(ix, s, pi)) { ok = false; break; }
      if (ok) out.push(s);
    }
    return out;
  }

  /** Ratings aggregated over a stint selection. Null when nothing matched at
      all; an individual rating is null when that end had no possessions, which
      a small enough filter can produce on one side only. */
  function agg(ix, sel) {
    const S = ix.payload.stints;
    let pf = 0, possFor = 0, pa = 0, possAgainst = 0, secs = 0;
    for (const s of sel) {
      const o = s * STRIDE;
      pf += S[o + 5]; possFor += S[o + 6]; pa += S[o + 7]; possAgainst += S[o + 8];
      secs += S[o + 9];
    }
    possFor /= 10; possAgainst /= 10;
    if (possFor <= 0 && possAgainst <= 0) return null;
    const mins = secs / 60;
    const ortg = possFor > 0 ? pf / possFor * 100 : null;
    const drtg = possAgainst > 0 ? pa / possAgainst * 100 : null;
    return {
      ortg, drtg,
      net: (ortg != null && drtg != null) ? ortg - drtg : null,
      poss: (possFor + possAgainst) / 2, mins,
      pace: mins > 0 ? ((possFor + possAgainst) / 2) / mins * 40 : null,
      stints: sel.length,
    };
  }

  /** Shot profile by zone over a stint selection.
      side "own" = shots we took; "opp" = shots the opponent took against us.
      Returns [[zone_index, makes, attempts], …] — the shape ZoneChart wants. */
  function zoneProfile(ix, sel, side) {
    const want = side === "own" ? 1 : 0;
    const inSelection = new Uint8Array(ix.n);
    for (const s of sel) inSelection[s] = 1;
    const made = new Array(N_ZONES).fill(0), att = new Array(N_ZONES).fill(0);
    const H = ix.payload.shots;
    for (let i = 0; i < H.length; i += SHOT_STRIDE) {
      if (!inSelection[H[i]]) continue;
      const code = H[i + 1];
      if ((code & 1) !== want) continue;
      const z = code >> 2;
      att[z]++; if ((code >> 1) & 1) made[z]++;
    }
    const out = [];
    for (let z = 0; z < N_ZONES; z++) if (att[z] > 0) out.push([z, made[z], att[z]]);
    return out;
  }

  /** Zone totals collapsed into readable buckets for the summary table. */
  const BUCKETS = [
    ["Rim", [0]],
    ["Close mid", [1, 2, 3]],
    ["Mid", [4, 5, 6, 7, 8]],
    ["Corner 3", [9, 13]],
    ["Above the break 3", [10, 11, 12]],
  ];
  function buckets(zoneData) {
    const byZone = {};
    for (const [z, m, a] of zoneData) byZone[z] = { m, a };
    return BUCKETS.map(([label, zones]) => {
      let made = 0, att = 0;
      zones.forEach(z => { const v = byZone[z]; if (v) { made += v.m; att += v.a; } });
      return { label, made, att, pct: att ? made / att : null };
    });
  }

  /** The headline call: ratings + both shot profiles for one filter. */
  function filter(ix, onIdxs, offIdxs, scope) {
    const sel = selectStints(ix, onIdxs, offIdxs, scope);
    return {
      ratings:  agg(ix, sel),
      ownZones: zoneProfile(ix, sel, "own"),
      oppZones: zoneProfile(ix, sel, "opp"),
      stints:   sel.length,
    };
  }

  return { index, filter, minutes, selectStints, agg, zoneProfile, buckets,
           BUCKETS, SCOPE_MASK };
})();
