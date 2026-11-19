/* predict.js — the game model, shared by the Predictor and Games pages.
 *
 *   predictGame(model, t1, t2, venue)  venue: "home1" | "home2" | "neutral"
 *       → {spread, prob, homeCourt, expPace, paceFactor, tempoAdj, formDiff, coef, intercept}
 *       spread > 0 = t1 favored by that many points; prob = t1's win probability.
 *   projScores(t1, t2, spread, expPace, lgDef, model) → {s1, s2}
 *       model (optional) applies the fitted total correction (total_intercept/total_coef).
 *   leagueDef(teams) → the league-average def_eff projScores needs.
 *
 * model and teams come from data/{season}/predictor.json (build_site.py).
 * Needs common.js (interp). Mirrors model_runtime.py predict_game.
 */
function predictGame(model, t1, t2, venue) {
  const neutral   = venue === "neutral";
  const team1Home = venue === "home1";
  const lgAdv     = model.league_home_adv;

  let homeCourt = 0;
  if (!neutral) {
    homeCourt = team1Home
      ? (t1.home_adv ?? lgAdv)
      : -(t2.home_adv ?? lgAdv);
  }

  const netDiff    = t1.net_eff  - t2.net_eff;
  const expPace    = (t1.pace    + t2.pace) / 2;
  const paceFactor = expPace / model.league_avg_pace;
  const tempoAdj   = netDiff * paceFactor;
  const formDiff   = t1.form - t2.form;

  const [c0, c1, c2] = model.coef;
  const spread = model.intercept + c0 * tempoAdj + c1 * homeCourt + c2 * formDiff;
  const prob   = interp(spread, model.iso_x, model.iso_y);

  return { spread, prob, homeCourt, expPace, paceFactor, tempoAdj, formDiff,
           coef: model.coef, intercept: model.intercept };
}

// The raw total runs low against finals (it isn't fit to scores the way the
// spread is), so predictor.py fits actual = total_intercept + total_coef × raw.
function projScores(t1, t2, spread, expPace, lgDef, model) {
  const pp1   = t1.off_eff + (t2.def_eff - lgDef);
  const pp2   = t2.off_eff + (t1.def_eff - lgDef);
  const raw   = (pp1 + pp2) / 100 * expPace;
  const total = model?.total_coef != null ? model.total_intercept + model.total_coef * raw : raw;
  return { s1: Math.round((total + spread) / 2),
           s2: Math.round((total - spread) / 2) };
}

function leagueDef(teams) {
  return teams.reduce((s, t) => s + t.def_eff, 0) / teams.length;
}
