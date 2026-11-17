/* predict.js — the game model, shared by the Predictor and Games pages.
 *
 *   predictGame(model, t1, t2, venue)  venue: "home1" | "home2" | "neutral"
 *       → {spread, prob, homeCourt, expPace, paceFactor, tempoAdj, formDiff, coef, intercept}
 *       spread > 0 = t1 favored by that many points; prob = t1's win probability.
 *   projScores(t1, t2, spread, expPace, lgDef) → {s1, s2}
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

// Mirrors pages/1_Prediction.py proj_scores()
function projScores(t1, t2, spread, expPace, lgDef) {
  const pp1   = t1.off_eff + (t2.def_eff - lgDef);
  const pp2   = t2.off_eff + (t1.def_eff - lgDef);
  const total = (pp1 + pp2) / 100 * expPace;
  return { s1: Math.round((total + spread) / 2),
           s2: Math.round((total - spread) / 2) };
}

function leagueDef(teams) {
  return teams.reduce((s, t) => s + t.def_eff, 0) / teams.length;
}
