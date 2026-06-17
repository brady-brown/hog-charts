import json
import os

import pandas as pd
import streamlit as st

import ui
from model_runtime import RuntimePredictor

st.set_page_config(page_title="Prediction | Hog Charts", page_icon="🔮", layout="wide")
ui.inject_css()

BASE = os.path.dirname(os.path.dirname(__file__))
ART = os.path.join(BASE, "artifacts")


@st.cache_resource
def load_predictor():
    return RuntimePredictor(ART)


@st.cache_data
def load_meta():
    with open(os.path.join(ART, "metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_ranks():
    df = pd.read_parquet(os.path.join(ART, "net_ratings.parquet"))
    return df.set_index("team").to_dict("index")


pred = load_predictor()
meta = load_meta()
ranks = load_ranks()

# League-average baselines for the projected-score estimate (efficiencies in this
# rating system are not centered at 100, so anchor to the actual league means).
LG_DEF = pred.lg_def

teams = pred.team_names()


def proj_scores(r):
    """Plausible final score whose MARGIN equals the model spread; total comes
    from each team's offense vs the other's defense at the projected pace."""
    pace = r["expected_pace"]
    pp1 = r["team1_off_eff"] + (r["team2_def_eff"] - LG_DEF)
    pp2 = r["team2_off_eff"] + (r["team1_def_eff"] - LG_DEF)
    total = (pp1 + pp2) / 100 * pace
    spread = r["team1_spread"]
    return round((total + spread) / 2), round((total - spread) / 2)


def rank_note(team, key):
    info = ranks.get(team)
    if not info:
        return ""
    return f"#{int(info[key])} in D-I"


# ------------------------------------------------------------------ header
st.markdown("## 🔮 Game Predictor")
st.markdown(
    f'<div style="color:{ui.MUTED};margin-top:-6px;margin-bottom:14px">'
    f"Pick two teams and where they play — get a projected score, win probability, "
    f"and exactly what's driving it. Model calibrated on "
    f'{min(meta["calibration_years"])}–{max(meta["calibration_years"])}, '
    f'{meta["backtest"]["accuracy"]*100:.1f}% accurate on the held-out {meta["backtest_year"]} season.'
    f"</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ inputs
c1, c2, c3 = st.columns([5, 5, 4])
default1 = teams.index("Duke Blue Devils") if "Duke Blue Devils" in teams else 0
default2 = teams.index("Arkansas Razorbacks") if "Arkansas Razorbacks" in teams else 1
with c1:
    team1 = st.selectbox("Team 1", teams, index=default1)
with c2:
    team2 = st.selectbox("Team 2", teams, index=default2)
with c3:
    venue = st.radio("Where is it played?",
                     [f"{team1} home", f"{team2} home", "Neutral site"],
                     horizontal=False)

go = st.button("Predict game", type="primary", use_container_width=True)

if team1 == team2:
    st.warning("Pick two different teams.")
    st.stop()

if not go and "predicted" not in st.session_state:
    st.info("Choose your matchup and hit **Predict game**.")
    st.stop()
st.session_state["predicted"] = True

# ------------------------------------------------------------------ predict
neutral = venue == "Neutral site"
team1_home = venue == f"{team1} home"
r = pred.predict_game(team1, team2, team1_home=team1_home, neutral_site=neutral, verbose=False)

t1_id, t2_id = pred.get_team_id(team1), pred.get_team_id(team2)
s1, s2 = proj_scores(r)

if neutral:
    sub1 = sub2 = "Neutral floor"
else:
    sub1 = "Home" if team1_home else "Away"
    sub2 = "Away" if team1_home else "Home"

ui.render_hero(t1_id, t2_id, r["team1"], r["team2"], sub1, sub2,
               s1, s2, r["team1_win_prob"], r["team2_win_prob"])

fav_t1 = r["team1_win_prob"] >= 0.5
ui.render_verdict(r["team1"] if fav_t1 else r["team2"],
                  r["team1_spread"] if fav_t1 else r["team2_spread"],
                  max(r["team1_win_prob"], r["team2_win_prob"]))

# ------------------------------------------------------------------ context chips
home_pts = pred.coef[1] * r["home_court_pts"]
home_team = None if neutral else (r["team1"] if team1_home else r["team2"])
home_chip = "Neutral floor" if neutral else f"{abs(home_pts):.1f} pts · {home_team.split()[-1]}"
st.markdown(f"""
<div class="hc-chips">
  <div class="hc-chip"><div class="n">{s1 + s2}</div><div class="l">Proj. total</div></div>
  <div class="hc-chip"><div class="n">{r['expected_pace']:.0f}</div><div class="l">Pace / poss</div></div>
  <div class="hc-chip"><div class="n">{home_chip}</div><div class="l">Home edge</div></div>
  <div class="hc-chip"><div class="n">{abs(r['team1_spread']):.1f}</div><div class="l">Spread</div></div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ factor breakdown
ui.card_open("Matchup factors")
ui.render_factor("Net rating", r["team1_net_eff"], r["team2_net_eff"],
                 note=f"{rank_note(team1,'rank')} vs {rank_note(team2,'rank')}")
ui.render_factor("Offense (pts/100)", r["team1_off_eff"], r["team2_off_eff"], fmt="{:.1f}",
                 note=f"{rank_note(team1,'off_rank')} vs {rank_note(team2,'off_rank')}")
ui.render_factor("Defense (pts/100)", r["team1_def_eff"], r["team2_def_eff"], fmt="{:.1f}",
                 higher_is_better=False,
                 note=f"{rank_note(team1,'def_rank')} vs {rank_note(team2,'def_rank')} · lower is better")
ui.render_factor("Recent form (L5)", r["team1_form"], r["team2_form"],
                 note="opponent-adjusted margin over last 5 games")
ui.card_close()

# ------------------------------------------------------------------ how the spread is built
with st.expander("How the model built this spread"):
    tempo_adj = (r["team1_net_eff"] - r["team2_net_eff"]) * r["pace_factor"]
    contrib = {
        "Efficiency × tempo": pred.coef[0] * tempo_adj,
        "Home court": pred.coef[1] * r["home_court_pts"],
        "Recent form": pred.coef[2] * (r["team1_form"] - r["team2_form"]),
        "Baseline": pred.intercept,
    }
    bd = pd.DataFrame(
        [{"Factor": k, f"Points toward {r['team1']}": round(v, 2)} for k, v in contrib.items()]
        + [{"Factor": "→ Predicted margin", f"Points toward {r['team1']}": round(r["team1_spread"], 2)}]
    )
    st.dataframe(bd, hide_index=True, use_container_width=True)
    st.caption(
        f"Win probability is a calibrated, monotone transform of the predicted margin, "
        f"so the favorite on the spread is always the favorite to win. "
        f"Pace factor {r['pace_factor']:.3f} means this game projects "
        f"{(r['pace_factor']-1)*100:+.1f}% vs the average tempo."
    )
