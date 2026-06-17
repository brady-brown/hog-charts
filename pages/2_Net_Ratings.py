import json
import os

import pandas as pd
import streamlit as st

import ui

st.set_page_config(page_title="Net Ratings | Hog Charts", layout="wide")
ui.inject_css()

BASE = os.path.dirname(os.path.dirname(__file__))
ART = os.path.join(BASE, "artifacts")


@st.cache_data
def load_ratings():
    df = pd.read_parquet(os.path.join(ART, "net_ratings.parquet"))
    df["logo"] = df["team_id"].map(ui.logo_url)
    df["record"] = df["wins"].astype(int).astype(str) + "–" + df["losses"].astype(int).astype(str)
    return df


@st.cache_data
def load_meta():
    with open(os.path.join(ART, "metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_conf_lookup():
    path = os.path.join(BASE, "player_stats.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["team_display_name", "conf."])
    return df.dropna().drop_duplicates("team_display_name").set_index("team_display_name")["conf."].to_dict()


df = load_ratings()
meta = load_meta()
conf_map = load_conf_lookup()
df["conf"] = df["team"].map(conf_map).fillna("—")

# ------------------------------------------------------------------ header
st.markdown("## Net Ratings")
st.markdown(
    f'<div style="color:{ui.MUTED};margin-top:-6px;margin-bottom:8px">'
    f"Opponent-adjusted efficiency for every Division I team. Net rating is points scored "
    f"minus points allowed per 100 possessions, adjusted for opponent strength. "
    f"Click any column to sort.</div>",
    unsafe_allow_html=True,
)

# #1 highlight
top = df.iloc[0]
st.markdown(f"""
<div class="hc-card" style="display:flex;align-items:center;gap:18px">
  <img src="{ui.logo_url(top['team_id'])}" style="width:58px;height:58px;object-fit:contain">
  <div style="flex:1">
    <div style="color:{ui.MUTED};font-size:.74rem;letter-spacing:.08em;font-weight:700">#1 NET RATING</div>
    <div style="font-size:1.3rem;font-weight:900">{top['team']}</div>
    <div style="color:{ui.MUTED};font-size:.85rem">{top['record']} · {top['conf']}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:2rem;font-weight:900;color:{ui.T1}">{top['net_eff']:+.1f}</div>
    <div style="color:{ui.MUTED};font-size:.74rem;letter-spacing:.06em;font-weight:700">NET / 100</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ filters
f1, f2, f3 = st.columns([3, 2, 2])
with f1:
    search = st.text_input("Search team", placeholder="e.g. Arkansas")
with f2:
    confs = sorted(c for c in df["conf"].unique() if c != "—")
    sel_conf = st.multiselect("Conference", confs, placeholder="All conferences")
with f3:
    gmax = int(df["games"].max())
    min_games = st.slider("Minimum games played", min_value=5, max_value=gmax, value=15, step=1)

view = df[df["games"] >= min_games]
if search:
    view = view[view["team"].str.contains(search, case=False, na=False)]
if sel_conf:
    view = view[view["conf"].isin(sel_conf)]

st.caption(f"{len(view)} of {len(df)} teams")

# ------------------------------------------------------------------ table
cols = ["rank", "logo", "team", "conf", "record", "net_eff", "off_eff", "def_eff",
        "sos", "pace", "home_court", "form"]
st.dataframe(
    view[cols],
    hide_index=True,
    use_container_width=True,
    height=620,
    column_config={
        "rank": st.column_config.NumberColumn("Rk", width="small",
                                              help="National rank by net rating"),
        "logo": st.column_config.ImageColumn(" ", width="small"),
        "team": st.column_config.TextColumn("Team", width="medium"),
        "conf": st.column_config.TextColumn("Conf", width="small", help="Conference"),
        "record": st.column_config.TextColumn("Record", width="small", help="Wins-losses this season"),
        "net_eff": st.column_config.NumberColumn(
            "Net", format="%+.1f",
            help="Net rating: points scored minus points allowed per 100 possessions, "
                 "adjusted for opponent strength. Higher is better."),
        "off_eff": st.column_config.NumberColumn(
            "Off", format="%.1f",
            help="Offensive rating: points scored per 100 possessions, adjusted for the "
                 "defenses faced. Higher is better."),
        "def_eff": st.column_config.NumberColumn(
            "Def", format="%.1f",
            help="Defensive rating: points allowed per 100 possessions, adjusted for the "
                 "offenses faced. Lower is better."),
        "sos": st.column_config.NumberColumn(
            "SOS", format="%+.1f",
            help="Strength of schedule: the average net rating of the opponents this team "
                 "played. Higher means a tougher schedule."),
        "pace": st.column_config.NumberColumn(
            "Pace", format="%.1f", help="Average possessions per game. Higher means a faster team."),
        "home_court": st.column_config.NumberColumn(
            "Home", format="%+.1f",
            help="Points this team's home court is worth, the edge built into a home prediction."),
        "form": st.column_config.NumberColumn(
            "Form", format="%+.1f",
            help="Recent form: average margin over the last 5 games versus what the ratings "
                 "expected. Positive means playing above their rating lately."),
    },
)

st.caption(
    f"Built {meta['built_at'][:10]} · {meta['season']} season · {meta['n_teams']} teams. "
    f"Lower Defense = better. Form is opponent-adjusted (residual) margin over the last "
    f"{meta['form_games']} games."
)
