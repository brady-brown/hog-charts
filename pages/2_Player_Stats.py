import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Player Stats | Hog Charts", page_icon="📊", layout="wide")
st.title("📊 Player Stats — 2025-26 Season")

BASE = os.path.dirname(os.path.dirname(__file__))

POS_MAP = {
    "Guard":   ["Guard", "Point Guard", "Shooting Guard"],
    "Forward": ["Forward", "Power Forward", "Small Forward"],
    "Center":  ["Center"],
}
REVERSE_POS = {raw: broad for broad, raws in POS_MAP.items() for raw in raws}

COUNTING_FILES = {
    "Overall":          "player_stats.csv",
    "Conference Only":  "player_stats_conf.csv",
}
RAPM_FILES = {
    "Overall":          "mbb_rapm_202526.csv",
    "Conference Only":  "mbb_rapm_202526_conf.csv",
}
ONOFF_FILES = {
    "Overall":          "mbb_onoff_2026_v2.csv",
    "Conference Only":  "mbb_onoff_2026_conf_v2.csv",
}


@st.cache_data
def load_conf_lookup():
    df = pd.read_csv(os.path.join(BASE, "player_stats.csv"))
    return (
        df[["team_display_name", "conf."]]
        .dropna()
        .drop_duplicates(subset=["team_display_name"])
        .rename(columns={"team_display_name": "Team", "conf.": "Conf"})
    )


@st.cache_data
def load_counting(scope):
    df = pd.read_csv(os.path.join(BASE, COUNTING_FILES[scope]))
    keep = {
        "athlete_display_name": "Player",
        "team_display_name":    "Team",
        "conf.":                "Conf",
        "athlete_position_name": "_RawPos",
        "games_played":         "GP",
        "minute_avg":           "MPG",
        "points_avg":           "PPG",
        "reb_avg":              "RPG",
        "ast_avg":              "APG",
        "steal_avg":            "SPG",
        "blocks_avg":           "BPG",
        "to_avg":               "TOV",
        "fg_pct":               "FG%",
        "efg_pct":              "eFG%",
        "3pt_pct":              "3P%",
        "ft_pct":               "FT%",
        "3ptm_avg":             "3PM",
        "3pta_avg":             "3PA",
        "fgm_avg":              "FGM",
        "fga_avg":              "FGA",
        "oreb_avg":             "OREB",
        "dreb_avg":             "DREB",
    }
    df = df.rename(columns=keep)
    df["Pos"] = df["_RawPos"].map(REVERSE_POS).fillna(df["_RawPos"])
    cols = ["Player", "Team", "Conf", "Pos", "GP", "MPG", "PPG", "RPG", "APG",
            "SPG", "BPG", "TOV", "FG%", "eFG%", "3P%", "FT%",
            "3PM", "3PA", "FGM", "FGA", "OREB", "DREB"]
    df = df[[c for c in cols if c in df.columns]]
    for col in ["FG%", "eFG%", "3P%", "FT%"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(1)
    for col in ["MPG","PPG","RPG","APG","SPG","BPG","TOV","3PM","3PA","FGM","FGA","OREB","DREB"]:
        if col in df.columns:
            df[col] = df[col].round(1)
    return df


@st.cache_data
def load_rapm(scope):
    df = pd.read_csv(os.path.join(BASE, RAPM_FILES[scope]))
    df = df.rename(columns={
        "athlete_display_name": "Player",
        "team_display_name":    "Team",
        "o_rapm":               "O-RAPM",
        "d_rapm":               "D-RAPM",
        "rapm":                 "RAPM",
        "total_poss":           "Possessions",
    })
    df["Possessions"] = df["Possessions"].round(0).astype(int)
    df = df.merge(load_conf_lookup(), on="Team", how="left")
    return df[["Player", "Team", "Conf", "RAPM", "O-RAPM", "D-RAPM", "Possessions"]]


@st.cache_data
def load_onoff(scope):
    df = pd.read_csv(os.path.join(BASE, ONOFF_FILES[scope]))
    cols = {
        "athlete_display_name": "Player",
        "team_display_name":    "Team",
        "poss_off_on":          "Poss On",
        "nrtg_on":              "NetRtg On",
        "ortg_on":              "ORtg On",
        "drtg_on":              "DRtg On",
        "nrtg_off":             "NetRtg Off",
        "ortg_off":             "ORtg Off",
        "drtg_off":             "DRtg Off",
        "on_off":               "On-Off",
        "efg_pct_on":           "eFG% On",
        "3p_pct_on":            "3P% On",
    }
    df = df.rename(columns=cols)[list(cols.values())]
    for c in ["NetRtg On","ORtg On","DRtg On","NetRtg Off","ORtg Off","DRtg Off","On-Off"]:
        df[c] = df[c].round(1)
    for c in ["eFG% On", "3P% On"]:
        df[c] = (df[c] * 100).round(1)
    df["Poss On"] = df["Poss On"].round(0).astype(int)
    df = df.merge(load_conf_lookup(), on="Team", how="left")
    return df


# ── Global scope toggle ─────────────────────────────────────────────────────────
scope = st.radio("Games", ["Overall", "Conference Only"], horizontal=True)

counting_df = load_counting(scope)
rapm_df     = load_rapm(scope)
onoff_df    = load_onoff(scope)

all_confs = sorted(counting_df["Conf"].dropna().unique())

tab1, tab2, tab3 = st.tabs(["Counting Stats", "RAPM", "On/Off Splits"])

# ── Counting Stats ──────────────────────────────────────────────────────────────
with tab1:
    all_teams_c = sorted(counting_df["Team"].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        team_c = st.selectbox("Team", ["All Teams"] + all_teams_c, key="cnt_team")
    with col2:
        conf_c = st.selectbox("Conference", ["All"] + all_confs, key="cnt_conf")
    with col3:
        pos_c = st.selectbox("Position", ["All", "Guard", "Forward", "Center"], key="cnt_pos")

    col4, col5 = st.columns(2)
    with col4:
        min_gp = st.number_input("Min Games Played", min_value=0, value=10, step=1, key="cnt_gp")
    with col5:
        sort_c = st.selectbox("Sort By", ["PPG","RPG","APG","SPG","BPG","FG%","eFG%","3P%","3PM","MPG"], key="cnt_sort")

    d = counting_df.copy()
    if team_c != "All Teams": d = d[d["Team"] == team_c]
    if conf_c != "All":       d = d[d["Conf"] == conf_c]
    if pos_c != "All":        d = d[d["Pos"] == pos_c]
    d = d[d["GP"] >= min_gp].sort_values(sort_c, ascending=False).reset_index(drop=True)
    d.index += 1

    st.caption(f"{len(d)} players shown")
    st.dataframe(d, width='stretch')

# ── RAPM ────────────────────────────────────────────────────────────────────────
with tab2:
    all_teams_r = sorted(rapm_df["Team"].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        team_r = st.selectbox("Team", ["All Teams"] + all_teams_r, key="rapm_team")
    with col2:
        conf_r = st.selectbox("Conference", ["All"] + all_confs, key="rapm_conf")
    with col3:
        min_poss_r = st.number_input("Min Possessions", min_value=0, value=500, step=100, key="rapm_poss")

    d = rapm_df.copy()
    if team_r != "All Teams": d = d[d["Team"] == team_r]
    if conf_r != "All":       d = d[d["Conf"] == conf_r]
    d = d[d["Possessions"] >= min_poss_r].sort_values("RAPM", ascending=False).reset_index(drop=True)
    d.index += 1

    st.caption(f"{len(d)} players shown")
    st.dataframe(d, width='stretch', column_config={
        "RAPM":   st.column_config.NumberColumn(format="%.2f"),
        "O-RAPM": st.column_config.NumberColumn(format="%.2f"),
        "D-RAPM": st.column_config.NumberColumn(format="%.2f"),
    })

# ── On/Off ──────────────────────────────────────────────────────────────────────
with tab3:
    all_teams_oo = sorted(onoff_df["Team"].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        team_oo = st.selectbox("Team", ["All Teams"] + all_teams_oo, key="oo_team")
    with col2:
        conf_oo = st.selectbox("Conference", ["All"] + all_confs, key="oo_conf")
    with col3:
        min_poss_oo = st.number_input("Min Possessions On", min_value=0, value=300, step=100, key="oo_poss")

    d = onoff_df.copy()
    if team_oo != "All Teams": d = d[d["Team"] == team_oo]
    if conf_oo != "All":       d = d[d["Conf"] == conf_oo]
    d = d[d["Poss On"] >= min_poss_oo].sort_values("On-Off", ascending=False).reset_index(drop=True)
    d.index += 1

    st.caption(f"{len(d)} players shown")
    st.dataframe(d, width='stretch')
