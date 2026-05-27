import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Lineup Stats | Hog Charts", page_icon="📋", layout="wide")
st.title("📋 Lineup Stats — 2025-26 Season")

BASE = os.path.dirname(os.path.dirname(__file__))

N_MAN_FILES = {
    1: {"Overall": "1_man_overall_stats.csv", "Conference": "1_man_conference_stats.csv"},
    2: {"Overall": "2_man_overall_stats.csv", "Conference": "2_man_conference_stats.csv"},
    3: {"Overall": "3_man_overall_stats.csv", "Conference": "3_man_conference_stats.csv"},
    5: {"Overall": "5_man_overall_stats.csv", "Conference": "5_man_conference_stats.csv"},
}

DISPLAY_COLS = ["Combo", "Team", "Mins", "Avg_Poss", "NetRtg", "ORtg", "DRtg",
                "FGM_100", "FGA_100", "FG3M_100", "FG3A_100", "AST_100", "TOV_100",
                "ORB_100", "DRB_100", "STL_100", "BLK_100"]

COL_LABELS = {
    "Avg_Poss": "Poss",
    "FGM_100": "FGM/100",
    "FGA_100": "FGA/100",
    "FG3M_100": "3PM/100",
    "FG3A_100": "3PA/100",
    "AST_100": "AST/100",
    "TOV_100": "TOV/100",
    "ORB_100": "ORB/100",
    "DRB_100": "DRB/100",
    "STL_100": "STL/100",
    "BLK_100": "BLK/100",
}


@st.cache_data
def load_lineup(n_man, scope):
    fname = N_MAN_FILES[n_man][scope]
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    existing = [c for c in DISPLAY_COLS if c in df.columns]
    df = df[existing].copy()
    for c in ["NetRtg", "ORtg", "DRtg"]:
        if c in df.columns:
            df[c] = df[c].round(1)
    for c in ["Mins", "Avg_Poss"]:
        if c in df.columns:
            df[c] = df[c].round(1)
    return df.rename(columns=COL_LABELS)


col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    n_man = st.selectbox("Lineup Size", [1, 2, 3, 5], format_func=lambda x: f"{x}-Man")
with col2:
    scope = st.selectbox("Games", ["Overall", "Conference"])

df = load_lineup(n_man, scope)

if df is None:
    st.error(f"Data file not found for {n_man}-man {scope}.")
    st.stop()

all_teams = sorted(df["Team"].dropna().unique()) if "Team" in df.columns else []

with col3:
    team_filter = st.selectbox("Filter by Team", ["All Teams"] + all_teams)

col_a, col_b = st.columns(2)
with col_a:
    min_mins = st.number_input("Min Minutes", min_value=0.0, value=10.0, step=5.0)
with col_b:
    sort_col = st.selectbox("Sort By", ["NetRtg", "ORtg", "DRtg", "Mins"], index=0)

display = df.copy()
if team_filter != "All Teams" and "Team" in display.columns:
    display = display[display["Team"] == team_filter]
if "Mins" in display.columns:
    display = display[display["Mins"] >= min_mins]
if sort_col in display.columns:
    display = display.sort_values(sort_col, ascending=(sort_col == "DRtg"))

display = display.reset_index(drop=True)
display.index += 1

st.caption(f"{len(display)} lineups shown")
st.dataframe(display, use_container_width=True)
