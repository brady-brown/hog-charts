import streamlit as st
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mbb_viz import MBBZoneEfficiencyVisualizer

st.set_page_config(page_title="Shot Charts | Hog Charts", layout="wide")
st.title("Shot Charts")

BASE = os.path.dirname(os.path.dirname(__file__))

# ── Team/player lists and game schedule from committed files ──────────────────
@st.cache_data
def get_roster_map():
    df = pd.read_csv(os.path.join(BASE, "player_stats.csv"))
    return df[["team_display_name", "athlete_display_name"]].dropna()

@st.cache_data
def get_game_schedule():
    return pd.read_parquet(os.path.join(BASE, "game_schedule.parquet"))

roster   = get_roster_map()
schedule = get_game_schedule()
all_teams = sorted(roster["team_display_name"].unique())

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Chart Controls")

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Player — Full Season", "Team — Full Season", "Territory Map",
     "Player — Single Game", "Team — Single Game"],
)

is_season = chart_type in ("Player — Full Season", "Team — Full Season")
is_game   = chart_type in ("Player — Single Game", "Team — Single Game")

if is_season:
    style = st.sidebar.radio(
        "Style", ["Zone Efficiency", "Shot Density"],
        captions=["FG% by court zone", "Heatmap of shot locations"],
    )
elif is_game:
    style = st.sidebar.radio(
        "Style", ["Zone Efficiency", "Shot Scatter"],
        captions=["FG% by court zone", "Every shot — green makes, red misses"],
    )
else:
    style = None

selected_team = st.sidebar.selectbox("Team", all_teams)

if chart_type in ("Player — Full Season", "Player — Single Game"):
    players = sorted(
        roster[roster["team_display_name"] == selected_team]["athlete_display_name"].unique()
    )
    selected_player = st.sidebar.selectbox("Player", players)
else:
    selected_player = None

if is_game:
    team_games = schedule[schedule["team"] == selected_team].sort_values("date")
    game_labels = team_games["label"].tolist()
    if game_labels:
        selected_label = st.sidebar.selectbox("Game", game_labels)
        selected_row   = team_games[team_games["label"] == selected_label].iloc[0]
        opponent       = selected_row["opponent"]
        game_date      = selected_row["date"]
    else:
        st.sidebar.warning("No games found for this team.")
        opponent  = None
        game_date = None
else:
    opponent  = None
    game_date = None

# ── Shot data — loaded per team, cached so same team is instant ───────────────
@st.cache_resource(show_spinner=f"Loading shot data...")
def load_viz(team_name):
    return MBBZoneEfficiencyVisualizer(season=2026, team_filter=team_name)

st.info("Set your options in the sidebar on the left, then click **Generate Chart**. "
        "The first load for a team takes about 30 seconds; the same team is instant after that.")

if st.sidebar.button("Generate Chart", type="primary"):
    try:
        with st.spinner(f"Loading {selected_team} shot data..."):
            viz = load_viz(selected_team)

        fig = None

        if chart_type == "Player — Full Season":
            header = f"{selected_player} — {selected_team} | 2026 Season"
            with st.spinner(f"Generating chart for {selected_player}..."):
                if style == "Zone Efficiency":
                    fig = viz.player_season_chart(selected_player, selected_team, return_fig=True)
                else:
                    fig = viz.player_season_density(selected_player, selected_team, return_fig=True)

        elif chart_type == "Team — Full Season":
            header = f"{selected_team} — 2026 Season"
            with st.spinner(f"Generating chart for {selected_team}..."):
                if style == "Zone Efficiency":
                    fig = viz.team_season_chart(selected_team, return_fig=True)
                else:
                    fig = viz.team_season_density(selected_team, return_fig=True)

        elif chart_type == "Territory Map":
            header = f"{selected_team} — Territory Map | Top Scorer per Zone"
            with st.spinner(f"Building territory map for {selected_team}..."):
                fig = viz.plot_team_zone_leaders(selected_team, return_fig=True)

        elif chart_type == "Player — Single Game":
            header = f"{selected_player} vs {opponent} ({game_date})"
            with st.spinner("Generating chart..."):
                if style == "Zone Efficiency":
                    fig = viz.player_game_chart(selected_player, selected_team, opponent, game_date, return_fig=True)
                else:
                    fig = viz.player_game_scatter(selected_player, selected_team, opponent, game_date, return_fig=True)

        elif chart_type == "Team — Single Game":
            header = f"{selected_team} vs {opponent} ({game_date})"
            with st.spinner("Generating chart..."):
                if style == "Zone Efficiency":
                    fig = viz.team_game_chart(selected_team, opponent, game_date, return_fig=True)
                else:
                    fig = viz.team_game_scatter(selected_team, opponent, game_date, return_fig=True)

        if fig:
            st.subheader(header)
            st.pyplot(fig)
        else:
            st.warning("No shot data found for this selection.")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
