import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mbb_viz import MBBZoneEfficiencyVisualizer

st.set_page_config(page_title="Shot Charts | Hog Charts", page_icon="🏀", layout="wide")
st.title("🏀 Shot Charts")


@st.cache_resource(show_spinner="Loading 2026 season data... (first load takes ~1 minute)")
def load_viz():
    return MBBZoneEfficiencyVisualizer(season=2026)


viz = load_viz()

all_teams = sorted(viz.box_df["team_display_name"].dropna().unique())

st.sidebar.header("Chart Controls")

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Player — Full Season", "Player — Single Game", "Team — Full Season", "Team — Single Game", "Territory Map"],
)

# Style options depend on season vs game vs territory
is_season = chart_type in ("Player — Full Season", "Team — Full Season")
is_game   = chart_type in ("Player — Single Game", "Team — Single Game")

if is_season:
    style = st.sidebar.radio(
        "Style",
        ["Zone Efficiency", "Shot Density"],
        captions=[
            "FG% breakdown by court zone",
            "Heatmap showing where shots are concentrated",
        ],
    )
elif is_game:
    style = st.sidebar.radio(
        "Style",
        ["Zone Efficiency", "Shot Scatter"],
        captions=[
            "FG% breakdown by court zone",
            "Every shot plotted — green makes, red misses",
        ],
    )
else:
    style = None  # Territory Map — no style choice

selected_team = st.sidebar.selectbox("Team", all_teams)

if chart_type in ("Player — Full Season", "Player — Single Game"):
    roster = sorted(
        viz.box_df[viz.box_df["team_display_name"] == selected_team]["athlete_display_name"]
        .dropna()
        .unique()
    )
    selected_player = st.sidebar.selectbox("Player", roster)

if is_game:
    opponent  = st.sidebar.selectbox("Opponent", [t for t in all_teams if t != selected_team])
    game_date = st.sidebar.text_input("Game Date (YYYY-MM-DD)", placeholder="e.g. 2026-01-20")

if st.sidebar.button("Generate Chart", type="primary"):
    try:
        fig = None

        if chart_type == "Player — Full Season":
            header = f"{selected_player} — {selected_team} | 2026 Season"
            with st.spinner(f"Generating chart for {selected_player}..."):
                if style == "Zone Efficiency":
                    fig = viz.player_season_chart(selected_player, selected_team, return_fig=True)
                else:
                    fig = viz.player_season_density(selected_player, selected_team, return_fig=True)

        elif chart_type == "Player — Single Game":
            if not game_date:
                st.error("Enter a game date.")
                st.stop()
            header = f"{selected_player} vs {opponent} ({game_date})"
            with st.spinner("Generating chart..."):
                if style == "Zone Efficiency":
                    fig = viz.player_game_chart(selected_player, selected_team, opponent, game_date, return_fig=True)
                else:
                    fig = viz.player_game_scatter(selected_player, selected_team, opponent, game_date, return_fig=True)

        elif chart_type == "Team — Full Season":
            header = f"{selected_team} — 2026 Season"
            with st.spinner(f"Generating chart for {selected_team}..."):
                if style == "Zone Efficiency":
                    fig = viz.team_season_chart(selected_team, return_fig=True)
                else:
                    fig = viz.team_season_density(selected_team, return_fig=True)

        elif chart_type == "Team — Single Game":
            if not game_date:
                st.error("Enter a game date.")
                st.stop()
            header = f"{selected_team} vs {opponent} ({game_date})"
            with st.spinner("Generating chart..."):
                if style == "Zone Efficiency":
                    fig = viz.team_game_chart(selected_team, opponent, game_date, return_fig=True)
                else:
                    fig = viz.team_game_scatter(selected_team, opponent, game_date, return_fig=True)

        elif chart_type == "Territory Map":
            header = f"{selected_team} — Territory Map | Top Scorer per Zone"
            with st.spinner(f"Building territory map for {selected_team}..."):
                fig = viz.plot_team_zone_leaders(selected_team, return_fig=True)

        if fig:
            st.subheader(header)
            st.pyplot(fig)
        else:
            st.warning("No shot data found for this selection.")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
