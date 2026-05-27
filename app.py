import streamlit as st

st.set_page_config(page_title="Hog Charts", page_icon="🐗", layout="wide")

st.title("🐗 Hog Charts")
st.subheader("Free College Basketball Analytics")

st.markdown("""
Built by Brady Brown & Wyatt Thompson — University of Arkansas data science students.

All data pulled from ESPN via API. No subscription required.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏀 Shot Charts")
    st.markdown("Zone efficiency charts for any player or team — full season or single game. Includes territory maps showing each team's top scorer by zone.")

with col2:
    st.markdown("### 📊 Player Stats")
    st.markdown("RAPM leaderboard and on/off splits for every player in the 2025-26 season. Filter by team, sort by any metric.")

with col3:
    st.markdown("### 📋 Lineup Stats")
    st.markdown("1, 2, 3, and 5-man lineup data per 100 possessions. Find the best and worst unit combinations for any team.")

st.divider()
st.caption("Data: 2025-26 NCAA Men's Basketball season via sportsdataverse. Follow us @hogcharts on Instagram.")
