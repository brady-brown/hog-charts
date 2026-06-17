import streamlit as st

import ui

st.set_page_config(page_title="Hog Charts", layout="wide")
ui.inject_css()

st.markdown(
    f"""
    <div style="text-align:center;margin:8px 0 4px">
      <div style="font-size:3rem;font-weight:900;letter-spacing:-.03em">Hog Charts</div>
      <div style="color:{ui.MUTED};font-size:1.05rem;margin-top:2px">
        Free and accessible college basketball analytics
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---- primary features -------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="hc-card" style="min-height:170px">
      <div style="font-size:1.5rem;font-weight:900">Game Predictor</div>
      <div style="color:{ui.MUTED};margin-top:8px">
        Predict a game based off of efficiency, tempo, recent form, and home court.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Prediction.py", label="Open the Predictor", use_container_width=True)

with c2:
    st.markdown(f"""
    <div class="hc-card" style="min-height:170px">
      <div style="font-size:1.5rem;font-weight:900">Net Ratings</div>
      <div style="color:{ui.MUTED};margin-top:8px">
        Opponent-adjusted efficiency for every Division I team.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Net_Ratings.py", label="Browse Net Ratings", use_container_width=True)

st.write("")
st.markdown(f'<div class="hc-sec">More tools</div>', unsafe_allow_html=True)

c3, c4, c5 = st.columns(3)
with c3:
    st.markdown("**Shot Charts** — zone efficiency for any player or team.")
    st.page_link("pages/3_Shot_Charts.py", label="Open")
with c4:
    st.markdown("**Player Stats** — RAPM, on/off, and counting stats.")
    st.page_link("pages/4_Player_Stats.py", label="Open")
with c5:
    st.markdown("**Lineup Stats** — 1-5 man unit data per 100 possessions.")
    st.page_link("pages/5_Lineup_Stats.py", label="Open")

st.divider()
st.caption(
    "Built by Brady Brown & Wyatt Thompson. Follow @hogcharts on Instagram."
)
