"""
ui.py — shared styling + components for the Hog Charts web app.

Centralizes the custom CSS theme, ESPN logo lookup, and the rendered HTML blocks
(matchup hero, win-prob bar, factor breakdown) so every page stays consistent
and stylish without repeating markup.
"""

import streamlit as st

# Team accent colors: team 1 is always warm (crimson), team 2 always cool (azure).
# Consistent across the whole UI so "left vs right" is unambiguous at a glance.
T1 = "#F0445A"   # crimson
T2 = "#3B82F6"   # azure
INK = "#F2F4F8"
MUTED = "#9AA4B2"
CARD = "#141925"
CARD2 = "#1B2230"
LINE = "#262F40"


def logo_url(team_id):
    return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{int(team_id)}.png"


def inject_css():
    """Global theme: fonts, spacing, cards, chips, bars. Call once per page."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"], .stMarkdown, p, span, div, label {
            font-family: 'Inter', -apple-system, sans-serif;
        }
        .block-container { padding-top: 2.2rem; max-width: 1100px; }
        #MainMenu, footer { visibility: hidden; }

        h1, h2, h3 { letter-spacing: -0.02em; font-weight: 800; }

        /* ---- generic card ---- */
        .hc-card {
            background: #141925;
            border: 1px solid #262F40;
            border-radius: 18px;
            padding: 22px 24px;
            margin-bottom: 16px;
        }

        /* ---- matchup hero ---- */
        .hc-hero {
            background: radial-gradient(120% 140% at 50% 0%, #1B2230 0%, #0E1320 70%);
            border: 1px solid #262F40;
            border-radius: 22px;
            padding: 26px 24px 22px;
            margin-bottom: 18px;
        }
        .hc-teams { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 10px; }
        .hc-team { text-align: center; }
        .hc-team img { width: 96px; height: 96px; object-fit: contain;
                       filter: drop-shadow(0 6px 18px rgba(0,0,0,0.5)); }
        .hc-team-name { font-weight: 800; font-size: 1.05rem; margin-top: 8px; line-height: 1.2; }
        .hc-team-sub { color: #9AA4B2; font-size: 0.8rem; margin-top: 2px; }
        .hc-vs { color: #6B7688; font-weight: 900; font-size: 1.1rem; letter-spacing: 0.05em; }
        .hc-score { font-size: 2.6rem; font-weight: 900; line-height: 1; }

        /* ---- verdict line ---- */
        .hc-verdict {
            text-align: center; margin: 18px auto 6px; font-size: 1.15rem; font-weight: 700;
        }
        .hc-verdict b { font-weight: 900; }

        /* ---- probability bar ---- */
        .hc-bar { height: 30px; border-radius: 999px; overflow: hidden; display: flex;
                  border: 1px solid #262F40; margin: 8px 0 4px; }
        .hc-bar > div { display: flex; align-items: center; font-weight: 800; font-size: 0.92rem;
                        color: #fff; padding: 0 14px; white-space: nowrap; }
        .hc-bar-l { justify-content: flex-start; }
        .hc-bar-r { justify-content: flex-end; }

        /* ---- factor rows ---- */
        .hc-factor { display: grid; grid-template-columns: 150px 1fr 150px; align-items: center;
                     gap: 14px; padding: 12px 0; border-bottom: 1px solid #1F2738; }
        .hc-factor:last-child { border-bottom: none; }
        .hc-factor .lab { color: #9AA4B2; font-size: 0.82rem; font-weight: 600; text-transform: uppercase;
                          letter-spacing: 0.04em; }
        .hc-factor .v1 { text-align: right; font-weight: 800; font-size: 1.05rem; }
        .hc-factor .v2 { text-align: left; font-weight: 800; font-size: 1.05rem; }
        .hc-factor .mid { text-align: center; }
        .hc-edge { display: inline-block; padding: 3px 10px; border-radius: 999px;
                   font-size: 0.78rem; font-weight: 800; }
        .hc-note { color: #6B7688; font-size: 0.78rem; text-align: center; margin-top: 2px; }

        /* ---- stat chips ---- */
        .hc-chips { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
        .hc-chip { background: #1B2230; border: 1px solid #262F40; border-radius: 12px;
                   padding: 10px 16px; text-align: center; min-width: 110px; }
        .hc-chip .n { font-weight: 900; font-size: 1.15rem; }
        .hc-chip .l { color: #9AA4B2; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }

        /* ---- section label ---- */
        .hc-sec { color: #9AA4B2; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                  letter-spacing: 0.08em; margin: 4px 0 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _logo(team_id, size=96):
    return (f'<img src="{logo_url(team_id)}" style="width:{size}px;height:{size}px;'
            f'object-fit:contain;filter:drop-shadow(0 6px 18px rgba(0,0,0,.5));" '
            f'onerror="this.style.opacity=0">')


def render_hero(t1_id, t2_id, name1, name2, venue_sub1, venue_sub2,
                score1, score2, win1, win2):
    """Top matchup card: logos, predicted score, win-prob bar."""
    c1, c2 = (T1, T2)
    bar_l_w = max(6, round(win1 * 100))
    bar_r_w = 100 - bar_l_w
    st.markdown(f"""
    <div class="hc-hero">
      <div class="hc-teams">
        <div class="hc-team">
          {_logo(t1_id)}
          <div class="hc-team-name" style="color:{c1}">{name1}</div>
          <div class="hc-team-sub">{venue_sub1}</div>
        </div>
        <div style="text-align:center">
          <div class="hc-score"><span style="color:{c1}">{score1}</span>
             <span style="color:#42506A">&ndash;</span>
             <span style="color:{c2}">{score2}</span></div>
          <div class="hc-vs" style="margin-top:6px">PROJECTED</div>
        </div>
        <div class="hc-team">
          {_logo(t2_id)}
          <div class="hc-team-name" style="color:{c2}">{name2}</div>
          <div class="hc-team-sub">{venue_sub2}</div>
        </div>
      </div>
      <div class="hc-bar">
        <div class="hc-bar-l" style="width:{bar_l_w}%;background:{c1}">{win1*100:.0f}%</div>
        <div class="hc-bar-r" style="width:{bar_r_w}%;background:{c2}">{win2*100:.0f}%</div>
      </div>
      <div style="display:flex;justify-content:space-between;color:#9AA4B2;font-size:.78rem;font-weight:600">
        <span>{name1} win probability</span><span>{name2} win probability</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(fav_name, fav_spread, fav_prob):
    st.markdown(
        f'<div class="hc-verdict"><b>{fav_name}</b> by '
        f'<b>{abs(fav_spread):.1f}</b> &nbsp;·&nbsp; {fav_prob*100:.0f}% to win</div>',
        unsafe_allow_html=True,
    )


def _edge_chip(diff, color, unit=""):
    sign = "+" if diff > 0 else ""
    return f'<span class="hc-edge" style="background:{color}22;color:{color}">{sign}{diff:.1f}{unit}</span>'


def render_factor(label, v1, v2, fmt="{:+.1f}", note="", higher_is_better=True):
    """One comparison row: team1 value | who has the edge | team2 value."""
    s1 = fmt.format(v1)
    s2 = fmt.format(v2)
    better1 = (v1 > v2) if higher_is_better else (v1 < v2)
    diff = v1 - v2
    edge_color = T1 if better1 else T2
    chip = _edge_chip(abs(diff) if better1 else -abs(diff), edge_color)
    note_html = f'<div class="hc-note">{note}</div>' if note else ""
    st.markdown(f"""
    <div class="hc-factor">
      <div class="v1" style="color:{T1 if better1 else INK}">{s1}</div>
      <div class="mid"><div class="lab">{label}</div>{chip}{note_html}</div>
      <div class="v2" style="color:{T2 if not better1 else INK}">{s2}</div>
    </div>
    """, unsafe_allow_html=True)


def card_open(title=None):
    head = f'<div class="hc-sec">{title}</div>' if title else ""
    st.markdown(f'<div class="hc-card">{head}', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)
