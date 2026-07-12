"""Generate scout.ipynb — run once: python coaches_suite/_make_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# 🏀 Coaches Suite — Opponent Scouting Report

A single-opponent scouting sheet built from Hog Charts shot + on/off data.

| # | View | Question it answers |
|---|------|---------------------|
| 1 | **Three-Point Threat Board** | Who is a real threat from deep? (tier + gravity) |
| 1b | **Who Runs the Offense** | How heliocentric are they? Who are the ball-handlers? (usage) |
| 1c | **Shot Diet at a Glance** | What % of shots are close / mid / three — offense & defense |
| 2 | **Offense — Shot Diet** | Where do they *like* to shoot? |
| 3 | **Offense — Shot Quality** | Where do they shoot *best*? (points per shot) |
| 3b | **Top Scorers + Recent Form** | Per-man shot charts, which way to force them, last-5 drift |
| 4 | **Defense — Shots Forced** | Where do they *force* opponents to shoot? |
| 5 | **Defense — Shots Allowed** | Where do opponents shoot *best* against them? |
| 5b | **Box-Out Board** | Who do we have to box out? (offensive-rebound rate) |
| 6 | **Who To Attack** | Which defender is the weak link? (on/off + foul trouble) |
| 🎯 | **Scouting Summary** | The one-screen game plan — all of the above, distilled |

Set the opponent below and run all. Everything is parametric — swap `TEAM` / `SEASON` to scout anyone.""")

co("""# ── CONFIG ─────────────────────────────────────────────────────────────
TEAM   = "Arkansas Razorbacks"   # fuzzy match; e.g. "Duke", "Houston Cougars"
SEASON = 2026                    # 2026 = 2025-26 season
ROOT   = ".."                    # repo root relative to coaches_suite/
""")

co("""import sys, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scout_lib as s
import court

pd.set_option("display.max_columns", 40)
plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15,
    "axes.axisbelow": True,
})

# palette ---------------------------------------------------------------------
INK, MUTED = "#1a1d21", "#6b7280"
TEAM_C = "#9D2235"
TIER_C = {"DON'T LEAVE": "#d1495b", "CLOSE OUT": "#edae49", "RESPECT": "#5b8a72",
          "SAG OFF": "#4a7ab5", "IGNORE": "#9aa0a6"}
DIET_CMAP = "PuOr"         # diverging: shot-share vs D-I avg (purple over / orange under)
EFF_CMAP  = "RdBu_r"       # diverging: PPS vs baseline (red hot / blue cold)
PPS_LIM   = 0.30           # symmetric colour limit for shot-quality (points-per-shot) charts
def dvlim(zone_df):        # symmetric colour limit for a diet chart's deltas
    return max(0.03, float(zone_df["delta"].abs().max()))
def tier_color(t): return TIER_C.get(t.replace(" *", ""), "#9aa0a6")
def surname(name):
    toks = [t for t in name.split() if t.rstrip(".") not in
            ("Jr", "Sr", "II", "III", "IV", "V")]
    return toks[-1] if toks else name

# jersey numbers — coaches & fans know guys by number, so tag every name -------
def jnum(jn):
    \"\"\"Clean a jersey number to a display string ('' if unknown).\"\"\"
    if jn is None: return ""
    s = str(jn).strip()
    return "" if s in ("", "nan", "None") else s
def pn(name, jn=None):
    \"\"\"'#21 D.J. Wagner' — number prefix when we have one, else just the name.\"\"\"
    j = jnum(jn)
    return f"#{j} {name}" if j else name
def jcol(df, name_col="name", jn_col="jn"):
    \"\"\"Prepend a '#' jersey column next to a name column for a display table.\"\"\"
    out = df.copy()
    out.insert(0, "#", out[jn_col].map(lambda v: jnum(v) or "—") if jn_col in out else "—")
    return out.drop(columns=[jn_col], errors="ignore")

# court-value builders --------------------------------------------------------
def diet_vals(zone_df):
    d = {}
    for _, r in zone_df.iterrows():
        d[r["zone"]] = {"c": r["delta"], "top": f"{r['share']*100:.0f}%",
                        "sub": f"{'+' if r['delta']>=0 else ''}{r['delta']*100:.0f} vs avg"}
    return d
def quality_vals(qdf):
    # Quality = points per shot vs an NCAA PPS baseline (rewards threes over
    # equal-FG% twos). Label shows the raw PPS; colour is PPS above/below baseline.
    d = {}
    for _, r in qdf.iterrows():
        pps = "—" if r["pps"] is None else f"{r['pps']:.2f}"
        if (not r["reliable"]) or r["pps_vs_base"] is None:
            d[r["zone"]] = {"c": None, "top": pps, "sub": f"{r['att']} att"}
        else:
            d[r["zone"]] = {"c": r["pps_vs_base"], "top": pps, "sub": f"{r['att']} att"}
    return d

data = s.load_data(SEASON, root=ROOT)
TID, NAME = s.resolve_team(data, TEAM)
TS  = s.team_shots(data, TID)     # their offense
OS_ = s.opp_shots(data, TID)      # the defense they played (opponent shots)
print(f"Scouting: {NAME}  (team_id={TID}, season {SEASON})")
print(f"  offense: {len(TS):,} FGA over {TS['game_id'].nunique()} games")
print(f"  defense: {len(OS_):,} opponent FGA faced")
""")

# ── 1. threat board ──────────────────────────────────────────────────────────
md("""## 1 · Three-Point Threat Board

Every rotation shooter gets a **defensive instruction tier** (volume + accuracy) and a
**0-100 gravity score** (league percentile: 55% volume, 45% accuracy). `*` = small sample (<15 3PA).

**DON'T LEAVE** never help off · **CLOSE OUT** run off the line · **RESPECT** contest, help ok ·
**SAG OFF** go under · **IGNORE** non-shooter.""")

co("""tp = s.three_point_threat(data, TID)
show = jcol(tp[["name","jn","tier","gravity","tpa_pg","tp_pct","tpa","tpm","games"]].copy())
show["tp_pct"] = (show["tp_pct"]*100).round(1)
show.columns = ["#","Player","Tier","Gravity","3PA/g","3P%","3PA","3PM","G"]
display(show.style.hide(axis="index")
        .background_gradient(cmap="Reds", subset=["Gravity"], vmin=0, vmax=100)
        .format({"Gravity":"{:.0f}","3PA/g":"{:.1f}","3P%":"{:.1f}"}))
""")

co("""# Form: volume x accuracy scatter — the natural read for 3pt threat.
plot = tp[tp["tpa"] >= 8].copy()
fig, ax = plt.subplots(figsize=(8.5, 5.6))
lg = s.LEAGUE_3P_PCT*100
ax.axhline(lg, color=MUTED, lw=1, ls="--", zorder=1)
ax.text(0.02, lg+0.4, "D-I avg 34.5%", color=MUTED, fontsize=9, transform=ax.get_yaxis_transform())
for _, r in plot.iterrows():
    ax.scatter(r["tpa_pg"], r["tp_pct"]*100, s=60+r["gravity"]*6,
               color=tier_color(r["tier"]), edgecolor="white", lw=1.5, zorder=3, alpha=0.9)
    ax.annotate(surname(r["name"]), (r["tpa_pg"], r["tp_pct"]*100), xytext=(7,4),
                textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("Volume — 3PA per game"); ax.set_ylabel("Accuracy — 3P%")
ax.set_title(f"{NAME} · deep-shooting threat  (bubble = gravity)", loc="left",
             fontsize=13, color=INK, weight="bold")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor=TIER_C[t],
          markersize=10, label=t) for t in TIER_C], loc="lower right",
          frameon=False, fontsize=8.5)
plt.tight_layout(); plt.show()
""")

# ── 1b. usage / ball-handling (heliocentrism) ────────────────────────────────
md("""## 1b · Who Runs the Offense — Usage & Ball-Handling

How **heliocentric** is the offense, and who are the primary ball-handlers? Rotation ranked by
**usage %** (share of possessions a player finishes on the floor). Read alongside:

- **AST%** — teammate FGs he creates. High USG + high AST = on-ball engine (blitz it); high USG +
  low AST = pure scorer (wall off, make others beat you).
- **Assisted%** — share of his *own* makes that were assisted. **Low = self-creator** (guard the
  drive), **high = spot-up finisher** (deny the catch).""")

co("""ub = s.usage_board(data, TID, NAME)
show = jcol(ub, name_col="n")
show.columns = ["#","Player","Pos","MPG","PPG","USG%","AST%","TOV%","Assisted%"]
display(show.style.hide(axis="index")
        .background_gradient(cmap="Purples", subset=["USG%"])
        .format({"MPG":"{:.1f}","PPG":"{:.1f}","USG%":"{:.1f}","AST%":"{:.1f}",
                 "TOV%":"{:.1f}","Assisted%":"{:.1f}"}, na_rep="—"))

top = ub.iloc[0]
top_share = ub["usg"].iloc[0] / max(ub["usg"].sum(), 1) * 100
kind = "creates his own looks" if (top.get("astdp") or 100) < 50 else "is mostly set up by others"
print(f"\\n▶ {pn(top['n'], top.get('jn'))} runs the show — {top['usg']:.0f}% usage "
      f"({top_share:.0f}% of the rotation's total), and {kind} "
      f"({top.get('astdp'):.0f}% of his makes assisted).")
""")

# ── 1c. general shot diet (close / mid / three, offense & defense) ────────────
md("""## 1c · Shot Diet at a Glance — Close · Mid · Three

The simplest read: what share of shots come from the three broad areas — **Close/Paint** (rim + close
mid), **Mid-Range**, and **Three** — both for **their offense** (where they shoot) and for the
**shots opponents take against them** (what their defense gives up). The `vs D-I` columns are the
gap to the Division-I average (**+ = more than a typical team, − = less**).""")

co("""diet_off = s.macro_frequency(TS, data)     # their offense
diet_def = s.macro_frequency(OS_, data)    # what opponents shoot vs. them
_lab = {"Paint / Restricted": "Close / Paint", "Mid-Range": "Mid-Range", "Three": "Three"}
diet = pd.DataFrame({
    "Area":          [_lab[z] for z in diet_off["zone"]],
    "Offense":       (diet_off["share"]*100).round(0),
    "Off vs D-I":    (diet_off["delta"]*100).round(0),
    "Defense (opp)": (diet_def["share"]*100).round(0),
    "Def vs D-I":    (diet_def["delta"]*100).round(0),
    "D-I avg":       (diet_off["lg_share"]*100).round(0),
})
print("SHOT DIET — share of field-goal attempts by area")
display(diet.style.hide(axis="index")
        .background_gradient(cmap=DIET_CMAP, subset=["Off vs D-I","Def vs D-I"], vmin=-12, vmax=12)
        .format({"Offense":"{:.0f}%","Defense (opp)":"{:.0f}%","D-I avg":"{:.0f}%",
                 "Off vs D-I":"{:+.0f}","Def vs D-I":"{:+.0f}"}))

def _g(df, area, col): return df.loc[df["Area"] == area, col].iloc[0]
print(f"\\n▶ OFFENSE: {_g(diet,'Close / Paint','Offense'):.0f}% close · "
      f"{_g(diet,'Mid-Range','Offense'):.0f}% mid · {_g(diet,'Three','Offense'):.0f}% three "
      f"(D-I avg {_g(diet,'Three','D-I avg'):.0f}% from deep).")
print(f"▶ DEFENSE: opponents take {_g(diet,'Close / Paint','Defense (opp)'):.0f}% close · "
      f"{_g(diet,'Mid-Range','Defense (opp)'):.0f}% mid · "
      f"{_g(diet,'Three','Defense (opp)'):.0f}% three against them.")
""")

# ── 2/3. offense courts ──────────────────────────────────────────────────────
md("""## 2 · 3 · Offense — Shot Diet & Shot Quality

Same 14-zone court as the website. **Left: where they shoot** — colored by how their shot-share in
each zone compares to the D-I average (**purple = they shoot here more than a typical team, orange =
less**; the label shows the raw share and the +/- vs average). **Right: how well they shoot** —
**points per shot** vs the NCAA baseline for that zone (FG% x point value, so a hot three counts
for more than an equally-accurate two — red = above / a strength to take away, blue = below, hatched
= too few attempts).

Read them together: the danger spots are **purple on the left AND red on the right**.""")

co("""oq = s.zone_quality(TS, root=ROOT)
od = s.zone_frequency(TS, data)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2))
court.draw_zone_court(a1, diet_vals(od["zone"]), scheme="diverging", cmap=DIET_CMAP,
    vmin=-dvlim(od["zone"]), vmax=dvlim(od["zone"]),
    title="Shot Diet — where they shoot", cbar_label="shot-share vs D-I avg",
    note=f"{od['n_fga']:,} FGA · purple = more / orange = less than avg")
court.draw_zone_court(a2, quality_vals(oq), scheme="diverging", cmap=EFF_CMAP,
    vmin=-PPS_LIM, vmax=PPS_LIM, title="Shot Quality — how well they shoot",
    cbar_label="points per shot vs NCAA baseline", note="label = PPS (attempts)")
fig.suptitle(f"{NAME} · OFFENSE", fontsize=15, weight="bold", color=TEAM_C, y=0.99)
plt.tight_layout(); plt.show()
""")

co("""# Priority: zones they shoot OFTEN and score ABOVE baseline PPS (take these away first).
m = od["zone"][["zone","share"]].merge(oq[["zone","pps","pps_vs_base","att"]], on="zone")
m = m[(m["att"] >= 10) & (m["pps_vs_base"].notna())].copy()
m["danger"] = (m["share"]*100) * m["pps_vs_base"].clip(lower=0)
top = m.sort_values("danger", ascending=False).head(6)[["zone","share","pps","pps_vs_base"]].copy()
top["share"] = (top["share"]*100).round(0)
top["pps"] = top["pps"].round(2); top["pps_vs_base"] = top["pps_vs_base"].round(2)
top.columns = ["Zone","% of shots","PPS","vs baseline"]
print("PRIORITY — spots they shoot often AND score above baseline (PPS):")
display(top.style.hide(axis="index").format({"PPS":"{:.2f}","vs baseline":"{:+.2f}"}))
""")

# ── macro (3-band) view ──────────────────────────────────────────────────────
md("""### Macro view — Paint/Restricted · Mid-Range · Three

The offense and defense reads collapsed into three broad areas. First **shot diet** (colored by
shot-share vs the D-I average — **purple = more than a typical team, orange = less**; label shows
raw share + the +/-), then **shot quality** (points per shot vs an attempt-weighted NCAA baseline
for that band). Left = offense (what *they* do), right = defense (what *opponents* do against them —
red = a hole).""")

co("""mod = s.macro_frequency(TS, data)        # offense diet
mdd = s.macro_frequency(OS_, data)       # defense: shots forced
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2))
mvl = max(dvlim(mod), dvlim(mdd))        # shared limit so the two panels compare
court.draw_zone_court(a1, diet_vals(mod), scheme="diverging", cmap=DIET_CMAP,
    vmin=-mvl, vmax=mvl, wedges=court.MACRO_WEDGES, label_size=13,
    title=f"{NAME} · OFFENSE diet by area", cbar_label="shot-share vs D-I avg",
    note="label = share (+/- vs avg) · purple = more / orange = less")
court.draw_zone_court(a2, diet_vals(mdd), scheme="diverging", cmap=DIET_CMAP,
    vmin=-mvl, vmax=mvl, wedges=court.MACRO_WEDGES, label_size=13,
    title=f"{NAME} · DEFENSE: shots forced by area", cbar_label="opp shot-share vs D-I avg",
    note="where they push opponents vs. a typical defense")
plt.tight_layout(); plt.show()
""")

co("""moq = s.macro_quality(TS, root=ROOT)     # offense
mdq = s.macro_quality(OS_, root=ROOT)    # defense faced
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2))
court.draw_zone_court(a1, quality_vals(moq), scheme="diverging", cmap=EFF_CMAP,
    vmin=-PPS_LIM, vmax=PPS_LIM, wedges=court.MACRO_WEDGES, label_size=13,
    title=f"{NAME} · OFFENSE by area", cbar_label="points per shot vs NCAA baseline",
    note="label = PPS (attempts)")
court.draw_zone_court(a2, quality_vals(mdq), scheme="diverging", cmap=EFF_CMAP,
    vmin=-PPS_LIM, vmax=PPS_LIM, wedges=court.MACRO_WEDGES, label_size=13,
    title=f"{NAME} · DEFENSE by area", cbar_label="opp points per shot vs baseline",
    note="red = opponents beat them there")
plt.tight_layout(); plt.show()
""")

# ── 3b. individual shot charts (top scorers) + directional read ──────────────
md("""### Top Scorers — individual shot quality & which way to force them

Their top six scorers, each on the same PPS-vs-baseline court (red = a spot they punish, blue =
one they don't, hatched = <5 attempts). Under each name is their **points-per-shot by court side**
— if a scorer is far better going one way, **force him to the other** ('force him weak'). Sample
sizes are smaller per player, so read the volume zones, not the fringes.""")

co("""scorers = s.top_scorers(data, TID, NAME, n=6)
fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
def _pps(dd, sd):
    v = dd.loc[dd["side"] == sd, "pps"].iloc[0]
    return "—" if v is None else f"{v:.2f}"
for ax, (_, p) in zip(axes.flat, scorers.iterrows()):
    psh = s.player_shots(data, p["id"])
    pq  = s.zone_quality(psh, root=ROOT, min_att=5)
    dsp = s.directional_split(psh)
    court.draw_zone_court(ax, quality_vals(pq), scheme="diverging", cmap=EFF_CMAP,
        vmin=-PPS_LIM, vmax=PPS_LIM, cbar=False, label_size=8,
        title=f"{pn(surname(p['n']), p.get('jn'))} · {p['ppg']:.1f} ppg",
        note=f"PPS by side  L {_pps(dsp,'Left')} · C {_pps(dsp,'Center')} · R {_pps(dsp,'Right')}   ({len(psh)} FGA)")
for ax in list(axes.flat)[len(scorers):]:
    ax.axis("off")
fig.suptitle(f"{NAME} · TOP SCORERS — shot quality (PPS vs baseline) + side split",
             fontsize=15, weight="bold", color=TEAM_C, y=0.99)
plt.tight_layout(); plt.show()
""")

# ── 3c. recent form ──────────────────────────────────────────────────────────
md("""### Recent Form — season vs. last 5 games

Has the shot profile drifted? Shot mix and points-per-shot by area, season-long vs. the **last 5
games** (injuries, role changes and cold streaks make an early-season scout stale). Watch the
**Δ PPS** and **Δ share** columns — a team leaning harder into (or away from) an area lately, or
suddenly hot/cold there, is the current book on them.""")

co("""form = s.form_summary(TS, data, n_games=5, root=ROOT)
ft = form["table"].copy()
for c in ["share","share_recent","share_delta"]:
    ft[c] = (ft[c]*100).round(0)
ft = ft[["band","share","share_recent","share_delta","pps","pps_recent","pps_delta"]]
ft.columns = ["Area","Share%","Share% L5","Δ share","PPS","PPS L5","Δ PPS"]
print(f"Season vs last {form['n_recent_games']} games:")
display(ft.style.hide(axis="index")
        .background_gradient(cmap="RdBu_r", subset=["Δ PPS"], vmin=-0.15, vmax=0.15)
        .format({"Share%":"{:.0f}","Share% L5":"{:.0f}","Δ share":"{:+.0f}",
                 "PPS":"{:.2f}","PPS L5":"{:.2f}","Δ PPS":"{:+.2f}"}, na_rep="—"))
""")

# ── 4/5. defense courts ──────────────────────────────────────────────────────
md("""## 4 · 5 · Defense — Shots Forced & Shots Allowed

The defense they play, from **opponents' shots in their games**. **Left: where they force shots**
(colored by opponent shot-share vs the D-I average — **purple = opponents shoot here more than
against a typical defense, orange = less**).
**Right: where opponents shoot best** against them (opponent points per shot vs baseline — **red = a
hole in their defense**, opponents beat them there; blue = they lock that spot down).

A good defense pushes opponents into blue zones and forces low-value shots.""")

co("""dq = s.zone_quality(OS_, root=ROOT)
dd = s.zone_frequency(OS_, data)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2))
court.draw_zone_court(a1, diet_vals(dd["zone"]), scheme="diverging", cmap=DIET_CMAP,
    vmin=-dvlim(dd["zone"]), vmax=dvlim(dd["zone"]),
    title="Shots Forced — where they push opponents", cbar_label="opp shot-share vs D-I avg",
    note=f"{dd['n_fga']:,} opp attempts · purple = more / orange = less than avg")
court.draw_zone_court(a2, quality_vals(dq), scheme="diverging", cmap=EFF_CMAP,
    vmin=-PPS_LIM, vmax=PPS_LIM, title="Shots Allowed — where opponents beat them",
    cbar_label="opp points per shot vs baseline", note="red = hole in the defense · label = PPS (att)")
fig.suptitle(f"{NAME} · DEFENSE", fontsize=15, weight="bold", color=INK, y=0.99)
plt.tight_layout(); plt.show()
""")

co("""# Where to hunt offense: opponents already beat them here (above baseline, real volume).
mm = dd["zone"][["zone","share"]].merge(dq[["zone","pps","pps_vs_base","att"]], on="zone")
mm = mm[(mm["att"] >= 10) & (mm["pps_vs_base"].notna())].copy()
weak = mm.sort_values("pps_vs_base", ascending=False).head(6)[["zone","share","pps","pps_vs_base"]].copy()
weak["share"] = (weak["share"]*100).round(0)
weak["pps"] = weak["pps"].round(2); weak["pps_vs_base"] = weak["pps_vs_base"].round(2)
weak.columns = ["Zone","% of opp shots","Opp PPS","vs baseline"]
print("ATTACK — spots opponents already score above baseline (PPS) against them:")
display(weak.style.hide(axis="index").format({"Opp PPS":"{:.2f}","vs baseline":"{:+.2f}"}))
""")

# ── 5b. box-out board ─────────────────────────────────────────────────────────
md("""## 5b · Box-Out Board — who crashes the glass

The players you have to **box out**, ranked by **offensive-rebound %** (share of their team's
misses they grab). **ORB% / DRB% / TRB%** are rate stats (per available rebound), so they're fair
across minutes; **Reb/100** is raw volume — total rebounds per 100 on-court possessions. High ORB%
bigs are the box-out assignment; a guard with real ORB% is a sneaky crasher to account for in
transition.""")

co("""rb = s.rebound_board(data, TID, NAME)
show = jcol(rb[["name","jn","pos","ht","gp","mpg","orbp","drbp","trbp","reb","reb_per100"]].copy())
show.columns = ["#","Player","Pos","Ht","G","MPG","ORB%","DRB%","TRB%","Reb","Reb/100"]
display(show.style.hide(axis="index")
        .background_gradient(cmap="Greens", subset=["ORB%"])
        .background_gradient(cmap="Blues", subset=["DRB%"])
        .format({"MPG":"{:.1f}","ORB%":"{:.1f}","DRB%":"{:.1f}","TRB%":"{:.1f}",
                 "Reb":"{:.0f}","Reb/100":"{:.1f}"}, na_rep="—"))

r0 = rb.iloc[0]
print(f"\\n▶ BOX OUT FIRST: {pn(r0['name'], r0.get('jn'))} ({r0['pos']}"
      f"{', '+str(r0['ht']) if pd.notna(r0['ht']) else ''})"
      f" — {r0['orbp']:.1f}% offensive-rebound rate.")
""")

# ── 6. attack board ──────────────────────────────────────────────────────────
md("""## 6 · Who To Attack — the defensive weak links

Rotation players (≥10 mpg, ≥150 defensive possessions) ranked by **defensive on/off**:

> `def on/off = drtg_on − drtg_off` — points allowed per 100 possessions with the player **on**
> the floor minus **off**. Positive = the team's defense gets **worse** with them on court → hunt them.

`drtg_on` is the raw points-allowed rate while they're on. This comes from box/PBP on-off, which is
**ready at the start of the season** (unlike RAPM, which needs a stabilising sample). Height / usage
are context only.""")

co("""atk = s.attack_board(data, TID, NAME)
show = jcol(atk[["name","jn","def_onoff","drtg_on","drtg_off","pos","ht","mpg","usg","def_poss"]].copy())
show.columns = ["#","Player","Def On/Off","DRtg On","DRtg Off","Pos","Ht","MPG","USG%","Def Poss"]
display(show.style.hide(axis="index")
        .background_gradient(cmap="Reds", subset=["Def On/Off"])
        .background_gradient(cmap="Reds", subset=["DRtg On"])
        .format({"Def On/Off":"{:+.1f}","DRtg On":"{:.1f}","DRtg Off":"{:.1f}",
                 "MPG":"{:.1f}"}, na_rep="—"))

t = atk.iloc[0]
print(f"\\n▶ PRIMARY TARGET: {pn(t['name'], t.get('jn'))} ({t['pos']}"
      f"{', '+str(t['ht']) if pd.notna(t['ht']) else ''})"
      f" — defense allows {t['drtg_on']:.1f}/100 with him on,"
      f" {t['def_onoff']:+.1f} worse than off.")
""")

# ── 6b. foul-trouble sheet ────────────────────────────────────────────────────
md("""### Foul-Trouble Board — who to hunt for cheap fouls

Rotation ranked by **personal fouls per 100 defensive possessions** (from full play-by-play). A
foul-prone defender you can attack early — get him two in the first half and change their rotation.
Pair it with the on/off board above: a high-foul defender who's *also* a weak link is the guy to go
at.""")

co("""fb = s.foul_board(data, TID, NAME)
if fb.empty:
    print("No foul data for this season (offline_pbp.csv is current-season only).")
else:
    show = jcol(fb[["name","jn","pos","fouls_per100","fouls","def_poss","mpg"]].copy())
    show.columns = ["#","Player","Pos","Fouls/100 def","Fouls","Def Poss","MPG"]
    display(show.style.hide(axis="index")
            .background_gradient(cmap="Reds", subset=["Fouls/100 def"])
            .format({"Fouls/100 def":"{:.2f}","MPG":"{:.1f}"}))
    f0 = fb.iloc[0]
    print(f"\\n▶ FOUL-TROUBLE TARGET: {pn(f0['name'], f0.get('jn'))} — {f0['fouls_per100']:.1f} fouls/100 def poss; "
          f"go at him early.")
""")

md("""## 🎯 Scouting Summary — the one-screen read

Everything above distilled to the game-plan bullets: who to guard, what to take away, and where to
attack. Players are tagged **`#number Name`** — coaches and fans know guys by their jersey. Run all
cells first (the summary reads the boards you just built).""")

co("""# One-screen game plan — aggregates every board above into coach-language bullets.
def _sh(r):   # shooter line: "#21 D.J. Wagner (39% on 6.2/g)"
    return f"{pn(r['name'], r.get('jn'))} ({r['tp_pct']*100:.0f}% on {r['tpa_pg']:.1f}/g)"
def _zone_line(r, opp=False):
    unit = "of opp shots" if opp else "of shots"
    return (f"{r['zone']}  —  {r['share']*100:.0f}% {unit}, "
            f"{r['pps']:.2f} PPS ({r['pps_vs_base']:+.2f} vs base)")

# offense danger spots — shoot often AND above baseline
_om = od["zone"][["zone","share"]].merge(oq[["zone","pps","pps_vs_base","att"]], on="zone")
_om = _om[(_om["att"] >= 10) & (_om["pps_vs_base"].notna())].copy()
_om["danger"] = (_om["share"]*100) * _om["pps_vs_base"].clip(lower=0)
off_spots = _om[_om["danger"] > 0].sort_values("danger", ascending=False).head(3)

# defensive holes — opponents already score above baseline here
_dm = dd["zone"][["zone","share"]].merge(dq[["zone","pps","pps_vs_base","att"]], on="zone")
_dm = _dm[(_dm["att"] >= 10) & (_dm["pps_vs_base"].notna())].copy()
def_holes = _dm[_dm["pps_vs_base"] > 0].sort_values("pps_vs_base", ascending=False).head(3)

# recent-form drift — biggest PPS swing over the last 5
ftab = form["table"].dropna(subset=["pps_delta"]).copy()
drift = ftab.reindex(ftab["pps_delta"].abs().sort_values(ascending=False).index).head(1)

bar = "═" * 68
print(bar)
print(f" SCOUTING SUMMARY — {NAME}")
print(f" {SEASON-1}-{str(SEASON)[2:]} season · {len(TS):,} FGA over {TS['game_id'].nunique()} games")
print(bar)

# ── OFFENSE: how we guard them ───────────────────────────────────────────────
print("\\n▌ OFFENSE — how we guard them")
dont  = tp[tp["tier"].str.startswith("DON'T LEAVE")]
close = tp[tp["tier"].str.startswith("CLOSE OUT")]
if len(dont):
    print("  • Never leave from three:  " + ",  ".join(_sh(r) for _, r in dont.iterrows()))
if len(close):
    print("  • Close out hard:  " + ",  ".join(_sh(r) for _, r in close.iterrows()))
if not len(dont) and not len(close):
    print("  • No high-end shooters to chase off the line — help freely.")

eng = ub.iloc[0]
_kind = ("creates his own looks — wall off the drive" if (eng.get("astdp") or 100) < 50
         else "is set up by others — deny the catch")
print(f"  • Runs the offense:  {pn(eng['n'], eng.get('jn'))} — {eng['usg']:.0f}% usage; {_kind}.")

sc = ",  ".join(f"{pn(surname(r['n']), r.get('jn'))} {r['ppg']:.0f}" for _, r in scorers.iterrows())
print(f"  • Score sheet (ppg):  {sc}")

if len(off_spots):
    print("  • Take away first (shoot here often AND make it):")
    for _, r in off_spots.iterrows():
        print("       – " + _zone_line(r))

if len(drift) and abs(drift.iloc[0]["pps_delta"]) >= 0.05:
    d = drift.iloc[0]
    arrow = "UP" if d["pps_delta"] > 0 else "DOWN"
    print(f"  • Trending: {d['band']} shot-making is {arrow} {abs(d['pps_delta']):.2f} PPS "
          f"over the last {form['n_recent_games']} games — the current book on them.")

# ── DEFENSE: how we attack them ──────────────────────────────────────────────
print("\\n▌ DEFENSE — how we attack them")
if len(def_holes):
    print("  • Hunt these spots (opponents already beat them here):")
    for _, r in def_holes.iterrows():
        print("       – " + _zone_line(r, opp=True))

if len(atk):
    a = atk.iloc[0]
    print(f"  • Primary target:  {pn(a['name'], a.get('jn'))} ({a['pos']}) — "
          f"defense allows {a['drtg_on']:.0f}/100 with him on, {a['def_onoff']:+.1f} vs off.")
if len(fb):
    f0 = fb.iloc[0]
    print(f"  • Foul-trouble target:  {pn(f0['name'], f0.get('jn'))} — "
          f"{f0['fouls_per100']:.1f} fouls/100 def poss; go at him early.")
if len(rb):
    r0 = rb.iloc[0]
    print(f"  • Box out first:  {pn(r0['name'], r0.get('jn'))} ({r0['pos']}) — "
          f"{r0['orbp']:.1f}% offensive-rebound rate.")
print(bar)
""")

md("""---
*Coaches Suite prototype · Hog Charts. Engine in `scout_lib.py`, court renderer in `court.py`.
Regenerate with `python coaches_suite/_make_notebook.py`. Scout any team by changing `TEAM` /
`SEASON` at the top.*""")

nb["cells"] = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"}}
with open("coaches_suite/scout.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote coaches_suite/scout.ipynb")
