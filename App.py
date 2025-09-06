# coronamax-public/app.py
# ------------------------------------------------------------
# CoronaMax (Lecture seule) – Streamlit
# - Tableau général (avec ligne Winamax en haut)
# - Détails joueur
# - Archives (classement "à la date" + PDFs par saison)
# - Classement par points
#
# Données lues depuis:
#   ./data/results_log.csv     (historique ligne-par-ligne)
#   ./pdfs/*.pdf               (PDFs archivés)
#   ./assets/coronamax_logo.png (optionnel)
# ------------------------------------------------------------
from __future__ import annotations

import base64
import io
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Tuple, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Paths (relatifs au repo public) ----------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdfs"
ASSETS = ROOT / "assets"
RESULTS_LOG = DATA_DIR / "results_log.csv"

# ----------------- Page config -----------------
st.set_page_config(page_title="CoronaMax (Lecture seule)", page_icon="🏆", layout="wide")

# ----------------- Logo optionnel -----------------
logo_cols = st.columns([1, 8])
with logo_cols[0]:
    logo = ASSETS / "coronamax_logo.png"
    if logo.exists():
        st.image(str(logo), width=90)

st.title("CoronaMax — Classements (lecture seule)")

# ============================================================
# Helpers généraux
# ============================================================
EURO_RE = re.compile(r"[^\d,.\-]+")

def parse_money(x) -> float:
    """'35,10 €' -> 35.10 ; robustesse sur None / '' / nombres."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x)
    s = EURO_RE.sub("", s).replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

def euro(v: float) -> str:
    return f"{float(v):.2f} €".replace(".", ",")

def list_files_sorted(folder: Path, patterns: Iterable[str] = ("*.pdf",)) -> List[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(folder.glob(pat))
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

def pick_series(df: pd.DataFrame, candidates: list[str], default=None) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)

# ============================================================
# Saisons & filtres
# ============================================================
def season_bounds_for(dt: date) -> Tuple[date, date]:
    """Saison: 01-08 (année N) -> 31-07 (année N+1)."""
    y = dt.year
    start = date(y, 8, 1) if dt >= date(y, 8, 1) else date(y - 1, 8, 1)
    end = date(start.year + 1, 7, 31)
    return start, end

def seasons_available(log: pd.DataFrame) -> list[Tuple[str, date, date]]:
    if log.empty:
        today = date.today()
        s0, s1 = season_bounds_for(today)
        label = f"Saison {s0.year}-{s1.year}"
        return [(label, s0, s1)]
    dt = pick_series(log, ["start_time", "processed_at"], None)
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce")
    dmin = dt.min().date()
    dmax = dt.max().date()

    # remonte et descend pour couvrir toutes les saisons
    s = season_bounds_for(dmin)
    seasons: list[Tuple[str, date, date]] = []
    cur_start, cur_end = s
    while cur_start <= dmax:
        label = f"Saison {cur_start.year}-{cur_end.year}"
        seasons.append((label, cur_start, cur_end))
        cur_start = date(cur_start.year + 1, 8, 1)
        cur_end = date(cur_end.year + 1, 7, 31)
    seasons.reverse()  # la plus récente d'abord
    return seasons

def pick_season(log: pd.DataFrame) -> Tuple[date, date, str]:
    opts = seasons_available(log)
    labels = [lab for (lab, _, _) in opts]
    choice = st.selectbox("Saison", labels, index=0, key="season_select")
    s0, s1 = next((a, b) for (lab, a, b) in opts if lab == choice)
    return s0, s1, choice

# ============================================================
# Chargement du log (lecture seule)
# ============================================================
@st.cache_data(show_spinner=False)
def load_results_log() -> pd.DataFrame:
    if not RESULTS_LOG.exists() or RESULTS_LOG.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_LOG)
    # Normalisations colonnes temps
    for col in ("start_time", "processed_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Colonnes attendues (fallback robustes)
    # - Pseudo, Position, GainCash|GainsCash, Bounty, Reentry|Recaves, buyin_total
    # - tournament_id (pour grouper), tournament_name
    return df

# ============================================================
# Classement (gains / frais)
# ============================================================
def standings_from_log(log: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le classement "gains/frais" (avec Winamax en tête).
    Colonnes finales: Place, Pseudo, Parties, Victoires, ITM, % ITM,
                      Recaves, Recaves en €, Bulles, Buy in, Frais,
                      Gains, Bénéfices
    """
    if log.empty:
        return pd.DataFrame(columns=[
            "Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
            "Bulles","Buy in","Frais","Gains","Bénéfices"
        ])

    df = log.copy()

    # Colonnes robustes
    pseudo  = pick_series(df, ["Pseudo"], "").astype(str)
    pos     = pd.to_numeric(pick_series(df, ["Position","Place","Rank"], 0), errors="coerce").fillna(0).astype(int)
    gain    = pick_series(df, ["GainCash","GainsCash","Gain","Gains"], 0.0).apply(parse_money)
    bounty  = pick_series(df, ["Bounty"], 0.0).apply(parse_money)
    reentry = pd.to_numeric(pick_series(df, ["Reentry","Recaves"], 0), errors="coerce").fillna(0).astype(int)
    buyin   = pick_series(df, ["buyin_total","Buyin","Buy-in"], 10.0).apply(parse_money)
    t_id    = pick_series(df, ["tournament_id"], "")

    # Agrégats par joueur
    base = pd.DataFrame({
        "Pseudo": pseudo,
        "Position": pos,
        "GainCash": gain,
        "Bounty": bounty,
        "Reentry": reentry,
        "buyin_total": buyin,
        "tournament_id": t_id,
    })

    # Parties: nb de lignes (1 ligne = 1 tournoi)
    parties = base.groupby("Pseudo")["Pseudo"].count().rename("Parties")
    victoires = base.groupby("Pseudo")["Position"].apply(lambda s: int((s == 1).sum())).rename("Victoires")
    itm = base.groupby("Pseudo")["GainCash"].apply(lambda s: int((s > 0).sum())).rename("ITM")
    recaves = base.groupby("Pseudo")["Reentry"].sum().rename("Recaves")
    gains = base.groupby("Pseudo").apply(lambda d: float((d["GainCash"] + d["Bounty"]).sum())).rename("Gains")
    # Buy-in: somme des buyin_total (1 par tournoi/joueur)
    buy_in = base.groupby("Pseudo")["buyin_total"].sum().rename("Buy in")
    # Recaves en € = somme(Reentry * buyin_total) ligne-à-ligne
    base["recaves_euro_row"] = base["Reentry"] * base["buyin_total"]
    recaves_euro = base.groupby("Pseudo")["recaves_euro_row"].sum().rename("Recaves en €")

    # Bulles: on recalcule par tournoi le 1er joueur non payé (GainCash<=0)
    bulles = pd.Series(0, index=parties.index, dtype=int)
    for tid, d in base.groupby("tournament_id"):
        d = d.sort_values("Position")
        no_paid = d[d["GainCash"] <= 0.0]
        if not no_paid.empty:
            bulle_pseudo = str(no_paid.iloc[0]["Pseudo"])
            if bulle_pseudo in bulles.index:
                bulles.at[bulle_pseudo] = int(bulles.get(bulle_pseudo, 0)) + 1

    out = pd.concat([parties, victoires, itm, recaves, recaves_euro, bulles, buy_in, gains], axis=1).fillna(0)
    out["% ITM"] = out.apply(lambda r: f"{(r['ITM']/r['Parties']):.0%}" if r["Parties"] > 0 else "0%", axis=1)
    out["Frais"] = out["Buy in"] + out["Recaves en €"]
    out["Bénéfices"] = out["Gains"] - out["Frais"]

    # Winamax en tête
    total_frais_autres = out["Frais"].sum()
    winamax = pd.DataFrame([{
        "Pseudo": "WINAMAX",
        "Parties": int(base["tournament_id"].nunique()),
        "Victoires": 0, "ITM": 0, "% ITM": "0%",
        "Recaves": 0, "Recaves en €": 0.0, "Bulles": 0,
        "Buy in": 0.0, "Frais": 0.0,
        "Gains": round(total_frais_autres * 0.10, 2),
        "Bénéfices": round(total_frais_autres * 0.10, 2),
    }]).set_index("Pseudo")

    out = pd.concat([winamax, out], axis=0, sort=False).fillna(0.0)

    # Tri: Winamax d'abord, puis par Bénéfices desc
    def sort_key(s: pd.Series):
        return s.apply(parse_money) if s.dtype == object else s

    out = out.reset_index()
    out["is_winamax"] = out["Pseudo"].str.upper().eq("WINAMAX")
    out = pd.concat([
        out[out["is_winamax"]],
        out[~out["is_winamax"]].sort_values(by="Bénéfices", key=sort_key, ascending=False)
    ]).reset_index(drop=True)
    out.insert(0, "Place", list(range(len(out))))
    out.drop(columns=["is_winamax"], inplace=True)

    # Formattage arrondi (nombre) — l'affichage € se fait plus tard
    for col in ("Recaves en €", "Buy in", "Frais", "Gains", "Bénéfices"):
        out[col] = out[col].astype(float)

    return out[[
        "Place","Pseudo","Parties","Victoires","ITM","% ITM",
        "Recaves","Recaves en €","Bulles","Buy in","Frais","Gains","Bénéfices"
    ]]

# ============================================================
# Style DataFrame (UI)
# ============================================================
def style_dataframe(d: pd.DataFrame) -> pd.io.formats.style.Styler:
    df = d.copy()

    # Entiers
    for c in ("Place","Parties","Victoires","ITM","Recaves","Bulles"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # € en string
    for c in ("Recaves en €","Buy in","Frais","Gains","Bénéfices"):
        if c in df.columns:
            df[c] = df[c].apply(euro)

    sty = df.style.set_properties(**{"text-align": "center", "font-size": "0.95rem"})
    # en-têtes
    sty = sty.set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#e6e6e6"), ("font-weight", "bold"), ("color", "#000")]
    }])

    # Pseudos orange gras
    if "Pseudo" in df.columns:
        sty = sty.set_properties(subset=["Pseudo"], **{
            "background-color": "#f7b329", "font-weight": "bold", "color": "#000"
        })

    # Parties rouge (<20) / vert (≥20)
    if "Parties" in df.columns:
        mask_red = df["Parties"] < 20
        mask_green = df["Parties"] >= 20
        sty = sty.apply(lambda s: ["color:#d00000" if v else "" for v in mask_red], subset=["Parties"])
        sty = sty.apply(lambda s: ["color:#107a10" if v else "" for v in mask_green], subset=["Parties"])

    # Dégradé sur Bénéfices
    if "Bénéfices" in df.columns:
        vals = df["Bénéfices"].apply(parse_money)
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = (vmax - vmin) if vmax != vmin else 1.0

        def grad(v):
            t = (parse_money(v) - vmin) / rng
            hue = int(120 * t)  # 0 rouge -> 120 vert
            return f"background-color:hsl({hue},65%,82%); font-weight:700;"
        sty = sty.apply(lambda col: [grad(v) for v in df["Bénéfices"]], subset=["Bénéfices"])

    # Ligne Winamax: blanc sur rouge + uppercase + faux "League Gothic" fallback
    try:
        idx = df.index[df["Pseudo"].str.upper() == "WINAMAX"][0]
        sty = sty.set_table_styles([{
            "selector": f"tbody tr:nth-child({idx+2}) td:nth-child(2)",
            "props": [
                ("background-color", "#d00000"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "0.03em")
            ],
        }], overwrite=False)
        # ligne pointillée sous Winamax
        sty = sty.set_table_styles([{
            "selector": f"tbody tr:nth-child({idx+2})",
            "props": [("border-bottom","3px dashed #222")]
        }], overwrite=False)
    except Exception:
        pass

    return sty

def show_table(df: pd.DataFrame, caption: str | None = None):
    st.dataframe(style_dataframe(df), width="stretch", hide_index=True)
    if caption:
        st.caption(caption)

# ============================================================
# Export JPG du tableau affiché
# ============================================================
def classement_df_to_jpg(df: pd.DataFrame, out_file: Path, title: str = "Classement"):
    fig, ax = plt.subplots(figsize=(14, 0.6 + 0.45 * (len(df) + 1)))
    ax.axis("off")

    # formatage € pour l’export
    d = df.copy()
    for c in ("Recaves en €","Buy in","Frais","Gains","Bénéfices"):
        if c in d.columns:
            d[c] = d[c].apply(lambda x: euro(parse_money(x)))

    tbl = ax.table(
        cellText=d.values,
        colLabels=d.columns,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)

    # en-têtes gris
    for j in range(d.shape[1]):
        tbl[(0, j)].set_facecolor("#e6e6e6")
        tbl[(0, j)].set_text_props(weight="bold")

    # mise en forme "Winamax"
    if "Pseudo" in d.columns:
        try:
            row_idx = d.index[d["Pseudo"].str.upper() == "WINAMAX"][0]
            # +1 car 0 = header
            r = row_idx + 1
            # pseudo = col 1
            tbl[(r, 1)].set_facecolor("#d00000")
            tbl[(r, 1)].set_text_props(color="white", weight="bold")
        except Exception:
            pass

    plt.title(title, fontsize=14, pad=12)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Classement par points
# ============================================================
def compute_points_table(log: pd.DataFrame, d1: date, d2: date) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

    df = log.copy()
    dt = pick_series(df, ["start_time","processed_at"], None)
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce")
    df = df[(dt.dt.date >= d1) & (dt.dt.date <= d2)].copy()
    if df.empty:
        return pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

    pseudo = pick_series(df, ["Pseudo"], "").astype(str)
    pos = pd.to_numeric(pick_series(df, ["Position","Place","Rank"], 0), errors="coerce").fillna(0).astype(int)
    tid = pick_series(df, ["tournament_id"], "").astype(str)
    gaincash = pick_series(df, ["GainCash","GainsCash","Gain","Gains"], 0.0).apply(parse_money)

    # participants par tournoi
    n_part = df.groupby(tid).transform("size")

    points = (n_part - pos + 1).clip(lower=1)
    itm_flag = (gaincash > 0).astype(int)
    win_flag = (pos == 1).astype(int)

    tmp = pd.DataFrame({
        "Pseudo": pseudo,
        "Points": points,
        "ITM": itm_flag,
        "Victoires": win_flag,
        "Parties": 1,
    })
    tmp = tmp[~tmp["Pseudo"].str.upper().eq("WINAMAX")]

    agg = tmp.groupby("Pseudo", as_index=False).agg(
        Parties=("Parties","sum"),
        ITM=("ITM","sum"),
        Victoires=("Victoires","sum"),
        Points=("Points","sum"),
    )

    agg = agg.sort_values(by=["Points","Victoires","ITM","Parties"], ascending=[False, False, False, False]).reset_index(drop=True)
    agg.insert(0, "Place", np.arange(1, len(agg)+1))
    return agg[["Place","Pseudo","Parties","ITM","Victoires","Points"]]

# ============================================================
# PAGES
# ============================================================
PAGES = ["🏆 Tableau", "👤 Détails joueur", "📚 Archives", "🏅 Classement par points"]
page = st.sidebar.radio("Navigation", PAGES, key="nav_radio")

log = load_results_log()

# -------------------- 🏆 TABLEAU --------------------
if page == "🏆 Tableau":
    st.header("Classement général (gains / frais)")
    if log.empty:
        st.info("Aucune donnée disponible pour l’instant.")
    else:
        d1, d2, lab = pick_season(log)
        sub = log.copy()
        dt = pick_series(sub, ["start_time","processed_at"], None)
        if not pd.api.types.is_datetime64_any_dtype(dt):
            dt = pd.to_datetime(dt, errors="coerce")
        sub = sub[(dt.dt.date >= d1) & (dt.dt.date <= d2)].copy()

        table = standings_from_log(sub)
        show_table(table, caption=f"{lab} — {d1:%d/%m/%Y} au {d2:%d/%m/%Y}")

        # Export CSV
        st.download_button(
            "⬇️ Exporter le tableau (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"classement_{d1:%Y%m%d}_{d2:%Y%m%d}.csv",
            type="secondary"
        )

        # Export JPG
        exp_cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
                    "Bulles","Buy in","Frais","Gains","Bénéfices"]
        snap = table[[c for c in exp_cols if c in table.columns]].copy()
        if st.button("🖼️ Exporter en JPG"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = ROOT / f"classement_{ts}.jpg"
            try:
                classement_df_to_jpg(snap, out, title=f"Classement — {lab}")
                st.success("JPG généré.")
                st.download_button("⬇️ Télécharger le JPG", data=out.read_bytes(), file_name=out.name, type="secondary")
            except Exception as e:
                st.error(f"Export impossible : {e}")

# -------------------- 👤 DÉTAILS JOUEUR --------------------
elif page == "👤 Détails joueur":
    st.header("Fiche joueur")
    if log.empty:
        st.info("Aucun historique pour l’instant.")
    else:
        who = st.selectbox("Choisir un joueur", sorted(log["Pseudo"].dropna().astype(str).unique().tolist()))
        if who:
            sub = log[log["Pseudo"].astype(str) == who].copy()
            if sub.empty:
                st.info("Pas de données pour ce joueur.")
            else:
                # Calculs par tournoi
                gain = pick_series(sub, ["GainCash","GainsCash","Gain","Gains"], 0.0).apply(parse_money)
                bounty = pick_series(sub, ["Bounty"], 0.0).apply(parse_money)
                reentry = pd.to_numeric(pick_series(sub, ["Reentry","Recaves"], 0), errors="coerce").fillna(0).astype(int)
                buyin = pick_series(sub, ["buyin_total","Buyin","Buy-in"], 10.0).apply(parse_money)

                sub = sub.copy()
                sub["Frais"] = buyin + reentry * buyin
                sub["GainsTot"] = gain + bounty
                sub["Bénéfices"] = sub["GainsTot"] - sub["Frais"]
                st.write(f"**Total tournois:** {len(sub)}  •  **Gains totaux:** {euro(sub['GainsTot'].sum())}  •  **Frais:** {euro(sub['Frais'].sum())}  •  **Balance:** {euro(sub['Bénéfices'].sum())}")

                # Tableau lisible
                show = sub[[c for c in ("start_time","tournament_name","Position","Reentry") if c in sub.columns]].copy()
                show.rename(columns={"start_time":"Date","Reentry":"Recaves"}, inplace=True)
                show["Buy-in"] = buyin.apply(euro)
                show["Frais"] = sub["Frais"].apply(euro)
                show["Gains"] = gain.apply(euro)
                show["Bounty"] = bounty.apply(euro)
                show["Bénéfices"] = sub["Bénéfices"].apply(euro)
                st.dataframe(show, width="stretch", hide_index=True)

                # Courbes (valeurs & cumuls)
                if "start_time" in sub.columns:
                    series = pd.DataFrame({
                        "Date": pd.to_datetime(sub["start_time"], errors="coerce"),
                        "Frais": sub["Frais"].astype(float),
                        "Gains": sub["GainsTot"].astype(float),
                        "Bénéfices": sub["Bénéfices"].astype(float),
                    }).sort_values("Date")
                    tabs = st.tabs(["Évolution", "Cumuls"])
                    tabs[0].altair_chart(
                        alt.Chart(series.melt("Date", var_name="Type", value_name="Valeur"))
                           .mark_line(point=True)
                           .encode(x="Date:T", y="Valeur:Q", color="Type:N")
                           .properties(height=320),
                        use_container_width=True
                    )
                    cums = series.copy()
                    cums[["Frais","Gains","Bénéfices"]] = cums[["Frais","Gains","Bénéfices"]].cumsum()
                    tabs[1].altair_chart(
                        alt.Chart(cums.melt("Date", var_name="Type", value_name="Valeur"))
                           .mark_line(point=True)
                           .encode(x="Date:T", y="Valeur:Q", color="Type:N")
                           .properties(height=320),
                        use_container_width=True
                    )

# -------------------- 📚 ARCHIVES --------------------
elif page == "📚 Archives":
    st.header("Archives")
    if log.empty:
        st.info("Pas d’historique pour le moment.")
    else:
        st.subheader("Classement « à la date »")
        d = st.date_input("Afficher l’état au", value=date.today(), key="archive_date")
        dt = pick_series(log, ["processed_at","start_time"], None)
        if not pd.api.types.is_datetime64_any_dtype(dt):
            dt = pd.to_datetime(dt, errors="coerce")
        sub = log[(dt.dt.date <= d)].copy()
        table = standings_from_log(sub)
        show_table(table, caption=f"État arrêté au {d:%d/%m/%Y}")

    st.subheader("PDFs archivés (par saison)")
    if not PDF_DIR.exists():
        st.caption("Aucun PDF archivé.")
    else:
        # on regroupe par saison via la date de modification du fichier
        files = list_files_sorted(PDF_DIR, ("*.pdf",))
        if not files:
            st.caption("Aucun PDF archivé.")
        else:
            # saison -> [files]
            buckets: dict[str, list[Path]] = {}
            for p in files:
                dt = datetime.fromtimestamp(p.stat().st_mtime).date()
                s0, s1 = season_bounds_for(dt)
                label = f"Saison {s0.year}-{s1.year}"
                buckets.setdefault(label, []).append(p)

            for lab in sorted(buckets.keys(), reverse=True):
                with st.expander(lab, expanded=False):
                    for p in buckets[lab]:
                        cols = st.columns([6, 2])
                        cols[0].write(f"**{p.name}**  \n_{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}_")
                        with cols[1]:
                            st.download_button("Télécharger (PDF)", data=p.read_bytes(), file_name=p.name)

# -------------------- 🏅 CLASSEMENT PAR POINTS --------------------
elif page == "🏅 Classement par points":
    st.header("Classement général – Points")
    if log.empty:
        st.info("Aucune donnée pour l’instant.")
    else:
        d1, d2, lab = pick_season(log)
        pts_table = compute_points_table(log, d1, d2)
        st.dataframe(pts_table, width="stretch", hide_index=True)

        # Export CSV
        st.download_button(
            "⬇️ Exporter (CSV)",
            data=pts_table.to_csv(index=False).encode("utf-8"),
            file_name=f"classement_points_{d1:%Y%m%d}_{d2:%Y%m%d}.csv",
            type="secondary"
        )

        # Export JPG
        if st.button("🖼️ Exporter en JPG"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = ROOT / f"classement_points_{ts}.jpg"
            try:
                classement_df_to_jpg(pts_table, out, title=f"Classement par points — {lab}")
                st.success("JPG généré.")
                st.download_button("⬇️ Télécharger le JPG", data=out.read_bytes(), file_name=out.name, type="secondary")
            except Exception as e:
                st.error(f"Export impossible : {e}")
