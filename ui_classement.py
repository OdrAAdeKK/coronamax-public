# ui_classement.py
from __future__ import annotations

import io
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_classement_unique import (
    BASE, PDF_DIR, PDF_DONE, ARCHIVE, SNAP_DIR,
    F_MASTER, RESULTS_LOG, JOURNAL_CSV,
    load_master_df, save_master_df,
    load_results_log, append_results_log,
    load_journal, save_journal,
    extract_from_pdf, build_rows_for_log, rebuild_master_from_log,
    standings_from_log, archive_pdf, parse_money, euro, current_season_bounds,
    classement_df_to_jpg,  # 👈 AJOUT
)


st.set_page_config(page_title="CoronaMax – Classement", page_icon="🏆", layout="wide")

import os

CMX_MODE = os.getenv("CMX_MODE", "admin").lower()  # "admin" ou "public"

READ_ONLY_PAGES = ["🏆 Tableau", "👤 Détails joueur", "📚 Archives", "🏅 Classement par points"]
ALL_PAGES = READ_ONLY_PAGES + ["⬆️ Importer", "♻️ Réinitialiser"]

st.sidebar.title("CoronaMax")
page = st.sidebar.radio("Navigation", READ_ONLY_PAGES if CMX_MODE=="public" else ALL_PAGES)

def ensure_admin():
    if CMX_MODE == "public":
        st.warning("Cette action n’est pas disponible en lecture seule.")
        st.stop()

# ==========================================
# Helpers UI
# ==========================================
def _inject_league_gothic():
    if not st.session_state.get("_lg_font_injected"):
        st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=League+Gothic&display=swap" rel="stylesheet">
        """, unsafe_allow_html=True)
        st.session_state["_lg_font_injected"] = True

def pdf_to_jpg(pdf: Path, dpi: int = 200) -> Path:
    """Convertit la 1ʳᵉ page d’un PDF en JPG dans SNAP_DIR, retourne le chemin du JPG."""
    from pdf2image import convert_from_path
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    jpg_path = SNAP_DIR / (pdf.stem + ".jpg")
    # (Re)génère si le JPG manque ou si le PDF est plus récent
    if (not jpg_path.exists()) or (pdf.stat().st_mtime > jpg_path.stat().st_mtime):
        images = convert_from_path(str(pdf), dpi=dpi)
        images[0].save(jpg_path, "JPEG")
    return jpg_path

# ==== Helpers stats joueur (coller après les imports existants) ====

def _col(df: pd.DataFrame, names: list[str], default=0):
    """Retourne la première colonne existante parmi 'names' sinon une Series remplie de 'default'."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df), index=df.index)

def _num_s(s: pd.Series) -> pd.Series:
    """to_numeric robuste -> float, NaN->0."""
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)

def _int_s(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def _bubble_by_tournament(df_rows: pd.DataFrame) -> dict:
    """
    Renvoie {tournament_id -> pseudo_bulle} en ignorant les bounties (KO) pour l'ITM.
    Suppose colonnes: tournament_id, Position, GainCash, Pseudo.
    """
    out = {}
    if df_rows.empty:
        return out
    req = {"tournament_id", "Position", "GainCash", "Pseudo"}
    if not req.issubset(df_rows.columns):
        return out
    for tid, grp in df_rows.groupby("tournament_id"):
        g = grp.copy()
        g["Position"] = _int_s(g["Position"])
        g["GainCash"] = _num_s(g["GainCash"])
        g = g.sort_values("Position", kind="mergesort")
        no_paid = g[g["GainCash"] <= 0.0]
        out[tid] = None if no_paid.empty else str(no_paid.iloc[0]["Pseudo"])
    return out

def _max_streak(values: list[int], target: int) -> int:
    m = cur = 0
    for v in values:
        cur = cur + 1 if v == target else 0
        m = max(m, cur)
    return m

def _max_drawdown(series: pd.Series) -> float:
    """Max drawdown sur une courbe cumulée."""
    if series.empty:
        return 0.0
    cummax = series.cummax()
    dd = (cummax - series).max()
    return float(dd if pd.notna(dd) else 0.0)

# --- Streamlit rerun helper (compat new/old versions)
def _rerun():
    if hasattr(st, "rerun"):      # Streamlit >= 1.27
        st.rerun()
    else:                         # legacy
        st.experimental_rerun()

# --- Streamlit rerun helper (compat old/new)
def _rerun():
    if hasattr(st, "rerun"):      # Streamlit récents
        st.rerun()
    else:                         # versions plus anciennes
        st.experimental_rerun()
        
        
@st.cache_data(show_spinner=False)
def _cached_standings(sub_log: pd.DataFrame) -> pd.DataFrame:
    # cache les calculs du tableau 'gains'
    return standings_from_log(sub_log, season_only=False)

@st.cache_data(show_spinner=False)
def _cached_points(sub_log: pd.DataFrame) -> pd.DataFrame:
    # cache le calcul du tableau 'points'
    return compute_points_table(sub_log, None, None)  # la fonction que tu as déjà

def pick_season(log: pd.DataFrame) -> tuple[date, date, str]:
    """Widget + bornes pour la saison choisie."""
    if log.empty:
        from app_classement_unique import current_season_bounds
        d1, d2 = current_season_bounds()
        return d1, d2, "Saison courante"
    opts = seasons_available(log)
    labels = [lab for (lab, _, _) in opts]
    default_idx = len(labels) - 1
    lab = st.selectbox("Saison", labels, index=default_idx, key="season_picker")
    d1, d2 = next((a[1], a[2]) for a in opts if a[0] == lab)
    return d1, d2, lab

# --- Seasons helpers (Aug 1 -> Jul 31) ---------------------------------------

def _season_bounds_for_day(d: date) -> tuple[date, date, str]:
    """Return (start, end, label) for the season containing date d."""
    start_year = d.year if d.month >= 8 else d.year - 1
    start = date(start_year, 8, 1)
    end   = date(start_year + 1, 7, 31)
    label = f"{start_year}–{start_year+1}"
    return start, end, label

def seasons_available(log: pd.DataFrame) -> list[tuple[date, date, str]]:
    """
    Build the list of available seasons (latest first) from log dates.
    Falls back to current season when log is empty.
    """
    if log is None or log.empty:
        s0, s1, lab = _season_bounds_for_day(date.today())
        return [(s0, s1, lab)]

    # pick the best date column present
    if "start_time" in log.columns:
        ds = pd.to_datetime(log["start_time"], errors="coerce").dt.date.dropna()
    elif "processed_at" in log.columns:
        ds = pd.to_datetime(log["processed_at"], errors="coerce").dt.date.dropna()
    else:
        ds = pd.Series([], dtype="object")

    if ds.empty:
        s0, s1, lab = _season_bounds_for_day(date.today())
        return [(s0, s1, lab)]

    seen = {}
    for day in ds:
        s0, s1, lab = _season_bounds_for_day(day)
        seen[lab] = (s0, s1, lab)

    # latest season first
    return [seen[k] for k in sorted(seen.keys(), reverse=True)]

def pick_season(log: pd.DataFrame, label: str = "Saison", key: str = "season_pick") -> tuple[date, date, str]:
    """
    UI selectbox over available seasons. Returns (start_date, end_date, label).
    """
    opts = seasons_available(log)
    labels = [lab for _, _, lab in opts]
    idx = 0  # default: latest
    choice = st.selectbox(label, labels, index=idx, key=key)
    # map label back to bounds
    m = {lab: (s0, s1, lab) for s0, s1, lab in opts}
    return m[choice]


# == JPG export of the classement table =======================
def classement_df_to_jpg(df_in: pd.DataFrame, out_path: Path, *, dpi: int = 220) -> Path:
    """
    Render a DataFrame (like the home classement table) to a JPG with basic styling.
    - Grey bold header
    - Pseudo column orange; WINAMAX in white on red, uppercase
    - Parties: red if <20, green if >=20
    - Bénéfices: green→red gradient
    """
    import math, colorsys
    import matplotlib.pyplot as plt

    df = df_in.copy()

    # Ensure types/formatting
    int_cols   = ["Place","Parties","Victoires","ITM","Recaves","Bulles"]
    money_cols = ["Recaves en €","Buy in","Frais","Gains","Bénéfices"]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in money_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: euro(parse_money(x)))

    # Prepare cell text
    cols = [c for c in df.columns]
    cell_text = df[cols].astype(str).values.tolist()

    # Prepare cell colours
    nrow, ncol = df.shape
    white = "#ffffff"
    orange = "#f7b329"
    red    = "#d32f2f"

    cell_colours = [[white for _ in range(ncol)] for _ in range(nrow)]

    # Pseudo column background
    if "Pseudo" in cols:
        j = cols.index("Pseudo")
        for i, name in enumerate(df["Pseudo"].astype(str)):
            if name.strip().lower() == "winamax":
                cell_text[i][j] = name.upper()
                cell_colours[i][j] = red
            else:
                cell_colours[i][j] = orange

    # Parties colour (text)
    parties_idx = cols.index("Parties") if "Parties" in cols else None

    # Bénéfices gradient
    if "Bénéfices" in cols:
        j_ben = cols.index("Bénéfices")
        ben_vals = df["Bénéfices"].apply(parse_money).astype(float)
        vmin, vmax = float(ben_vals.min()), float(ben_vals.max())
        rng = (vmax - vmin) if vmax != vmin else 1.0

        def ben_bg(v):
            t = (parse_money(v) - vmin) / rng
            # 0→red, 1→green (use HLS so we can get soft backgrounds)
            h = 120 * t / 360.0
            r, g, b = colorsys.hls_to_rgb(h, 0.85, 0.6)  # lightness 85%
            return (r, g, b)

        for i in range(nrow):
            cell_colours[i][j_ben] = ben_bg(df.iloc[i, j_ben])

    # Matplotlib table
    # Width/height scale with number of rows/cols for decent spacing
    w = max(9, ncol * 1.2)
    h = max(2.5, nrow * 0.55)
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.axis("off")

    header_color = "#e6e6e6"
    tbl = ax.table(cellText=cell_text,
                   cellColours=cell_colours,
                   colLabels=cols,
                   colColours=[header_color]*ncol,
                   loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    # Header bold + black text
    for k, cell in tbl.get_celld().items():
        r, c = k
        if r == 0:  # header row in mpl-table
            cell.set_text_props(fontweight="bold", color="#000000")
        # Pseudo/Winamax text color white on red
        if r > 0 and "Pseudo" in cols and c == cols.index("Pseudo"):
            name = df.iloc[r-1][ "Pseudo" ]
            if str(name).strip().lower() == "winamax":
                cell.set_text_props(color="#ffffff", fontweight="bold")

        # Parties colouring
        if parties_idx is not None and r > 0 and c == parties_idx:
            val = df.iloc[r-1, parties_idx]
            if int(val) >= 20:
                cell.set_text_props(color="#107a10")
            else:
                cell.set_text_props(color="#d00000")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out_path

# --- PDF -> JPG (1re page) en mémoire -----------------------
from io import BytesIO
try:
    from app_classement_unique import POPLER_PATH  # si dispo dans ton pipeline
except Exception:
    POPLER_PATH = None

def pdf_first_page_jpg_bytes(pdf_path: Path, dpi: int = 200) -> bytes | None:
    """Retourne les octets JPG (1re page) du PDF, ou None si échec."""
    try:
        from pdf2image import convert_from_path
        imgs = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            poppler_path=POPLER_PATH if POPLER_PATH else None
        )
        if not imgs:
            return None
        buf = BytesIO()
        imgs[0].save(buf, "JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        st.warning(f"Conversion JPG impossible pour {pdf_path.name} : {e}")
        return None

def list_files_sorted(folder: Path, patterns: Iterable[str] = ("*.pdf",)) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(folder.glob(pat))
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

def style_dataframe(d: pd.DataFrame) -> pd.io.formats.style.Styler:
    _inject_league_gothic()

    df = d.copy()

    # force ints
    for c in ("Parties","Victoires","ITM","Recaves","Bulles","Place"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # € formatting
    for c in ("Recaves en €","Buy in","Frais","Gains","Bénéfices"):
        if c in df.columns:
            df[c] = df[c].apply(lambda x: euro(parse_money(x)))

    # --- make WINAMAX uppercase just for display ---
    try:
        idx_w = df.index[df["Pseudo"].astype(str).str.strip().str.lower() == "winamax"][0]
        df.at[idx_w, "Pseudo"] = str(df.at[idx_w, "Pseudo"]).upper()
    except Exception:
        pass

    sty = df.style.set_properties(**{
        "text-align": "center",
        "font-size": "0.95rem",
    })

    # headers: light grey + bold
    sty = sty.set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#e6e6e6"), ("font-weight", "bold"), ("color", "#000")]
    }])

    # column 'Pseudo': special style for WINAMAX, orange for others
    def _pseudo_styles(col: pd.Series):
        out = []
        for v in col.astype(str):
            if v.strip().upper() == "WINAMAX":
                out.append(
                    'background-color:#d32f2f; color:#fff; font-weight:900 !important; '
                    'text-transform:uppercase !important; '
                    'font-family:"League Gothic", Impact, sans-serif !important; '
                    'letter-spacing:0.5px;'
                )
            else:
                out.append('background-color:#f7b329; color:#000; font-weight:bold;')
        return out
    if "Pseudo" in df.columns:
        sty = sty.apply(_pseudo_styles, subset=["Pseudo"])

    # Parties color rule
    if "Parties" in df.columns:
        sty = sty.apply(lambda s: ["color:#d00000" if v < 20 else "" for v in df["Parties"]], subset=["Parties"])
        sty = sty.apply(lambda s: ["color:#107a10" if v >= 20 else "" for v in df["Parties"]], subset=["Parties"])

    # Gradient on Bénéfices
    if "Bénéfices" in df.columns:
        vals = df["Bénéfices"].apply(parse_money)
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = (vmax - vmin) or 1.0
        def grad(v):
            t = (parse_money(v) - vmin) / rng
            hue = int(120 * t)  # 0 red -> 120 green
            return f"background-color:hsl({hue},65%,82%); font-weight:700;"
        sty = sty.apply(lambda col: [grad(v) for v in df["Bénéfices"]], subset=["Bénéfices"])

    # dashed separator below Winamax row
    try:
        idx = df.index[df["Pseudo"].astype(str).str.upper() == "WINAMAX"][0]
        sty = sty.set_table_styles([{
            "selector": f"tbody tr:nth-child({idx+2})",
            "props": [("border-bottom","3px dashed #222")]
        }], overwrite=False)
    except Exception:
        pass

    return sty

def _pick_series(df: pd.DataFrame, names: list[str], default=0.0) -> pd.Series:
    """Retourne la 1ère série existante parmi `names`, sinon une série constante."""
    for n in names:
        if n in df.columns:
            s = df[n]
            # si jamais c'est un scalaire (rare), renvoyer une série de même longueur
            if not hasattr(s, "shape"):
                return pd.Series([default] * len(df), index=df.index)
            return s
    return pd.Series([default] * len(df), index=df.index)

def compute_points_table(log: pd.DataFrame, d1: date, d2: date) -> pd.DataFrame:
    """
    Classement par points sur [d1, d2].
    - Points = N_participants - position + 1 (min=1)
    - ITM    = nb de tournois où GainCash > 0 (Bounty ignoré)
    - Victoires = nb de positions == 1
    - Parties   = nb de participations
    - Winamax exclu
    """
    cols = ["Place","Pseudo","Parties","ITM","Victoires","Points"]
    if log is None or log.empty:
        return pd.DataFrame(columns=cols)

    df = log.copy()

    # Filtre période (si start_time présent)
    if "start_time" in df.columns:
        df = df[(df["start_time"].dt.date >= d1) & (df["start_time"].dt.date <= d2)].copy()
        if df.empty:
            return pd.DataFrame(columns=cols)

    # Séries robustes
    pseudo  = _pick_series(df, ["Pseudo"], "").astype(str)
    tour_id = _pick_series(df, ["tournament_id"], "").astype(str)
    pos     = pd.to_numeric(_pick_series(df, ["Position","Place","Rank"], 0), errors="coerce").fillna(0).astype(int)

    # Gains "cash" pour l'ITM (ignore bounty)
    gaincash_raw = _pick_series(df, ["GainsCash","GainCash","Gains","Gain"], 0.0)
    gaincash = gaincash_raw.apply(parse_money)  # gère "35,10 €", etc.

    # N participants par tournoi (robuste)
    # -> même index, aucun besoin de columns spécifiques
    n_part = tour_id.map(tour_id.value_counts()).astype(int)

    # Points (min 1)
    points = (n_part - pos + 1).clip(lower=1).astype(int)

    # Indicateurs
    itm_flag = (gaincash > 0).astype(int)
    win_flag = (pos == 1).astype(int)

    tmp = pd.DataFrame({
        "Pseudo": pseudo,
        "Points": points,
        "ITM": itm_flag,
        "Victoires": win_flag,
        "Parties": 1,
    })

    # Exclure Winamax
    tmp = tmp[~tmp["Pseudo"].str.lower().eq("winamax")]

    if tmp.empty:
        return pd.DataFrame(columns=cols)

    agg = tmp.groupby("Pseudo", as_index=False).agg(
        Parties=("Parties","sum"),
        ITM=("ITM","sum"),
        Victoires=("Victoires","sum"),
        Points=("Points","sum"),
    )

    # Tri + Place (Points desc, puis Victoires, ITM, Parties)
    agg = agg.sort_values(
        by=["Points","Victoires","ITM","Parties"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)
    agg.insert(0, "Place", np.arange(1, len(agg)+1))

    return agg[cols]




def show_table(df: pd.DataFrame, caption: str | None = None):
    st.dataframe(style_dataframe(df), use_container_width=True, hide_index=True)
    if caption:
        st.caption(caption)


# --- Pending queue for tournaments to validate ---
def _init_pending_queue():
    if "pending_tourneys" not in st.session_state:
        st.session_state["pending_tourneys"] = {}  # tid -> dict

def _enqueue_parsed_tournament(parsed, src_path: Path):
    """
    Store one parsed tournament in the session queue.
    Also refuses empty parses (0 rows) with a clear message.
    """
    from app_classement_unique import build_rows_for_log  # local import to avoid cycles
    _init_pending_queue()
    rows = build_rows_for_log(parsed)  # -> DataFrame with Position/Pseudo/GainCash/Bounty/Reentry, etc.

    # 🔎 DIAGNOSTIC si 0 ligne
    if rows is None or len(rows) == 0:
        st.error(
            "Aucune ligne de résultats détectée dans ce PDF.\n"
            "Vérifie le format Winamax (table ‘Résultats’) ou réessaie avec un autre export."
        )
        # Montre les métadonnées détectées (si dispo) pour aider au debug
        try:
            st.caption(
                f"Debug parsing — tournoi: **{parsed.tournament_name}**, "
                f"date: **{pd.to_datetime(parsed.start_time):%d/%m/%Y %H:%M}**, "
                f"buy-in total: **{getattr(parsed, 'buyin_total', 'n/a')}**, "
                f"KO: **{getattr(parsed, 'is_ko', 'n/a')}**"
            )
        except Exception:
            pass
        return  # ne pas mettre en file

    # ✅ OK: mettre en file
    st.session_state.pending_tourneys[parsed.tournament_id] = {
        "df": rows,
        "src_path": str(src_path),
        "name": parsed.tournament_name,
        "start_time": parsed.start_time,
    }

def compute_bubble_from_rows(df_rows: pd.DataFrame) -> str | None:
    """
    Retourne le pseudo détecté 'à la bulle' à partir d'un DataFrame
    (colonnes attendues : Position, GainCash, Pseudo). Ignore Bounty.
    Robuste si colonnes manquantes.
    """
    import pandas as pd
    if df_rows is None or len(df_rows) == 0:
        return None

    d = df_rows.copy()
    n = len(d)

    # Helpers: toujours renvoyer une Series de longueur n
    def as_series(name: str, default):
        if name in d.columns:
            s = d[name]
            # Si c'est un scalaire (rare), duplique-le
            if not hasattr(s, "shape"):
                return pd.Series([default] * n, index=d.index)
            return s
        else:
            return pd.Series([default] * n, index=d.index)

    pos_s   = pd.to_numeric(as_series("Position", 0), errors="coerce").fillna(0).astype(int)
    gain_s  = pd.to_numeric(as_series("GainCash", 0.0), errors="coerce").fillna(0.0)
    pseudo_s= as_series("Pseudo", "").astype(str)

    tmp = pd.DataFrame({
        "Position": pos_s,
        "GainCash": gain_s,
        "Pseudo": pseudo_s
    }, index=d.index)

    # Tri par position croissante
    tmp = tmp.sort_values("Position", kind="mergesort")

    # Bulle = premier joueur sans GainCash strictement positif
    no_paid = tmp.loc[tmp["GainCash"] <= 0.0]
    if no_paid.empty:
        return None
    return str(no_paid["Pseudo"].iloc[0])


# -------------------------------
# Barre latérale & navigation
# -------------------------------
st.sidebar.title("CoronaMax")

# Définis les pages disponibles selon le mode
PAGES_PUBLIC = ["🏆 Tableau", "👤 Détails joueur", "📚 Archives", "🏅 Classement par points"]
PAGES_FULL   = PAGES_PUBLIC + ["⬆️ Importer", "♻️ Réinitialiser"]

CMX_MODE = os.environ.get("CMX_MODE", "local").lower()  # "public" ou "local"
PAGES = PAGES_PUBLIC if CMX_MODE == "public" else PAGES_FULL

# IMPORTANT : key unique pour éviter les collisions
page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=0,
    key="nav_main"   # <--- clé unique
)


# ==========================================
# 1) 🏆 TABLEAU
# ==========================================
if page == "🏆 Tableau":
    st.title("Classement général")

    if RESULTS_LOG.exists() and RESULTS_LOG.stat().st_size > 0:
        log = load_results_log()
        d1, d2, _ = pick_season(log)  # utilise ta fonction saison
        with st.expander("Filtrer par période (saison active par défaut)", expanded=False):
            d1 = st.date_input("Du", value=d1)
            d2 = st.date_input("Au", value=d2)
        sub = log[(log["start_time"].dt.date >= d1) & (log["start_time"].dt.date <= d2)].copy()
        table = standings_from_log(sub, season_only=False)
    else:
        table = load_master_df()

    if table.empty:
        st.info("Aucune donnée pour l’instant. Va dans **Importer** pour traiter des PDFs.")
    else:
        show_table(table)

        # Export CSV de l’affichage
        st.download_button(
            "⬇️ Exporter (CSV affiché)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="classement.csv",
            type="secondary"
        )

        # --- Export JPG du classement (PLACÉ DANS LA BRANCHE TABLEAU)
        export_cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM",
                       "Recaves","Recaves en €","Bulles","Buy in","Frais",
                       "Gains","Bénéfices"]
        export_df = table[[c for c in export_cols if c in table.columns]].copy()

        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("🖼️ Exporter le tableau en JPG"):
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                jpg_file = SNAP_DIR / f"classement_{ts}.jpg"
                try:
                    classement_df_to_jpg(export_df, jpg_file)
                    st.session_state["last_jpg_export"] = str(jpg_file)
                    st.success("JPG généré.")
                except Exception as e:
                    st.error(f"Export impossible : {e}")

        with c2:
            jp = st.session_state.get("last_jpg_export")
            if jp and Path(jp).exists():
                st.download_button(
                    "⬇️ Télécharger le JPG",
                    data=Path(jp).read_bytes(),
                    file_name=Path(jp).name,
                    type="secondary"
                )


# ==========================================
# 🏅 CLASSEMENT PAR POINTS
# ==========================================
elif page == "🏅 Classement par points":
    st.title("Classement général – Points")

    try:
        import numpy as np
        import pandas as pd
        from pathlib import Path

        # ---------- Safe fallbacks (used only if not provided elsewhere) ----------
        if "parse_money" not in globals():
            def parse_money(x):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return 0.0
                if isinstance(x, (int, float)):
                    return float(x)
                s = str(x).strip().replace("\u00a0", " ").replace("€", "").replace(" ", "")
                s = s.replace(",", ".")
                try:
                    return float(s)
                except Exception:
                    return 0.0

        def _pick_series(df: pd.DataFrame, names, default):
            for n in names:
                if n in df.columns:
                    return df[n]
            return pd.Series([default] * len(df), index=df.index)

        if "compute_points_table" not in globals():
            def compute_points_table(log: pd.DataFrame, d1, d2) -> pd.DataFrame:
                """
                Classement par points sur [d1, d2].
                - Points = N_participants - position + 1 (min 1)
                - ITM = nb de tournois où GainCash > 0 (bounty ignoré)
                - Victoires = nb de premières places
                - Parties = nb de participations
                Exclut Winamax.
                """
                if log.empty:
                    return pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

                df = log.copy()
                # Fenêtre de dates
                if "start_time" in df.columns:
                    df = df[(df["start_time"].dt.date >= d1) & (df["start_time"].dt.date <= d2)].copy()
                if df.empty:
                    return pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

                # Colonnes robustes
                pos     = pd.to_numeric(_pick_series(df, ["Position","Place","Rank"], 0), errors="coerce").fillna(0).astype(int)
                pseudo  = _pick_series(df, ["Pseudo"], "").astype(str)
                tour_id = _pick_series(df, ["tournament_id"], "").astype(str)

                # Gains cash pour ITM (sans bounty)
                gaincash_raw = _pick_series(df, ["GainsCash","GainCash","Gains","Gain"], 0.0)
                gaincash     = gaincash_raw.apply(parse_money)

                # N participants par tournoi (si pas de tournament_id, fallback groupby date)
                if "tournament_id" in df.columns:
                    n_part = df.groupby("tournament_id")["Pseudo"].transform("size")
                else:
                    grp = ["start_time"]
                    if "tournament_name" in df.columns: grp.append("tournament_name")
                    n_part = df.groupby(grp)["Pseudo"].transform("size") if grp[0] in df.columns else pd.Series([len(df)]*len(df), index=df.index)

                points   = (n_part - pos + 1).clip(lower=1)
                itm_flag = (gaincash > 0).astype(int)
                win_flag = (pos == 1).astype(int)

                tmp = pd.DataFrame({
                    "Pseudo": pseudo,
                    "Points": points,
                    "ITM": itm_flag,
                    "Victoires": win_flag,
                    "Parties": 1,
                })

                # Exclure Winamax
                tmp = tmp[~tmp["Pseudo"].str.lower().eq("winamax")]

                agg = tmp.groupby("Pseudo", as_index=False).agg(
                    Parties=("Parties","sum"),
                    ITM=("ITM","sum"),
                    Victoires=("Victoires","sum"),
                    Points=("Points","sum"),
                )
                # Tri par Points desc, puis Victoires, ITM, Parties
                agg = agg.sort_values(by=["Points","Victoires","ITM","Parties"],
                                      ascending=[False, False, False, False]).reset_index(drop=True)
                agg.insert(0, "Place", np.arange(1, len(agg)+1))
                return agg[["Place","Pseudo","Parties","ITM","Victoires","Points"]]

        if "current_season_bounds" not in globals():
            # Fallback saison: 01/08/N → 31/07/N+1 (N = année selon aujourd’hui)
            from datetime import date
            def current_season_bounds():
                today = date.today()
                if today.month >= 8:
                    s0 = date(today.year, 8, 1)
                    s1 = date(today.year+1, 7, 31)
                else:
                    s0 = date(today.year-1, 8, 1)
                    s1 = date(today.year, 7, 31)
                return s0, s1

        if "seasons_available" not in globals():
            def seasons_available(log: pd.DataFrame):
                if log.empty or "start_time" not in log.columns:
                    s0, s1 = current_season_bounds()
                    return [("courante", s0, s1)]
                years = sorted(log["start_time"].dt.year.unique().tolist())
                seasons = []
                for y in years + [max(years)+1]:
                    # saison démarre 01/08/y → 31/07/y+1
                    from datetime import date
                    s0 = date(y, 8, 1)
                    s1 = date(y+1, 7, 31)
                    lab = f"{y}-{(y+1)%100:02d}"
                    seasons.append((lab, s0, s1))
                return seasons

        if "pick_season" not in globals():
            def pick_season(log: pd.DataFrame, key="season_points"):
                opts = seasons_available(log)
                labels = [lab for (lab, _, _) in opts]
                default_idx = max(0, len(labels)-1)
                choice = st.selectbox("Saison", labels, index=default_idx, key=key)
                for lab, s0, s1 in opts:
                    if lab == choice:
                        return s0, s1, lab
                # fallback
                s0, s1 = current_season_bounds()
                return s0, s1, "courante"

        # ---------- Data + UI ----------
        log = load_results_log()

        if log.empty:
            st.info("Aucun historique pour l’instant. Va dans **Importer** pour traiter des PDFs.")
        else:
            d1_season, d2_season, season_label = pick_season(log, key="season_points")

            with st.expander("Filtrer (saison active par défaut)", expanded=False):
                mode = st.radio("Mode d’affichage", ["Période", "À la date"], horizontal=True, key="pts_mode")
                if mode == "Période":
                    c1, c2 = st.columns(2)
                    with c1:
                        d1 = st.date_input("Du", value=d1_season, key="pts_d1")
                    with c2:
                        d2 = st.date_input("Au", value=d2_season, key="pts_d2")
                else:
                    d_at = st.date_input("Afficher l’état au", value=d2_season, key="pts_at")
                    d1, d2 = d1_season, d_at

            pts_table = compute_points_table(log, d1, d2)

            if pts_table.empty:
                st.info(f"Aucune donnée entre {d1:%d/%m/%Y} et {d2:%d/%m/%Y}.")
            else:
                # show_table si dispo, sinon dataframe simple
                if "show_table" in globals():
                    show_table(pts_table, caption=f"Période : {d1:%d/%m/%Y} → {d2:%d/%m/%Y} (Saison {season_label})")
                else:
                    st.dataframe(pts_table, use_container_width=True, hide_index=True)

                # Export CSV
                st.download_button(
                    "⬇️ Exporter (CSV affiché)",
                    data=pts_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"classement_points_{d1:%Y%m%d}_{d2:%Y%m%d}.csv",
                    type="secondary"
                )

                # Export JPG (optionnel, si helper dispo)
                export_cols_pts = ["Place","Pseudo","Parties","ITM","Victoires","Points"]
                snap_pts = pts_table[[c for c in export_cols_pts if c in pts_table.columns]].copy()

                c1, c2 = st.columns([1, 3])
                with c1:
                    if "classement_df_to_jpg" in globals() and "SNAP_DIR" in globals():
                        if st.button("🖼️ Exporter ce classement (JPG)", key="btn_pts_jpg"):
                            from datetime import datetime
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            jpg_file = SNAP_DIR / f"classement_points_{ts}.jpg"
                            try:
                                classement_df_to_jpg(
                                    snap_pts, jpg_file,
                                    title=f"Classement par points – {d1:%d/%m/%Y} → {d2:%d/%m/%Y}"
                                )
                                st.session_state["last_pts_jpg_export"] = str(jpg_file)
                                st.success("JPG généré.")
                            except Exception as e:
                                st.error(f"Export impossible : {e}")
                    else:
                        st.caption("Astuce : ajoute `classement_df_to_jpg(...)` pour activer l’export JPG ici.")

                with c2:
                    jp = st.session_state.get("last_pts_jpg_export")
                    if jp and Path(jp).exists():
                        st.download_button("⬇️ Télécharger le JPG",
                                           data=Path(jp).read_bytes(),
                                           file_name=Path(jp).name,
                                           type="secondary")

    except Exception as e:
        # Montre l'erreur plutôt qu'un écran blanc
        st.error("Une erreur est survenue dans la page 'Classement par points'.")
        st.exception(e)


# ==========================================
# 2) ⬆️ IMPORTER
# ==========================================

elif page == "⬆️ Importer":
    ensure_admin()
    st.title("Importer des résultats (PDF Winamax)")

    _init_pending_queue()

    # 1) Ajouter de nouveaux PDFs dans la file de validation
    up = st.file_uploader("Déposez un ou plusieurs PDFs", type=["pdf"], accept_multiple_files=True)
    if up and st.button("Analyser et mettre en file", type="primary"):
        logs = []
        for f in up:
            tmp = PDF_DIR / f.name
            tmp.write_bytes(f.read())
            try:
                parsed = extract_from_pdf(tmp)
                # construire un aperçu éditable (et vérifier au moins 1 ligne)
                from app_classement_unique import build_rows_for_log
                rows_preview = build_rows_for_log(parsed)
                if rows_preview is None or rows_preview.empty:
                    st.warning(f"⚠️ {f.name} : 0 ligne détectée (non ajouté).")
                    tmp.unlink(missing_ok=True)
                    continue
                _enqueue_parsed_tournament(parsed, tmp)
                logs.append(f"✅ Ajouté en file : {f.name} — {len(rows_preview)} ligne(s)")
            except Exception as e:
                tmp.unlink(missing_ok=True)
                logs.append(f"❌ Erreur sur {f.name} : {e}")
        if logs:
            st.text("\n".join(logs))
        st.info("Faites défiler pour valider chaque tournoi en file.")

    # helper rerun (compat versions récentes/anciennes de Streamlit)
    def _rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    # 2) File d’attente : tableau éditable + Valider/Annuler
    if not st.session_state.pending_tourneys:
        st.caption("Aucun tournoi en attente de validation.")
    else:
        st.subheader("Tournois en attente de validation")

        for tid, item in list(st.session_state.pending_tourneys.items()):
            st.markdown(f"**{item['name']} — {pd.to_datetime(item['start_time']):%d/%m/%Y %H:%M}**")

            edit = st.data_editor(
                item["df"],
                num_rows="dynamic",
                width="stretch",      # remplace use_container_width=True
                hide_index=True,
                key=f"edit_{tid}"
            )

            # 🔎 Indication bulle sur les données éditées
            bubble_name = compute_bubble_from_rows(edit)
            st.info(f"**Bulle détectée :** {bubble_name or '(aucune)'}")

            c1, c2 = st.columns(2)

            with c1:
                if st.button("✅ Valider ce tournoi", key=f"commit_{tid}"):
                    # 1) Écrit dans le log détaillé
                    append_results_log(edit.copy())

                    # 2) Journal + archivage du PDF source
                    journal = load_journal()
                    journal.loc[len(journal)] = {
                        "sha1": tid,
                        "filename": Path(item["src_path"]).name,
                        "processed_at": datetime.now()
                    }
                    save_journal(journal)
                    archive_pdf(Path(item["src_path"]))

                    # 3) Retirer de la file + reconstruire le master
                    del st.session_state.pending_tourneys[tid]
                    rebuild_master_from_log()

                    # 4) JPG auto du classement (gains) après validation
                    try:
                        cur_log = load_results_log()
                        d1, d2, _ = pick_season(cur_log)  # utilise le sélecteur de saison si dispo
                        sub = cur_log[(cur_log["start_time"].dt.date >= d1) & (cur_log["start_time"].dt.date <= d2)].copy()
                        tbl = standings_from_log(sub, season_only=False)
                        export_cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
                                       "Bulles","Buy in","Frais","Gains","Bénéfices"]
                        snap = tbl[[c for c in export_cols if c in tbl.columns]].copy()
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        jpg_auto = SNAP_DIR / f"classement_auto_{ts}_{tid[:8]}.jpg"
                        classement_df_to_jpg(snap, jpg_auto)
                        st.success(f"Tournoi ajouté. JPG auto : {jpg_auto.name}")
                    except Exception as e:
                        st.warning(f"Classement mis à jour (sans JPG auto) : {e}")

                    _rerun()  # rafraîchit immédiatement la file

            with c2:
                if st.button("🗑️ Annuler / retirer de la file", key=f"cancel_{tid}"):
                    Path(item["src_path"]).unlink(missing_ok=True)
                    del st.session_state.pending_tourneys[tid]
                    st.warning("Tournoi retiré de la file.")
                    _rerun()

            st.divider()

        # ---- Rollback (annuler le dernier import déjà validé)
        st.subheader("Annuler le dernier import")
        if st.button("🧹 Annuler le dernier tournoi importé"):
            from app_classement_unique import rollback_last_import  # import local pour éviter de toucher les imports globaux
            info = rollback_last_import()
            if info.get("ok"):
                msg = f"{info['msg']}"
                if info.get("pdf_back"):
                    msg += f" — PDF remis en entrée : {info['pdf_back']}"
                st.success(msg)
                _rerun()
            else:
                st.info(info.get("msg", "Rien à annuler."))


# ==========================================
# 3) 👤 DÉTAILS JOUEUR
# ==========================================
elif page == "👤 Détails joueur":
    st.title("Fiche joueur")
    log = load_results_log()

    if log.empty:
        st.info("Aucun historique pour l’instant.")
    else:
        who = st.selectbox("Choisir un joueur", sorted(log["Pseudo"].dropna().unique().tolist()))
        sub = log[log["Pseudo"] == who].copy().sort_values("start_time")

        if sub.empty:
            st.info("Pas de données pour ce joueur.")
        else:
            # Colonnes robustes
            pos      = _int_s(_col(sub, ["Position"], 0))
            buyin    = _num_s(_col(sub, ["buyin_total"], 0.0))         # buy-in + rake (total unitaire du tournoi)
            recaves  = _int_s(_col(sub, ["Reentry", "Recaves"], 0))    # alias tolérés
            gaincash = _num_s(_col(sub, ["GainCash", "GainsCash"], 0.0))
            bounty   = _num_s(_col(sub, ["Bounty"], 0.0))

            # Frais / Gains / Profit par tournoi
            frais_row   = buyin * (1.0 + recaves)      # ✅ frais = buy-in TOTAL (incl. rake) + recaves * buy-in TOTAL
            gains_tot   = gaincash + bounty
            profit_row  = gains_tot - frais_row

            # Séries temps
            sub["Frais"]     = frais_row
            sub["GainsTot"]  = gains_tot
            sub["Profit"]    = profit_row
            sub["Date"]      = pd.to_datetime(sub.get("start_time", pd.NaT))

            # KPIs de base
            n        = int(len(sub))
            wins     = int((pos == 1).sum())
            itm_flag = (gaincash > 0).astype(int)
            itm_cnt  = int(itm_flag.sum())
            top3_cnt = int((pos <= 3).sum())

            # last place (%)
            # field_size par tournoi = max(Position)
            field_by_t = sub.groupby("tournament_id")["Position"].max().rename("field_size")
            sub2 = sub.join(field_by_t, on="tournament_id")
            last_cnt = int((pos == _int_s(sub2["field_size"])).sum())

            # bulle (recalculée à la volée)
            bubbles_map = _bubble_by_tournament(sub)
            bubble_cnt = 0
            for tid, p_bulle in bubbles_map.items():
                if p_bulle and who == p_bulle:
                    bubble_cnt += 1

            # Streaks + drawdown
            cash_streak_max   = _max_streak(itm_flag.tolist(), 1)
            nocash_streak_max = _max_streak(itm_flag.tolist(), 0)
            cum_profit = profit_row.cumsum()
            dd_max = _max_drawdown(cum_profit)

            # ROI et profits
            frais_tot   = float(frais_row.sum())
            gains_tot_s = float(gains_tot.sum())
            profit_tot  = float(profit_row.sum())
            roi         = (profit_tot / frais_tot) if frais_tot > 0 else 0.0
            profit_avg  = float(profit_row.mean()) if n > 0 else 0.0

            # Recaves : taux & rentabilité
            recaves_any   = (recaves > 0)
            rec_rate      = float(recaves_any.mean()) if n > 0 else 0.0
            rec_mean      = float(recaves.mean()) if n > 0 else 0.0
            prof_with_rec = float(profit_row[recaves_any].mean()) if recaves_any.any() else 0.0
            prof_no_rec   = float(profit_row[~recaves_any].mean()) if (~recaves_any).any() else 0.0

            # KPIs (badges)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Parties", n)
                st.metric("Victoires %", f"{(wins/n if n else 0):.0%}")
                st.metric("Top 3 %", f"{(top3_cnt/n if n else 0):.0%}")
            with k2:
                st.metric("ITM %", f"{(itm_cnt/n if n else 0):.0%}")
                st.metric("Dernière place %", f"{(last_cnt/n if n else 0):.0%}")
                st.metric("Bulles", bubble_cnt)
            with k3:
                st.metric("ROI", f"{roi:.0%}")
                st.metric("Bénéfice total", euro(profit_tot))
                st.metric("Bénéfice moyen", euro(profit_avg))
            with k4:
                st.metric("Taux de recave", f"{rec_rate:.0%}")
                st.metric("Recaves moy.", f"{rec_mean:.2f}")
                st.metric("Max drawdown", euro(dd_max))

            st.divider()

            # Graphes
            t1, t2 = st.tabs(["Évolution & Streaks", "Positions & Segments"])

            # Évolution
            with t1:
                left, right = st.columns([3,2], gap="large")

                with left:
                    # Bénéfice cumulé (ligne) + 0 (référence)
                    evol = sub[["Date","Profit"]].copy()
                    evol["Cumul"] = evol["Profit"].cumsum()
                    base = alt.Chart(evol).mark_line(point=True).encode(
                        x="Date:T", y=alt.Y("Cumul:Q", title="Bénéfice cumulé (€)")
                    ).properties(height=320)
                    zero = alt.Chart(evol).mark_rule().encode(y=alt.datum(0))
                    st.altair_chart(base + zero, use_container_width=True)

                with right:
                    # Rentabilité recave vs sans recave
                    seg = pd.DataFrame({
                        "Segment": ["Avec recave", "Sans recave"],
                        "Bénéfice moyen": [prof_with_rec, prof_no_rec]
                    })
                    chart = alt.Chart(seg).mark_bar().encode(
                        x="Segment:N",
                        y=alt.Y("Bénéfice moyen:Q", title="€"),
                        color="Segment:N"
                    ).properties(height=320)
                    st.altair_chart(chart, use_container_width=True)

                st.caption(f"Streak ITM max: **{cash_streak_max}** — Streak sans ITM max: **{nocash_streak_max}**")

            # Positions & Segments
            with t2:
                c1, c2 = st.columns(2, gap="large")

                with c1:
                    # Histogramme des positions
                    hist = alt.Chart(sub).mark_bar().encode(
                        x=alt.X("Position:Q", bin=alt.Bin(maxbins=20), title="Position"),
                        y=alt.Y("count():Q", title="Occurrences")
                    ).properties(height=300)
                    st.altair_chart(hist, use_container_width=True)

                with c2:
                    # ROI par buy-in (moyennes)
                    grp = sub.copy()
                    grp["Frais"] = frais_row
                    grp["Profit"]= profit_row
                    roi_buyin = grp.groupby("buyin_total").apply(
                        lambda d: (d["Profit"].sum()/d["Frais"].sum()) if d["Frais"].sum()>0 else 0.0
                    ).reset_index(name="ROI")
                    if not roi_buyin.empty:
                        roi_buyin["ROI%"] = roi_buyin["ROI"] * 100
                        ch = alt.Chart(roi_buyin).mark_bar().encode(
                            x=alt.X("buyin_total:Q", title="Buy-in total (€)"),
                            y=alt.Y("ROI%:Q", title="ROI (%)")
                        ).properties(height=300)
                        st.altair_chart(ch, use_container_width=True)
                    else:
                        st.caption("Pas assez de données pour le ROI par buy-in.")

            st.divider()

            # Table tournois (lisible)
            show = sub[["Date","tournament_name","Position"]].copy()
            show["Recaves"]  = recaves
            show["buyin_total"] = buyin.apply(lambda x: euro(x))
            show["Frais"]    = frais_row.apply(lambda x: euro(x))
            show["GainCash"] = gaincash.apply(lambda x: euro(x))
            show["Bounty"]   = bounty.apply(lambda x: euro(x))
            show["Bénéfice"] = profit_row.apply(lambda x: euro(x))
            st.dataframe(show.rename(columns={
                "tournament_name":"Tournoi",
                "buyin_total":"Buy-in"
            }), use_container_width=True, hide_index=True)


# ==========================================
# 4) 📚 ARCHIVES
# ==========================================
if CMX_MODE != "public":
    from app_classement_unique import build_public_snapshot, BASE
    st.divider()
    st.subheader("Publication (snapshot public)")
    dst = BASE / "public_snapshot"

    colA, colB = st.columns([1,2])
    with colA:
        if st.button("📤 Générer le snapshot local"):
            try:
                build_public_snapshot(dst)
                st.success(f"Snapshot prêt dans : {dst}")
            except Exception as e:
                st.error(f"Échec génération : {e}")

    with colB:
        with st.expander("Uploader par SFTP (optionnel)"):
            host = st.text_input("Hôte SFTP")
            user = st.text_input("Utilisateur")
            pwd  = st.text_input("Mot de passe", type="password")
            remote_dir = st.text_input("Dossier distant (ex: /www/public_snapshot)")

            if st.button("🚀 Uploader le snapshot"):
                try:
                    import os, paramiko
                    transport = paramiko.Transport((host, 22))
                    transport.connect(username=user, password=pwd)
                    sftp = paramiko.SFTPClient.from_transport(transport)

                    def put_dir(local, remote):
                        for root, dirs, files in os.walk(local):
                            rel = os.path.relpath(root, local)
                            rdir = remote if rel == "." else f"{remote}/{rel}".replace("\\", "/")
                            # mkdir -p
                            parts = rdir.strip("/").split("/")
                            cur = ""
                            for part in parts:
                                cur = f"{cur}/{part}" if cur else f"/{part}"
                                try:
                                    sftp.stat(cur)
                                except IOError:
                                    sftp.mkdir(cur)
                            for f in files:
                                sftp.put(os.path.join(root, f), f"{rdir}/{f}")

                    put_dir(str(dst), remote_dir)
                    sftp.close(); transport.close()
                    st.success("Upload terminé.")
                except Exception as e:
                    st.error(f"Upload SFTP impossible : {e}")


elif page == "📚 Archives":
    st.title("Archives")

    # 4.1) Classement « à la date »
    st.subheader("Classement « à la date »")
    log = load_results_log()
    if log.empty:
        st.info("Pas d’historique pour le moment.")
    else:
        # On garde le sélecteur de saison (pratique), avec possibilité de préciser une date
        d1, d2, season_label = pick_season(log, key="season_arch_classement")
        d = st.date_input("Afficher l’état au", value=d2)
        sub = log[(log["start_time"].dt.date <= d)].copy()
        table = standings_from_log(sub, season_only=False)
        show_table(table, caption=f"État arrêté au {d:%d/%m/%Y} (saison {season_label})")

    st.divider()

    # 4.2) PDFs (par saison, triés par date décroissante)
    st.subheader("PDFs")
    if log.empty:
        st.caption("Aucun PDF archivé.")
    else:
        # Choix de saison pour filtrer les PDFs
        s1, s2, lab = pick_season(log, key="season_arch_pdfs")
        st.caption(f"Saison sélectionnée : {lab} (du {s1:%d/%m/%Y} au {s2:%d/%m/%Y})")

        pdfs = list_files_sorted(PDF_DONE, ("*.pdf",))
        # filtre sur la saison via la date de modification du fichier
        pdfs = [p for p in pdfs if s1 <= datetime.fromtimestamp(p.stat().st_mtime).date() <= s2]

        if not pdfs:
            st.caption("Aucun PDF pour cette saison.")
        else:
            # affichage plus compact : tableau en liste avec 2 boutons par ligne
            for p in pdfs:
                ts = datetime.fromtimestamp(p.stat().st_mtime)
                cols = st.columns([6, 2, 2])
                cols[0].write(f"**{p.name}**  \n_{ts:%Y-%m-%d %H:%M}_")

                with cols[1]:
                    st.download_button(
                        "⬇️ PDF",
                        data=p.read_bytes(),
                        file_name=p.name
                    )
                with cols[2]:
                    jpg_bytes = pdf_first_page_jpg_bytes(p)
                    if jpg_bytes:
                        st.download_button(
                            "⬇️ JPG",
                            data=jpg_bytes,
                            file_name=p.with_suffix(".jpg").name
                        )
                    else:
                        st.button("JPG indisponible", disabled=True)

# ==========================================
# 5) ♻️ RÉINITIALISER
# ==========================================

elif page == "♻️ Réinitialiser":
    ensure_admin()
    st.title("Réinitialiser la saison courante")
    st.warning("Attention : remise à zéro des agrégats. Les PDFs archivés peuvent être re-traités ensuite.")

    # options
    do_move = st.checkbox("Remettre les PDFs archivés dans « PDF_A_TRAITER » (suppression de l’horodatage)", value=True)
    do_reprocess = st.checkbox("Relancer automatiquement le traitement après réinitialisation", value=False)

    if st.button("⚠️ Lancer la réinitialisation", type="primary"):
        # backup simples
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (ARCHIVE / "BACKUPS").mkdir(parents=True, exist_ok=True)
        if F_MASTER.exists():
            (ARCHIVE / "BACKUPS" / f"MASTER_{ts}.csv").write_bytes(F_MASTER.read_bytes())
        if RESULTS_LOG.exists():
            (ARCHIVE / "BACKUPS" / f"results_log_{ts}.csv").write_bytes(RESULTS_LOG.read_bytes())
        if JOURNAL_CSV.exists():
            (ARCHIVE / "BACKUPS" / f"journal_{ts}.csv").write_bytes(JOURNAL_CSV.read_bytes())

        # vider master
        save_master_df(pd.DataFrame(columns=load_master_df().columns))

        # vider journal
        save_journal(pd.DataFrame(columns=["sha1","filename","processed_at"]))

        # remettre PDFs
        if do_move:
            for p in list_files_sorted(PDF_DONE, ("*.pdf",)):
                name = p.name.split("__")[0] + ".pdf" if "__" in p.name else p.name
                dst = PDF_DIR / name
                dst.write_bytes(p.read_bytes())
                p.unlink(missing_ok=True)

        # (option) retraitement
        if do_reprocess:
            # mini retraitement : on ouvre Importer et l’utilisateur valide tournoi par tournoi
            st.success("Réinitialisation OK. Va sur l’onglet **Importer** pour valider à nouveau les tournois.")
        else:
            st.success("Réinitialisation OK.")
