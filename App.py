# App.py
from __future__ import annotations

import os, re, io, shutil
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Union, List


import numpy as np
import pandas as pd
import streamlit as st

from app_classement_unique import (
    BASE, ARCHIVE, PDF_DIR, PDF_DONE, SNAP_DIR, DATA_DIR,
    RESULTS_LOG, JOURNAL_CSV,
    RESULTS_LOG_COLUMNS, JOURNAL_COLUMNS,
    parse_money, euro, current_season_bounds,
    load_results_log_any, load_latest_master_any,
    load_journal, save_journal, append_results_log,
    extract_from_pdf, build_rows_for_log, build_manual_rows_for_log,
    render_manual_results_pdf,
    standings_from_log, compute_points_table,
    compute_bubble_from_rows,
    classement_df_to_jpg, classement_points_df_to_jpg,
    archive_pdf, rollback_last_import,
    publish_public_snapshot, safe_unlink,
)

# -----------------------------------------------------------------------------
# Page config & CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CoronaMax", page_icon="🏆", layout="wide")

HIDE_CHROME = """
<style>
/* masque le menu hamburger + footer streamlit en public */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(HIDE_CHROME, unsafe_allow_html=True)



# --- Diagnostics & compat Cloud --------------------------------------------
# Afficher le vrai traceback à l'écran (au lieu du "Oh no.")
try:
    st.set_option("client.showErrorDetails", True)
except Exception:
    pass

# Matplotlib en mode headless (Cloud)
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

# Détecter si la version Streamlit supporte la SelectboxColumn
_HAS_SELECTBOX_COL = hasattr(st, "column_config") and hasattr(st.column_config, "SelectboxColumn")

# -----------------------------------------------------------------------------
# Helpers UI
# -----------------------------------------------------------------------------
def list_files_sorted(root: Path, patterns: tuple[str,...] = ("*",)) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

def style_dataframe(d: pd.DataFrame) -> pd.io.formats.style.Styler:
    df = d.copy()

    # types
    int_cols = [c for c in ["Parties","Victoires","ITM","Recaves","Bulles","Place"] if c in df.columns]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    money_cols = [c for c in ["Recaves en €","Buy in","Frais","Gains","Bénéfices"] if c in df.columns]
    for c in money_cols:
        df[c] = pd.to_numeric(df[c].apply(parse_money), errors="coerce").fillna(0.0)

    pct_series = None
    if "% ITM" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            if "ITM" in df.columns and "Parties" in df.columns:
                df["% ITM"] = (df["ITM"] / df["Parties"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        pct_series = df["% ITM"].copy()

    sty = df.style.set_properties(**{"text-align":"center","font-size":"0.95rem","border-color":"#222"})
    sty = sty.set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#e6e6e6"),
                  ("font-weight", "bold"),("color","#000"),
                  ("border","1px solid #222")]
    }])

    if "Pseudo" in df.columns:
        def _pseudo_style(col):
            out=[]
            for val in col:
                if str(val).strip().lower()=="winamax":
                    out.append("background-color:#c62828;color:#fff;font-weight:900;text-transform:uppercase;"
                               "font-family:'League Gothic', Impact, sans-serif;letter-spacing:0.5px;")
                else:
                    out.append("background-color:#f7b329;color:#000;font-weight:700;")
            return out
        sty = sty.apply(_pseudo_style, subset=["Pseudo"])

    if "Parties" in df.columns:
        sty = sty.apply(lambda s: ["color:#d00000" if v < 20 else "color:#107a10" for v in s],
                        subset=["Parties"])

    if "Bénéfices" in df.columns:
        vals = df["Bénéfices"].apply(parse_money).astype(float)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        rng = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        def _benef_css(v):
            x = (parse_money(v) - vmin) / rng
            hue = int(120 * x)
            return f"background-color:hsl({hue},78%,62%); color:#111; font-weight:700;"
        sty = sty.apply(lambda s: [_benef_css(v) for v in df["Bénéfices"]], subset=["Bénéfices"])

    try:
        if "Pseudo" in df.columns:
            idx = df.index[df["Pseudo"].str.lower() == "winamax"][0]
            sty = sty.set_table_styles([{
                "selector": f"tbody tr:nth-child({idx+2})",
                "props": [("border-bottom","3px dashed #222")]
            }], overwrite=False)
    except Exception:
        pass

    fmt = {}
    for c in money_cols:
        fmt[c] = lambda x: euro(x)
    if pct_series is not None:
        fmt["% ITM"] = lambda x: f"{int(round(x*100, 0))}%" if pd.notnull(x) else "0%"
    sty = sty.format(fmt)
    return sty

def show_table(df, height: Optional[Union[int, str]] = None, caption: Optional[str] = None):
    """
    Affiche le tableau avec le style existant.
    - height: entier (px) ou "auto"/"stretch". Si None, on n'envoie pas le paramètre
      et on applique un CSS anti-scroll pour auto-ajuster la hauteur.
    """
    styled = style_dataframe(df)  # conserve ta mise en forme

    kwargs = dict(width="stretch", hide_index=True)

    if isinstance(height, int) or height in ("auto", "stretch"):
        kwargs["height"] = height
    else:
        # Fallback anti-scroll compatible toutes versions
        st.markdown(
            """
            <style>
            div[data-testid="stDataFrame"] div[role="grid"] { height: auto; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(styled, **kwargs)

    if caption:
        st.caption(caption)


# --- ROSTER joueurs (pseudos connus) ----------------------------------------
from app_classement_unique import DATA_DIR, load_results_log_any

ROSTER_CSV = DATA_DIR / "players_roster.csv"

def load_roster() -> list[str]:
    """Charge la liste des pseudos connus (depuis DATA/players_roster.csv).
    Si le fichier n'existe pas, on la dérive du results_log actuel."""
    try:
        if ROSTER_CSV.exists():
            df = pd.read_csv(ROSTER_CSV)
            pseudos = (
                df.get("Pseudo", pd.Series([], dtype=str))
                  .astype(str).str.strip()
                  .dropna().unique().tolist()
            )
        else:
            log = load_results_log_any()
            pseudos = (
                log.get("Pseudo", pd.Series([], dtype=str))
                   .astype(str).str.strip()
                   .dropna().unique().tolist()
            )
        # tri insensible à la casse, sans doublons vides
        pseudos = sorted({p for p in pseudos if p.strip()}, key=str.casefold)
        return pseudos
    except Exception:
        return []

def save_roster(pseudos: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Pseudo": sorted({p.strip() for p in pseudos if p.strip()}, key=str.casefold)})
    df.to_csv(ROSTER_CSV, index=False, encoding="utf-8")




# PDF → JPG (1re page) pour Archives
def pdf_first_page_to_jpg(pdf_path: Path, out_path: Path, dpi: int = 220) -> None:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        doc.close()
        return
    except Exception:
        pass
    # fallback “rien”
    raise RuntimeError("Conversion JPG indisponible (PyMuPDF manquant).")


# -----------------------------------------------------------------------------
# PUBLIC / ADMIN (une seule source)
# -----------------------------------------------------------------------------
IS_PUBLIC = os.getenv("CMX_MODE", "local").lower() == "public"

def _admin_key_expected() -> str:
    try:
        if "ADMIN_KEY" in st.secrets and st.secrets["ADMIN_KEY"]:
            return str(st.secrets["ADMIN_KEY"])
    except Exception:
        pass
    return os.getenv("ADMIN_KEY", "").strip() or "coronamax"

def is_admin() -> bool:
    if IS_PUBLIC:
        return False
    if os.getenv("ADMIN_MODE", "0") == "1":
        return True
    return bool(st.session_state.get("is_admin", False))

def ensure_admin() -> None:
    if not is_admin():
        st.error("⛔ Accès réservé à l’administrateur.")
        st.stop()

with st.sidebar.expander("🔐 Admin", expanded=not is_admin() and not IS_PUBLIC):
    if IS_PUBLIC:
        st.caption("Mode public : aucune action admin ici.")
    elif is_admin():
        st.success("Mode admin activé")
        if st.button("Se déconnecter", key="adm_off"):
            st.session_state["is_admin"] = False
            st.rerun()
    else:
        adm_try = st.text_input("Clé admin", type="password", placeholder="••••••••")
        if st.button("Activer", key="adm_on"):
            if adm_try.strip() == _admin_key_expected():
                st.session_state["is_admin"] = True
                st.success("Mode admin activé")
                st.rerun()
            else:
                st.error("Clé incorrecte")

NAV_PUBLIC = ["🏆 Tableau","👤 Détails joueur","📚 Archives","🏅 Classement par points"]
NAV_ADMIN  = NAV_PUBLIC + ["⬆️ Importer","♻️ Réinitialiser"]
page = st.sidebar.radio("Navigation", NAV_PUBLIC if (IS_PUBLIC or not is_admin()) else NAV_ADMIN,
                        key="nav_main")


# --- Helper hauteur auto pour tables ---------------------------------
def _auto_table_height(df, max_rows_no_scroll: int = 40,
                       row_px: int = 34, header_px: int = 38, padding_px: int = 16,
                       min_px: int = 200, max_px: Optional[int] = None) -> int:
    n = min(len(df), max_rows_no_scroll)
    h = header_px + padding_px + row_px * max(n, 1)
    if max_px is not None:
        h = min(h, max_px)
    return max(h, min_px)



# =============================================================================
# 1) 🏆 Tableau
# =============================================================================
if page == "🏆 Tableau":
    st.title("Classement général")

    log = load_results_log_any()
    if log is not None and not log.empty:
        s0, s1 = current_season_bounds()
        with st.expander("Filtrer par période (saison active par défaut)", expanded=False):
            d1 = st.date_input("Du", value=s0)
            d2 = st.date_input("Au", value=s1)
        sub = log.copy()
        sub = sub[(sub["start_time"].dt.date >= d1) & (sub["start_time"].dt.date <= d2)]
        table = standings_from_log(sub, season_only=False)
    else:
        table = load_latest_master_any()

    # --- Affichage conservant le style + hauteur auto -------------------
    height = _auto_table_height(table, max_rows_no_scroll=25)  # pas de scroll jusqu'à ~25 lignes
    try:
        # Si show_table accepte height (nouvelle signature)
        show_table(table, height=height)
    except TypeError:
        # Back-compat : si show_table(table) n'accepte pas height,
        # on force une hauteur auto par CSS puis on appelle show_table(table)
        st.markdown(
            """
            <style>
            /* Forcer l'auto-hauteur du grid quand il y a peu de lignes */
            div[data-testid="stDataFrame"] div[role="grid"] { height: auto; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        show_table(table)

    # Export CSV
    st.download_button("⬇️ Exporter (CSV affiché)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="classement.csv", type="secondary")

    # Export JPG
    export_cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
                   "Bulles","Buy in","Frais","Gains","Bénéfices"]
    export_df = table[[c for c in export_cols if c in table.columns]].copy()

    c1, c2 = st.columns([1,3])
    with c1:
        if st.button("🖼️ Exporter en JPG"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = SNAP_DIR / f"classement_{ts}.jpg"
            try:
                classement_df_to_jpg(export_df, out, title=None)
                st.success("JPG généré.")
                st.download_button("⬇️ Télécharger le JPG", data=out.read_bytes(),
                                   file_name=out.name, type="secondary")
            except Exception as e:
                st.error(f"Export impossible : {e}")

    # Publication snapshot (visible admin)
    if is_admin():
        st.subheader("Publication (snapshot public)")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("💾 Générer le snapshot local", key="snap_local"):
                ok, msg = publish_public_snapshot(push_to_github=False)
                (st.success if ok else st.error)("Snapshot local généré ✅" if ok else "Échec génération ❌")
                st.caption(msg)
        with cc2:
            if st.button("🚀 Publier sur GitHub", key="snap_push"):
                ok, msg = publish_public_snapshot(push_to_github=True, message="CoronaMax: MAJ snapshot public")
                (st.success if ok else st.error)("Publication effectuée ✅" if ok else "Échec publication ❌")
                st.caption(msg)
        st.caption("Local = écrit les CSV dans data/. GitHub = même chose + push dans ton dépôt public (data/**).")



# =============================================================================
# 2) 👤 Détails joueur
# =============================================================================

elif page == "👤 Détails joueur":
    import pandas as pd
    import numpy as np
    import altair as alt
    from datetime import date
    from app_classement_unique import (
        current_season_bounds, euro, parse_money, load_results_log_any
    )

    st.title("Fiche joueur")

    log = load_results_log_any()
    if log is None or log.empty:
        st.info("Aucun historique pour l’instant.")
        st.stop()

    # --------- Helpers robustes (colonnes variables) ----------
    def _pick_series(df: pd.DataFrame, candidates: list[str], default):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([default] * len(df), index=df.index)

    # borne par saison (défaut = saison courante)
    s0, s1 = current_season_bounds()
    with st.expander("Filtrer par période (saison active par défaut)", expanded=False):
        d1 = st.date_input("Du", value=s0)
        d2 = st.date_input("Au", value=s1)

    # choix du joueur (sans Winamax)
    pseudos = sorted([p for p in log["Pseudo"].dropna().unique().tolist()
                      if str(p).strip().lower() != "winamax"])
    if not pseudos:
        st.info("Aucun joueur trouvé.")
        st.stop()
    who = st.selectbox("Choisir un joueur", pseudos)

    # sous-ensemble période + joueur
    df = log.copy()
    if "start_time" in df.columns and not df["start_time"].isna().all():
        df["start_date"] = pd.to_datetime(df["start_time"]).dt.date
        df = df[(df["start_date"] >= d1) & (df["start_date"] <= d2)]
    df = df[df["Pseudo"] == who].copy()
    if df.empty:
        st.info("Pas de données pour ce joueur sur la période.")
        st.stop()

    # Colonnes clefs robustes
    pos      = pd.to_numeric(_pick_series(df, ["Position", "Place", "Rank"], 0), errors="coerce").fillna(0).astype(int)
    gaincash = _pick_series(df, ["GainsCash", "GainCash", "Gains"], 0.0).apply(parse_money).fillna(0.0)
    bounty   = _pick_series(df, ["Bounty"], 0.0).apply(parse_money).fillna(0.0)
    reentry  = pd.to_numeric(_pick_series(df, ["Reentry", "Re-entry"], 0), errors="coerce").fillna(0).astype(int)
    buyin_t  = _pick_series(df, ["buyin_total", "buy_in_total", "Buy in total"], 0.0).apply(parse_money).fillna(0.0)
    t_id     = _pick_series(df, ["tournament_id"], "").astype(str)
    t_name   = _pick_series(df, ["tournament_name"], "").astype(str)
    start_ts = pd.to_datetime(_pick_series(df, ["start_time"], pd.NaT), errors="coerce")

    # ITM (cash hors bounty)
    itm_flag = (gaincash > 0).astype(int)
    # Victoires
    win_flag = (pos == 1).astype(int)

    # -------- Bulles (corrigé) : calculées sur le log complet, puis on regarde si 'who' est le 1er non-payé
    def bubble_map(full_log: pd.DataFrame) -> dict:
        if full_log.empty:
            return {}
        L = {}
        g = full_log.copy()
        g["gc"] = _pick_series(g, ["GainsCash", "GainCash", "Gains"], 0.0).apply(parse_money).fillna(0.0)
        g["pos"] = pd.to_numeric(_pick_series(g, ["Position", "Place", "Rank"], 0), errors="coerce").fillna(0).astype(int)
        for tid, grp in g.groupby(_pick_series(g, ["tournament_id"], "").astype(str)):
            grp = grp.sort_values("pos")
            no_paid = grp[grp["gc"] <= 0.0]
            if no_paid.empty:
                continue
            L[tid] = str(no_paid.iloc[0]["Pseudo"])
        return L

    bubbles_by_tid = bubble_map(log)
    bubble_flag = t_id.map(lambda tid: 1 if bubbles_by_tid.get(tid) == who else 0)

    # Recaves € / Frais / Gains totaux / Bénéfices
    recaves_euro = reentry * buyin_t
    frais = buyin_t + recaves_euro
    gains_tot = gaincash + bounty
    benef = gains_tot - frais

    # ---- Agrégats
    parties    = int(len(df))
    wins       = int(win_flag.sum())
    itm        = int(itm_flag.sum())
    pct_itm    = (itm / parties) * 100 if parties else 0.0
    bulles     = int(bubble_flag.sum())
    recaves_nb = int(reentry.sum())
    recaves_e  = float(recaves_euro.sum())
    buyins     = float(buyin_t.sum())
    total_fees = float(frais.sum())
    g_cash     = float(gaincash.sum())
    g_bounty   = float(bounty.sum())
    g_total    = float(gains_tot.sum())
    net        = float(benef.sum())
    roi_total  = (g_total - total_fees) / total_fees if total_fees > 0 else 0.0  # ROI sur gains totaux uniquement
    avg_finish = float(pos.replace(0, np.nan).mean()) if (pos > 0).any() else 0.0

    # Points (formule N_participants - pos + 1, min=1) sur la période
    log_period = log.copy()
    if "start_time" in log_period.columns and not log_period["start_time"].isna().all():
        log_period["start_date"] = pd.to_datetime(log_period["start_time"]).dt.date
        log_period = log_period[(log_period["start_date"] >= d1) & (log_period["start_date"] <= d2)]
    lp_pos   = pd.to_numeric(_pick_series(log_period, ["Position", "Place", "Rank"], 0), errors="coerce").fillna(0).astype(int)
    lp_tid   = _pick_series(log_period, ["tournament_id"], "").astype(str)
    lp_pseudo= _pick_series(log_period, ["Pseudo"], "").astype(str)

    if not log_period.empty:
        n_part = log_period.groupby(lp_tid).transform("size")
        pts_row = (n_part - lp_pos + 1).clip(lower=1)
        pts_total = int(pts_row[lp_pseudo == who].sum())
    else:
        pts_total = 0

    # ---- TUILES (comme avant)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parties", parties)
    c2.metric("Victoires", wins)
    c3.metric("ITM", f"{itm} ({pct_itm:.0f}%)")
    c4.metric("Bulles", bulles)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Buy-ins", euro(buyins))
    c6.metric("Recaves (nb)", recaves_nb)
    c7.metric("Recaves en €", euro(recaves_e))
    c8.metric("Frais totaux", euro(total_fees))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Gains (cash)", euro(g_cash))
    c10.metric("Bounty", euro(g_bounty))
    c11.metric("Gains totaux", euro(g_total))
    c12.metric("Bénéfices", euro(net))

    c13, c14, c15 = st.columns(3)
    c13.metric("Place moyenne", f"{avg_finish:.2f}" if avg_finish else "—")
    c14.metric("Points", pts_total)
    c15.metric("ROI (total)", f"{roi_total:.0%}")

    # ---- Graphiques
    st.subheader("Évolution des bénéfices (cumul)")
    evo = pd.DataFrame({
        "Date": start_ts,
        "Bénéfices": benef
    }).dropna(subset=["Date"]).sort_values("Date")
    if not evo.empty:
        evo["Cumul"] = evo["Bénéfices"].cumsum()
        st.altair_chart(
            alt.Chart(evo).mark_line(point=True).encode(
                x="Date:T", y="Cumul:Q"
            ).properties(height=300),
            use_container_width=True
        )
    else:
        st.caption("Aucune donnée temporelle exploitable.")

    st.subheader("Distribution des places")
    places = pos[pos > 0]
    if not places.empty:
        dist = places.value_counts().sort_index().reset_index()
        dist.columns = ["Place", "Occurrences"]
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(
                x=alt.X("Place:O", sort="ascending"),
                y="Occurrences:Q"
            ).properties(height=260),
            use_container_width=True
        )
    else:
        st.caption("Aucune place enregistrée.")

    # ---- Historique détaillé
    st.subheader("Historique")
    hist = pd.DataFrame({
        "Date": start_ts.dt.strftime("%Y-%m-%d %H:%M"),
        "Tournoi": t_name,
        "Place": pos,
        "Reentry": reentry,
        "Buy in": buyin_t.apply(euro),
        "Frais": frais.apply(euro),
        "GainsCash": gaincash.apply(euro),
        "Bounty": bounty.apply(euro),
        "Gains": gains_tot.apply(euro),
        "Bénéfices": benef.apply(euro),
    })
    hist = hist.sort_values("Date")
    st.dataframe(hist, use_container_width=True, hide_index=True)



# =============================================================================
# 3) 📚 Archives
# =============================================================================
elif page == "📚 Archives":
    st.title("Archives")

    # Classement “à la date” (basé sur la DATE DE TOURNOI)
    st.subheader("Classement « à la date »")
    log = load_results_log_any()
    if log.empty:
        st.info("Pas d’historique pour le moment.")
    else:
        d = st.date_input("Afficher l’état au", value=date.today())
        sub = log[log["start_time"].dt.date <= d].copy()
        table = standings_from_log(sub, season_only=False)
        show_table(table, caption=f"État arrêté au {d:%d/%m/%Y}")

    # PDFs archivés
    st.subheader("PDFs archivés (par saison)")
    pdfs = list_files_sorted(PDF_DONE, ("*.pdf",))
    if not pdfs:
        st.caption("Aucun PDF archivé.")
    else:
        with st.expander("Saison courante", expanded=True):
            for p in pdfs:
                cols = st.columns([6, 2, 2])
                cols[0].write(f"**{p.name}**  \n_{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}_")
                with cols[1]:
                    st.download_button("Télécharger (PDF)", data=p.read_bytes(),
                                       file_name=p.name, type="secondary", key=f"dlpdf_{p.name}")
                with cols[2]:
                    try:
                        jpg_path = SNAP_DIR / "archived_jpg" / (p.stem + ".jpg")
                        need_regen = (not jpg_path.exists()) or (jpg_path.stat().st_mtime < p.stat().st_mtime)
                        if need_regen:
                            pdf_first_page_to_jpg(p, jpg_path, dpi=220)
                        st.download_button("Télécharger (JPG)", data=jpg_path.read_bytes(),
                                           file_name=jpg_path.name, type="secondary", key=f"dljpg_{p.name}")
                    except Exception as e:
                        st.button("JPG indisponible", disabled=True, key=f"nojpg_{p.name}")
                        st.caption(f"⚠️ Conversion JPG échouée : {e}")


# =============================================================================
# 4) 🏅 Classement par points
# =============================================================================
elif page == "🏅 Classement par points":
    st.title("Classement général — Points")

    log = load_results_log_any()
    if log is None or log.empty:
        st.info("Aucune donnée pour l’instant. Va dans **Importer** pour traiter des PDFs.")
    else:
        s0, s1 = current_season_bounds()
        with st.expander("Filtrer par période (saison active par défaut)", expanded=False):
            d1 = st.date_input("Du", value=s0)
            d2 = st.date_input("Au", value=s1)

        pts = compute_points_table(log, d1, d2)
        if pts.empty:
            st.info("Pas de résultats sur cette période.")
        else:
            st.dataframe(pts, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Exporter (CSV Points)",
                data=pts.to_csv(index=False).encode("utf-8"),
                file_name="classement_points.csv", type="secondary")

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("🖼️ Exporter le classement Points en JPG"):
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        jpg_path = SNAP_DIR / f"classement_points_{ts}.jpg"
                        classement_points_df_to_jpg(pts, jpg_path)
                        st.session_state["last_points_jpg_export"] = str(jpg_path)
                        st.success("JPG (Points) généré.")
                    except Exception as e:
                        st.error(f"Export impossible : {e}")
            with c2:
                jp = st.session_state.get("last_points_jpg_export")
                if jp and Path(jp).exists():
                    st.download_button("⬇️ Télécharger le JPG (Points)",
                        data=Path(jp).read_bytes(),
                        file_name=Path(jp).name, type="secondary")


# =============================================================================
# 5) ⬆️ Importer (ADMIN)
# =============================================================================
elif page == "⬆️ Importer":
    ensure_admin()
    st.title("Importer des résultats (PDF Winamax)")
    st.session_state.setdefault("pending_tourneys", {})
    # Import local pour la saisie manuelle (évite d'éditer le header du fichier)
    from app_classement_unique import build_manual_rows_for_log

    if "pending_tourneys" not in st.session_state:
        st.session_state["pending_tourneys"] = {}

    # --- Import PDF classique -------------------------------------------------
    up = st.file_uploader("Déposez un ou plusieurs PDFs", type=["pdf"], accept_multiple_files=True)
    if up and st.button("Analyser et mettre en file", type="primary"):
        logs = []
        for f in up:
            tmp = PDF_DIR / f.name
            tmp.write_bytes(f.read())
            try:
                parsed = extract_from_pdf(tmp)
                rows_preview = build_rows_for_log(parsed)
                if rows_preview.empty:
                    st.warning(f"⚠️ {f.name} : 0 ligne détectée (non ajouté).")
                    safe_unlink(tmp)
                    continue
                st.session_state.pending_tourneys[parsed.tournament_id] = {
                    "df": rows_preview,
                    "src_path": str(tmp),
                    "name": parsed.tournament_name,
                    "start_time": parsed.start_time,
                    "is_manual": False,
                }
                logs.append(f"✅ Ajouté : {f.name} — {len(rows_preview)} ligne(s)")
            except Exception as e:
                safe_unlink(tmp)
                logs.append(f"❌ Erreur sur {f.name} : {e}")
        if logs:
            st.text("\n".join(logs))
        st.info("Faites défiler pour valider chaque tournoi.")

# --- Saisie manuelle (NOUVEAU) -------------------------------------------

# Compat Cloud : certaines versions n'ont pas SelectboxColumn
_HAS_SELECTBOX_COL = hasattr(st, "column_config") and hasattr(st.column_config, "SelectboxColumn")



with st.expander("➕ Saisie manuelle d'un tournoi (sans PDF)", expanded=False):
    t_name = st.text_input("Nom du tournoi", placeholder="Ex. CoronaMax #123")

    cdt, cht, cbi = st.columns([2, 1.3, 1.7])
    with cdt:
        t_date = st.date_input("Date", value=date.today(), key="manual_date")
    with cht:
        t_time = st.time_input("Heure", value=datetime.now().replace(second=0, microsecond=0).time(), key="manual_time")
    with cbi:
        t_buyin = st.number_input("Buy-in total (Buy-in + Rake)", min_value=0.0, step=0.5, format="%.2f", key="manual_buyin")

    # --- Helpers roster (pseudos connus) ---------------------------------
    def _load_roster() -> List[str]:
        try:
            from app_classement_unique import DATA_DIR, load_results_log_any
            roster_csv = Path(DATA_DIR) / "players_roster.csv"
            if roster_csv.exists():
                df = pd.read_csv(roster_csv)
                pseudos = df.get("Pseudo", pd.Series([], dtype=str)).astype(str).str.strip()
                return sorted({p for p in pseudos if p}, key=str.casefold)
            # sinon on dérive du log actuel
            log = load_results_log_any()
            pseudos = log.get("Pseudo", pd.Series([], dtype=str)).astype(str).str.strip()
            return sorted({p for p in pseudos if p}, key=str.casefold)
        except Exception:
            return []

    def _save_roster(pseudos: List[str]) -> None:
        try:
            from app_classement_unique import DATA_DIR
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            roster_csv = Path(DATA_DIR) / "players_roster.csv"
            df = pd.DataFrame({"Pseudo": sorted({p.strip() for p in pseudos if isinstance(p, str) and p.strip()}, key=str.casefold)})
            df.to_csv(roster_csv, index=False, encoding="utf-8")
        except Exception as e:
            print(f"[Importer] WARNING: save roster failed: {e}")

    known_pseudos = _load_roster()
    assist = st.toggle("Saisie assistée (auto-complétion des pseudos)", value=True, key="manual_assist")

    # Table d'édition des lignes
    if "manual_rows_df" not in st.session_state:
        st.session_state.manual_rows_df = pd.DataFrame([
            {"Position": 1, "Pseudo": "", "GainsCash": 0.0, "Bounty": 0.0, "Reentry": 0},
            {"Position": 2, "Pseudo": "", "GainsCash": 0.0, "Bounty": 0.0, "Reentry": 0},
            {"Position": 3, "Pseudo": "", "GainsCash": 0.0, "Bounty": 0.0, "Reentry": 0},
        ])

    colcfg = {
        "Position": st.column_config.NumberColumn("Position", step=1, format="%d", help="Place (1,2,3,...)"),
        "Pseudo": (
            st.column_config.SelectboxColumn(
                "Pseudo",
                options=known_pseudos,
                help="Tape pour filtrer les pseudos connus"
            ) if (assist and _HAS_SELECTBOX_COL) else
            st.column_config.TextColumn("Pseudo", help="Saisie libre")
        ),
        "GainsCash": st.column_config.NumberColumn("Gains (€)", format="%.2f", help="Gains cash"),
        "Bounty": st.column_config.NumberColumn("Bounty (€)", format="%.2f"),
        "Reentry": st.column_config.NumberColumn("Recaves", step=1, format="%d"),
    }



    manual_edit = st.data_editor(
        st.session_state.manual_rows_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="manual_editor",
        column_config=colcfg,
    )

    # Aperçu bulle
    try:
        _bubble = compute_bubble_from_rows(manual_edit)
        st.caption(f"Bulle détectée : **{_bubble or '(aucune)'}**")
    except Exception:
        pass

    # Mise en file
    if st.button("➕ Mettre en file (saisie manuelle)", type="primary", key="manual_queue"):
        if not t_name.strip():
            st.warning("Renseigne d'abord le nom du tournoi.")
        else:
            dt = datetime.combine(t_date, t_time)
            df_manual = build_manual_rows_for_log(t_name.strip(), dt, float(t_buyin), manual_edit.copy())
            # filtre sécurité : au moins 1 pseudo non vide
            has_rows = (df_manual["Pseudo"].astype(str).str.strip() != "").any()
            if not has_rows:
                st.warning("Ajoute au moins une ligne avec un pseudo.")
            else:
                # Mettre à jour le roster dès maintenant (optionnel mais pratique)
                try:
                    new_pseudos = df_manual["Pseudo"].astype(str).str.strip().tolist()
                    merged = sorted(set(known_pseudos) | {p for p in new_pseudos if p}, key=str.casefold)
                    _save_roster(merged)
                except Exception as e:
                    print(f"[Importer] WARNING: roster update failed: {e}")

                tid = str(df_manual["tournament_id"].iloc[0])
                st.session_state.pending_tourneys[tid] = {
                    "df": df_manual,
                    "src_path": "",  # pas de PDF
                    "name": t_name.strip(),
                    "start_time": dt,
                    "is_manual": True,
                }
                st.success(f"✅ Ajouté : {t_name.strip()} — {len(df_manual)} ligne(s)")



# --- File d'attente / Validation -----------------------------------------
if not st.session_state.get("pending_tourneys"):
    st.caption("Aucun tournoi en attente de validation.")
else:
    st.subheader("Tournois en attente de validation")
    to_rerun = False

    for tid, item in list(st.session_state.get("pending_tourneys", {}).items()):
        st.markdown(f"**{item['name']} — {pd.to_datetime(item['start_time']):%d/%m/%Y %H:%M}**")
        edit = st.data_editor(
            item["df"], num_rows="dynamic", width="stretch", hide_index=True, key=f"edit_{tid}"
        )
        bubble_name = compute_bubble_from_rows(edit)
        st.info(f"**Bulle détectée :** {bubble_name or '(aucune)'}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Valider ce tournoi", key=f"commit_{tid}"):
                # 1) Append au log
                append_results_log(edit.copy())
                
                try:
                    new_pseudos = (
                        edit.get("Pseudo")
                            .astype(str).str.strip()
                            .dropna().tolist()
                    )
                    merged = sorted(set(load_roster()) | {p for p in new_pseudos if p}, key=str.casefold)
                    save_roster(merged)
                except Exception as e:
                    print(f"[Importer] WARNING: roster update failed: {e}")
                    
                # 2) Journal + archivage conditionnels (robuste)
                journal = load_journal()

                # srcp peut être None si saisie manuelle ou si la génération PDF échoue
                srcp = Path(item.get("src_path")) if item.get("src_path") else None

                # ➕ Si pas de PDF (saisie manuelle), on génère un PDF temporaire dans PDF_DIR
                if (srcp is None) or (not srcp.is_file()):
                    try:
                        from app_classement_unique import render_manual_results_pdf
                        srcp = render_manual_results_pdf(
                            edit.copy(),
                            item["name"],
                            pd.to_datetime(item["start_time"])
                        )
                    except Exception as e:
                        print(f"[Importer] render_manual_results_pdf KO: {e}")
                        srcp = None  # ❗️ surtout pas Path("")

                filename = srcp.name if (srcp and srcp.is_file()) else f"(saisie manuelle) {item['name']}"
                journal.loc[len(journal)] = {
                    "sha1": tid,
                    "filename": filename,
                    "processed_at": datetime.now(),
                }
                save_journal(journal)

                if srcp and srcp.is_file():
                    archive_pdf(srcp)

                # retirer de la file
                del st.session_state["pending_tourneys"][tid]

                # 3) Rebuild des snapshots (gains + points + miroirs publics)
                from app_classement_unique import (
                    load_results_log_any,
                    standings_from_log,
                    compute_points_table,
                    current_season_bounds,
                    DATA_DIR,
                )

                cur_log = load_results_log_any()
                table_gains = standings_from_log(cur_log, season_only=False)

                # Points sur la saison courante
                s0, s1 = current_season_bounds()
                table_points = compute_points_table(cur_log, d1=s0, d2=s1)

                DATA_DIR.mkdir(parents=True, exist_ok=True)

                # Gains
                (DATA_DIR / "latest_master.csv").write_text(
                    table_gains.to_csv(index=False), encoding="utf-8"
                )
                # Points (saison en cours)
                (DATA_DIR / "points_table.csv").write_text(
                    table_points.to_csv(index=False), encoding="utf-8"
                )
                # Journal public
                (DATA_DIR / "journal.csv").write_text(
                    journal.to_csv(index=False), encoding="utf-8"
                )
                # Miroir du results_log (utile si append_results_log ne le fait pas déjà)
                try:
                    (DATA_DIR / "results_log.csv").write_text(
                        cur_log.to_csv(index=False), encoding="utf-8"
                    )
                except Exception as e:
                    print(f"[Importer] WARNING: miroir data/results_log.csv KO: {e}")

                # 4) Diagnostics
                st.caption(f"📄 ARCHIVE/results_log.csv → {len(cur_log)} lignes")
                st.caption(f"🧮 latest_master.csv: {len(table_gains)} lignes")
                st.caption(f"🏅 points_table.csv: {len(table_points)} lignes")

                st.success("Tournoi ajouté. Classement mis à jour.")
                to_rerun = True

        with c2:
            if st.button("🗑️ Annuler / retirer de la file", key=f"cancel_{tid}"):
                srcp = Path(item.get("src_path")) if item.get("src_path") else None
                if srcp and srcp.is_file():
                    safe_unlink(srcp)
                # protéger l'accès même si clé absente
                pend = st.session_state.get("pending_tourneys", {})
                if tid in pend:
                    del pend[tid]
                st.warning("Tournoi retiré de la file.")
                to_rerun = True

        st.divider()

    if to_rerun:
        st.rerun()

    # Rollback
    st.subheader("Annuler le dernier import")
    if st.button("🧹 Annuler le dernier tournoi importé"):
        info = rollback_last_import()
        if info.get("ok"):
            st.success(f"{info['msg']} — PDF remis: {info.get('pdf_back','')}")
            st.rerun()
        else:
            st.info(info.get("msg", "Rien à annuler."))


# =============================================================================
# 6) ♻️ Réinitialiser (ADMIN)
# =============================================================================
if page == "♻️ Réinitialiser":
    ensure_admin()
    st.title("Réinitialiser la saison courante")
    st.warning("Attention : remise à zéro des agrégats. Les PDFs archivés peuvent être re-traités ensuite.")

    do_move = st.checkbox("Remettre les PDFs archivés dans « PDF_A_TRAITER » (suppression horodatage)", value=True)

    if st.button("⚠️ Lancer la réinitialisation", type="primary"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        BACKUPS_DIR = RESULTS_LOG.parent / "BACKUPS"
        BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if RESULTS_LOG.exists():
                shutil.copy2(RESULTS_LOG, BACKUPS_DIR / f"results_log_{ts}.csv")
        except Exception as e:
            st.warning(f"Backup raté pour results_log.csv : {e}")
        try:
            if JOURNAL_CSV.exists():
                shutil.copy2(JOURNAL_CSV, BACKUPS_DIR / f"journal_{ts}.csv")
        except Exception as e:
            st.warning(f"Backup raté pour journal.csv : {e}")

        pd.DataFrame(columns=RESULTS_LOG_COLUMNS).to_csv(RESULTS_LOG, index=False, encoding="utf-8")
        pd.DataFrame(columns=JOURNAL_COLUMNS).to_csv(JOURNAL_CSV, index=False, encoding="utf-8")

        if do_move:
            PDF_DIR.mkdir(parents=True, exist_ok=True)

            def _unique_dest(base_dir: Path, name: str) -> Path:
                dest = base_dir / name
                if not dest.exists():
                    return dest
                stem, suffix = Path(name).stem, Path(name).suffix
                n = 1
                while True:
                    cand = base_dir / f"{stem}_dup{n}{suffix}"
                    if not cand.exists():
                        return cand
                    n += 1

            for p in list_files_sorted(PDF_DONE, ("*.pdf",)):
                clean_name = (p.name.split("__")[0] + ".pdf") if "__" in p.name else p.name
                dst = _unique_dest(PDF_DIR, clean_name)
                try:
                    shutil.move(str(p), str(dst))
                except Exception as e:
                    st.warning(f"Impossible de déplacer {p.name} -> {dst.name} : {e}")

        st.success("Réinitialisation OK. Allez dans **Importer** pour revalider.")
