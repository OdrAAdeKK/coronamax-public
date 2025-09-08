# App.py
from __future__ import annotations
import os
from datetime import datetime, date
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import shutil

from app_classement_unique import (
    BASE, PDF_DIR, PDF_DONE, SNAP_DIR, DATA_DIR,
    RESULTS_LOG, JOURNAL_CSV,
    # loaders & utils (UNIFIÉS)
    load_results_log_any, load_journal_any, load_latest_master_any,
    load_results_log, append_results_log, load_journal, save_journal,
    build_rows_for_log, extract_from_pdf, standings_from_log, compute_points_table,
    publish_public_snapshot, archive_pdf, rollback_last_import,
    parse_money, euro, current_season_bounds, classement_df_to_jpg, compute_bubble_from_rows,
    classement_points_df_to_jpg,
)

st.set_page_config(page_title="CoronaMax", page_icon="🏆", layout="wide")

# ---- Thème + police (à placer après st.set_page_config)
def _inject_global_css():
    st.markdown("""
    <style>
    /* Police League Gothic via jsdelivr (léger et gratuit) */
    @font-face {
      font-family: 'League Gothic';
      src: url('https://cdn.jsdelivr.net/gh/theleagueof/league-gothic/webfonts/leaguegothic-regular-webfont.woff2') format('woff2'),
           url('https://cdn.jsdelivr.net/gh/theleagueof/league-gothic/webfonts/leaguegothic-regular-webfont.woff') format('woff');
      font-weight: 700;
      font-style: normal;
      font-display: swap;
    }

    /* Arrondis homogènes des dataframes Streamlit */
    .stDataFrame .row-heading.level0,
    .stDataFrame .col_heading,
    .stDataFrame [data-testid="stTable"] table {
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

_inject_global_css()


CMX_MODE = os.getenv("CMX_MODE", "private")  # "public" -> lecture seule
IS_PUBLIC = CMX_MODE == "public"


# --- Schémas canon des CSV publics/local ---
RESULTS_LOG_COLUMNS = [
    "tournament_id",     # identifiant (sha1)
    "tournament_name",   # nom court du tournoi
    "start_time",        # horodatage officiel du tournoi (ISO)
    "Pseudo",            # joueur
    "Position",          # place dans le tournoi (1..N)
    "GainsCash",         # gains hors bounty (euros)
    "Bounty",            # bounty (euros)
    "Reentry",           # nombre de recaves (entier)
    "buyin_total",       # buy-in + rake (euros)
    "processed_at",      # date/heure de traitement (ISO)
]

JOURNAL_COLUMNS = [
    "sha1",              # identifiant (sha1)
    "filename",          # nom du PDF archivisé
    "processed_at",      # date/heure de traitement (ISO)
]

# ==========
# Sidebar (unique)
# ==========
st.sidebar.title("CoronaMax")
ALL_PAGES  = ["🏆 Tableau","👤 Détails joueur","📚 Archives","🏅 Classement par points","⬆️ Importer","♻️ Réinitialiser"]
READ_PAGES = ["🏆 Tableau","👤 Détails joueur","📚 Archives","🏅 Classement par points"]
page = st.sidebar.radio("Navigation", READ_PAGES if IS_PUBLIC else ALL_PAGES, key="nav_main")

# --- à mettre près des imports en haut de App.py ---
import time
from pathlib import Path  # si pas déjà présent

# --- PDF → JPG (first page) ---------------------------------------------------
def pdf_first_page_to_jpg(pdf_path: Path, jpg_path: Path, dpi: int = 220) -> Path:
    """
    Render the first page of a PDF to a JPG file.
    Uses PyMuPDF (fitz). Creates parent folder if needed.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF (fitz) n'est pas installé.") from e

    jpg_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            raise RuntimeError("PDF vide")
        page = doc.load_page(0)
        # zoom from DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(jpg_path)
    finally:
        doc.close()
    return jpg_path


def safe_unlink(p: Path, retries: int = 5, delay: float = 0.2) -> None:
    """Supprime un fichier en réessayant si Windows le garde momentanément verrouillé."""
    for _ in range(retries):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
    # dernier essai (on ignore si ça échoue encore)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

# ==========
# Helpers UI
# ==========
def list_files_sorted(folder: Path, patterns=("*.pdf",)):
    out = []
    for pat in patterns: out.extend(folder.glob(pat))
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

def style_dataframe(d: pd.DataFrame) -> pd.io.formats.style.Styler:
    import numpy as np  # <- ensure np exists in this scope

    df = d.copy()

    # — Forçage types (robuste)
    int_cols = [c for c in ["Parties","Victoires","ITM","Recaves","Bulles","Place"] if c in df.columns]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    money_cols = [c for c in ["Recaves en €","Buy in","Frais","Gains","Bénéfices"] if c in df.columns]
    for c in money_cols:
        # conserve des floats (on formattera plus bas avec euro())
        df[c] = pd.to_numeric(df[c].apply(parse_money), errors="coerce").fillna(0.0)

    # % ITM (ratio; on formatera en % plus tard)
    if "% ITM" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            if "ITM" in df.columns and "Parties" in df.columns:
                df["% ITM"] = (df["ITM"] / df["Parties"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        pct_series = df["% ITM"].copy()
    else:
        pct_series = None

    # — Construction Styler
    sty = df.style.set_properties(**{
        "text-align": "center",
        "font-size": "0.95rem",
        "border-color": "#222"
    })

    # 1) En-têtes gris + gras
    sty = sty.set_table_styles([{
        "selector": "th",
        "props": [("background-color", "#e6e6e6"),
                  ("font-weight", "bold"),
                  ("color", "#000"),
                  ("border", "1px solid #222")]
    }])

    # 2) Colonne Pseudo : WINAMAX rouge/blanc + autres orange/noir gras
    if "Pseudo" in df.columns:
        def _pseudo_style(col):
            out = []
            for val in col:
                if str(val).strip().lower() == "winamax":
                    out.append(
                        "background-color:#c62828;color:#fff;font-weight:900;"
                        "text-transform:uppercase;font-family:'League Gothic', Impact, sans-serif;"
                        "letter-spacing:0.5px;"
                    )
                else:
                    out.append("background-color:#f7b329;color:#000;font-weight:700;")
            return out
        sty = sty.apply(_pseudo_style, subset=["Pseudo"])

    # 3) Parties rouge (<20) / vert (≥20)
    if "Parties" in df.columns:
        sty = sty.apply(
            lambda s: ["color:#d00000" if v < 20 else "color:#107a10" for v in s],
            subset=["Parties"]
        )

    # 4) Dégradé lisible sur Bénéfices (vert ↔ rouge, plus contrasté)
    if "Bénéfices" in df.columns:
        vals = df["Bénéfices"].astype(float)
        if len(vals) == 0:
            vmin, vmax, rng = 0.0, 1.0, 1.0
        else:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            rng = (vmax - vmin) if (vmax - vmin) != 0 else 1.0

        def _benef_css(v):
            try:
                x = (float(v) - vmin) / rng  # 0..1
            except Exception:
                x = 0.5
            x = 0.0 if np.isnan(x) else min(max(x, 0.0), 1.0)
            hue = int(120 * x)                 # 0 = rouge -> 120 = vert
            # plus sombre + texte noir lisible
            return f"background-color:hsl({hue},78%,62%); color:#111; font-weight:700;"

        sty = sty.apply(lambda s: [_benef_css(v) for v in df["Bénéfices"]],
                        subset=["Bénéfices"])

    # 5) Ligne pointillée SOUS Winamax
    try:
        if "Pseudo" in df.columns:
            idx = df.index[df["Pseudo"].astype(str).str.lower() == "winamax"][0]
            # nth-child = +2 (1 = en-têtes). Applique aux <td> pour forcer l'affichage.
            sty = sty.set_table_styles([{
                "selector": f"tbody tr:nth-child({idx+2}) td",
                "props": [("border-bottom","3px dashed #222")]
            }], overwrite=False)
    except Exception:
        pass

    # 6) Formats d'affichage finaux
    fmt = {}
    for c in money_cols:
        fmt[c] = lambda x: euro(x)  # "0,00 €"
    if pct_series is not None:
        fmt["% ITM"] = lambda x: f"{int(round((0 if pd.isna(x) else x)*100, 0))}%"

    sty = sty.format(fmt)

    return sty


def show_table(df: pd.DataFrame, caption: str | None = None):
    st.dataframe(style_dataframe(df), use_container_width=True, hide_index=True)
    if caption: st.caption(caption)

def seasons_available(log: pd.DataFrame) -> list[str]:
    if log.empty or "start_time" not in log.columns: return []
    years = sorted({(d.year if d.month>=8 else d.year-1) for d in log["start_time"].dt.date})
    return [f"{y}-{y+1}" for y in years]

def pick_season(log: pd.DataFrame):
    s0, s1 = current_season_bounds()
    opts = seasons_available(log)
    default = f"{s0.year}-{s1.year}"
    sel = st.selectbox("Saison", opts or [default], index=(opts.index(default) if default in opts else 0))
    y = int(sel.split("-")[0])
    return date(y,8,1), date(y+1,7,31), sel

def compute_bubble_from_rows(df_rows: pd.DataFrame) -> str | None:
    if df_rows is None or df_rows.empty: return None
    d = df_rows.copy()
    pos = pd.to_numeric(d.get("Position", 0), errors="coerce").fillna(0).astype(int)
    gains = pd.to_numeric(d.get("GainsCash", 0.0), errors="coerce").fillna(0.0)
    pseudo= d.get("Pseudo", "").astype(str)
    tmp = pd.DataFrame({"Position":pos,"GainsCash":gains,"Pseudo":pseudo}).sort_values("Position")
    no_paid = tmp[tmp["GainsCash"] <= 0.0]
    return None if no_paid.empty else str(no_paid.iloc[0]["Pseudo"])


# ==========
# PAGE 1 — Tableau
# ==========
if page == "🏆 Tableau":
    st.title("Classement général")

    if IS_PUBLIC:
        log = load_results_log_any()
        table = load_latest_master_any()
        if table.empty and not log.empty:
            s0, s1 = current_season_bounds()
            sub = log[(log["start_time"].dt.date >= s0) & (log["start_time"].dt.date <= s1)]
            table = standings_from_log(sub, season_only=False)
    else:
        if RESULTS_LOG.exists() and RESULTS_LOG.stat().st_size > 0:
            log = load_results_log_any()
            d1, d2, _ = pick_season(log)
            with st.expander("Filtrer par période (saison active par défaut)", expanded=False):
                d1 = st.date_input("Du", value=d1)
                d2 = st.date_input("Au", value=d2)
            sub = log[(log["start_time"].dt.date >= d1) & (log["start_time"].dt.date <= d2)]
            table = standings_from_log(sub, season_only=False)
        else:
            table = load_latest_master_any()

    if table.empty:
        st.info("Aucune donnée disponible pour l’instant.")
    else:
        show_table(table)
        st.download_button("⬇️ Exporter (CSV affiché)", data=table.to_csv(index=False).encode("utf-8"),
                           file_name="classement.csv", type="secondary")

        # Export JPG
        export_cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
                       "Bulles","Buy in","Frais","Gains","Bénéfices"]
        export_df = table[[c for c in export_cols if c in table.columns]].copy()
        c1, c2 = st.columns([1,3])
        with c1:
            if st.button("🖼️ Exporter le tableau en JPG"):
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
                st.download_button("⬇️ Télécharger le JPG", data=Path(jp).read_bytes(),
                                   file_name=Path(jp).name, type="secondary")

    # Publication snapshot (local uniquement)
    if not IS_PUBLIC:
        st.subheader("Publication (snapshot public)")
        if st.button("📤 Générer le snapshot local"):
            ok, msg = publish_public_snapshot()
            st.success("Snapshot publique publiée ✅" if ok else "Échec") 
            st.caption(msg)

# ==========
# PAGE 2 — Importer (LOCAL uniquement)
# ==========
elif page == "⬆️ Importer" and not IS_PUBLIC:
    st.title("Importer des résultats (PDF Winamax)")
    if "pending_tourneys" not in st.session_state:
        st.session_state["pending_tourneys"] = {}

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
                    tmp.unlink(missing_ok=True); continue
                st.session_state.pending_tourneys[parsed.tournament_id] = {
                    "df": rows_preview, "src_path": str(tmp),
                    "name": parsed.tournament_name, "start_time": parsed.start_time
                }
                logs.append(f"✅ Ajouté : {f.name} — {len(rows_preview)} ligne(s)")
            except Exception as e:
                safe_unlink(tmp)
                logs.append(f"❌ Erreur sur {f.name} : {e}")
        if logs: st.text("\n".join(logs))
        st.info("Faites défiler pour valider chaque tournoi.")

    if not st.session_state.pending_tourneys:
        st.caption("Aucun tournoi en attente de validation.")
    else:
        st.subheader("Tournois en attente de validation")
        to_rerun = False
        for tid, item in list(st.session_state.pending_tourneys.items()):
            st.markdown(f"**{item['name']} — {pd.to_datetime(item['start_time']):%d/%m/%Y %H:%M}**")
            edit = st.data_editor(item["df"], num_rows="dynamic", width="stretch", hide_index=True, key=f"edit_{tid}")
            bubble_name = compute_bubble_from_rows(edit)
            st.info(f"**Bulle détectée :** {bubble_name or '(aucune)'}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Valider ce tournoi", key=f"commit_{tid}"):
                    append_results_log(edit.copy())
                    journal = load_journal()
                    journal.loc[len(journal)] = {"sha1":tid, "filename": Path(item["src_path"]).name, "processed_at": datetime.now()}
                    save_journal(journal)
                    archive_pdf(Path(item["src_path"]))
                    del st.session_state.pending_tourneys[tid]
                    # Rebuild snapshot courant
                    cur_log = load_results_log_any()
                    table = standings_from_log(cur_log, season_only=False)
                    table.to_csv(DATA_DIR / "latest_master.csv", index=False, encoding="utf-8")
                    st.success("Tournoi ajouté. Classement mis à jour.")
                    to_rerun = True
            with c2:
                if st.button("🗑️ Annuler / retirer de la file", key=f"cancel_{tid}"):
                    safe_unlink(Path(item["src_path"]))
                    del st.session_state.pending_tourneys[tid]
                    st.warning("Tournoi retiré de la file.")
                    to_rerun = True
            st.divider()
        if to_rerun:
            if hasattr(st, "rerun"): st.rerun()
            else: st.experimental_rerun()

        # Rollback
        st.subheader("Annuler le dernier import")
        if st.button("🧹 Annuler le dernier tournoi importé"):
            info = rollback_last_import()
            if info.get("ok"):
                st.success(f"{info['msg']} — PDF remis: {info.get('pdf_back','')}")
                if hasattr(st, "rerun"): st.rerun()
                else: st.experimental_rerun()
            else:
                st.info(info.get("msg","Rien à annuler."))


# --- Détails joueur ----------------------------------------------------------

elif page == "👤 Détails joueur":
    
    def render_player_details(log: pd.DataFrame) -> None:
        import numpy as np
        import pandas as pd
        import altair as alt
        from datetime import date
        from app_classement_unique import current_season_bounds, euro, parse_money

        st.title("Fiche joueur")

        if log is None or log.empty:
            st.info("Aucun historique pour l’instant.")
            return

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
            return
        who = st.selectbox("Choisir un joueur", pseudos)

        # sous-ensemble période + joueur
        df = log.copy()
        if "start_time" in df.columns and not df["start_time"].isna().all():
            df["start_date"] = pd.to_datetime(df["start_time"]).dt.date
            df = df[(df["start_date"] >= d1) & (df["start_date"] <= d2)]
        df = df[df["Pseudo"] == who].copy()
        if df.empty:
            st.info("Pas de données pour ce joueur sur la période.")
            return

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

        # Bulles: par tournoi = 1er joueur SANS gain cash (>0)
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

        # Recaves €
        recaves_euro = reentry * buyin_t
        # Frais = buy-in + recaves €
        frais = buyin_t + recaves_euro
        # Gains totaux = cash + bounty
        gains_tot = gaincash + bounty
        # Bénéfices
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
        roi_total = (net / total_fees * 100.0) if total_fees > 0 else None
        avg_finish = float(pos.replace(0, np.nan).mean()) if (pos > 0).any() else 0.0

        # Points (formule N_participants - pos + 1, min=1)
        # calcule avec le log global sur la période pour avoir N_participants correct
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

        # ---- TUILES
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
        c15.metric("ROI (total)", f"{roi_total:.0f}%" if roi_total is not None else "—")


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

    log = load_results_log()         # (déjà importée en haut du fichier)
    render_player_details(log)

# ==========
# PAGE 4 — 📚 Archives
# ==========
elif page == "📚 Archives":
    st.title("Archives")

    # --- Classement à la date (basé sur la DATE DE TOURNOI, pas la date d'import)
    st.subheader("Classement « à la date »")
    log = load_results_log_any()
    if log.empty:
        st.info("Pas d’historique pour le moment.")
    else:
        d_cut = st.date_input("Afficher l’état au", value=date.today())

        # Saison correspondant à cette date (01/08 -> 31/07)
        s0, s1 = current_season_bounds(d_cut)

        # Filtre : (a) par saison de la date choisie, (b) jusqu'à d_cut inclus
        df = log.copy()
        df["__d"] = pd.to_datetime(df["start_time"], errors="coerce").dt.date
        sub = df[(df["__d"] >= s0) & (df["__d"] <= d_cut)].copy()

        table = standings_from_log(sub, season_only=False)
        if table.empty:
            st.info("Aucune donnée jusqu’à cette date dans cette saison.")
        else:
            show_table(table, caption=f"État arrêté au {d_cut:%d/%m/%Y} (dates de tournoi)")

    # --- PDFs archivés groupés par saison (à partir de la date dans le nom du fichier)
    st.subheader("PDFs archivés (par saison)")

    # petites helpers locales (pas d'effet ailleurs)
    import re

    def _parse_dt_from_filename(name: str) -> tuple[date | None, str]:
        """
        Essaie d'extraire (date, time_str) depuis "… du 07-09-2025 21:15 …" ou "… du 07-09-2025 2115 …"
        Retourne (date | None, "HH:MM" ou "").
        """
        base = Path(name).stem
        m = re.search(r"du\s+(\d{2}-\d{2}-\d{4})\s+(\d{2}[:h]?\d{2})", base, flags=re.I)
        if not m:
            return None, ""
        dstr, tstr = m.group(1), m.group(2).lower().replace("h", ":")
        try:
            dparts = dstr.split("-")
            d_obj = date(int(dparts[2]), int(dparts[1]), int(dparts[0]))
        except Exception:
            return None, ""
        if re.fullmatch(r"\d{4}", tstr):
            tstr = f"{tstr[:2]}:{tstr[2:]}"
        return d_obj, tstr

    def _season_label(d_obj: date) -> str:
        """Saison 01/08 -> 31/07."""
        if d_obj.month >= 8:
            return f"{d_obj.year}-{d_obj.year+1}"
        else:
            return f"{d_obj.year-1}-{d_obj.year}"

    pdfs = list_files_sorted(PDF_DONE, ("*.pdf",))
    if not pdfs:
        st.caption("Aucun PDF archivé.")
    else:
        # Regroupe par saison -> liste [(dt, time_str, Path)]
        seasons: dict[str, list[tuple[date, str, Path]]] = {}
        for p in pdfs:
            d_obj, tstr = _parse_dt_from_filename(p.name)
            if not d_obj:
                # si on ne trouve pas de date dans le nom, on classe dans "Inconnue"
                seasons.setdefault("Inconnue", []).append((date.min, "", p))
                continue
            lab = _season_label(d_obj)
            seasons.setdefault(lab, []).append((d_obj, tstr, p))

        # Affiche saisons par ordre décroissant (2025-2026, puis 2024-2025, …, puis "Inconnue")
        def _season_sort_key(label: str):
            if label == "Inconnue":
                return (0, label)
            try:
                y1, y2 = label.split("-")
                return (int(y1), label)
            except Exception:
                return (0, label)

        for season in sorted(seasons.keys(), key=_season_sort_key, reverse=True):
            with st.expander(f"Saison {season}", expanded=True):
                # tri décroissant par date de tournoi, puis heure si dispo
                entries = seasons[season]
                entries.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)

                for d_obj, tstr, p in entries:
                    cols = st.columns([6, 2, 2])
                    # libellé (avec date si connue)
                    when = f"{d_obj:%Y-%m-%d}" if d_obj != date.min else "date inconnue"
                    if tstr:
                        when += f" {tstr}"
                    cols[0].write(f"**{p.name}**  \n_{when}_")

                    # PDF
                    with cols[1]:
                        st.download_button(
                            "PDF",
                            data=p.read_bytes(),
                            file_name=p.name,
                            type="secondary",
                            key=f"dlpdf_{p.name}"
                        )

                    # JPG (génère/actualise si plus ancien que le PDF)
                    with cols[2]:
                        try:
                            jpg_path = SNAP_DIR / "archived_jpg" / (p.stem + ".jpg")
                            need_regen = (not jpg_path.exists()) or (jpg_path.stat().st_mtime < p.stat().st_mtime)
                            if need_regen:
                                pdf_first_page_to_jpg(p, jpg_path, dpi=220)
                            st.download_button(
                                "JPG",
                                data=jpg_path.read_bytes(),
                                file_name=jpg_path.name,
                                type="secondary",
                                key=f"dljpg_{p.name}"
                            )
                        except Exception as e:
                            st.button("JPG indisponible", disabled=True, key=f"nojpg_{p.name}")
                            st.caption(f"⚠️ Conversion JPG échouée : {e}")

# ==========
# PAGE — 🏅 Classement par points
# ==========
elif page == "🏅 Classement par points":
    from pathlib import Path
    from datetime import datetime
    from app_classement_unique import (
        load_results_log_any,
        compute_points_table,
        current_season_bounds,
        classement_points_df_to_jpg,  # <- la fonction JPG que tu as ajoutée
        euro,
    )

    st.title("Classement général — Points")

    log = load_results_log_any()
    if log is None or log.empty:
        st.info("Aucune donnée pour l’instant. Va dans **Importer** pour traiter des PDFs.")
    else:
        # Sélection de période (par défaut : saison courante)
        s0, s1 = current_season_bounds()
        with st.expander("Filtrer par période (saison active par défaut)", expanded=False):
            d1 = st.date_input("Du", value=s0)
            d2 = st.date_input("Au", value=s1)

        # Table points
        pts = compute_points_table(log, d1, d2)
        if pts.empty:
            st.info("Pas de résultats sur cette période.")
        else:
            st.dataframe(pts, use_container_width=True, hide_index=True)

            # Export CSV
            st.download_button(
                "⬇️ Exporter (CSV Points)",
                data=pts.to_csv(index=False).encode("utf-8"),
                file_name="classement_points.csv",
                type="secondary",
            )

            # Export JPG (largeur auto pour éviter le rognage)
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
                    st.download_button(
                        "⬇️ Télécharger le JPG (Points)",
                        data=Path(jp).read_bytes(),
                        file_name=Path(jp).name,
                        type="secondary",
                    )

# ==========
# PAGE 6 — Réinitialiser (LOCAL uniquement)
# ==========
elif page == "♻️ Réinitialiser" and not IS_PUBLIC:
    st.title("Réinitialiser la saison courante")
    st.warning("Attention : remise à zéro des agrégats. Les PDFs archivés peuvent être re-traités ensuite.")

    do_move = st.checkbox(
        "Remettre les PDFs archivés dans « PDF_A_TRAITER » (suppression horodatage)",
        value=True
    )

    if st.button("⚠️ Lancer la réinitialisation", type="primary"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Backups dans data/BACKUPS
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

        # --- Vider les CSV (en conservant les entêtes attendues)
        pd.DataFrame(columns=RESULTS_LOG_COLUMNS).to_csv(RESULTS_LOG, index=False, encoding="utf-8")
        pd.DataFrame(columns=JOURNAL_COLUMNS).to_csv(JOURNAL_CSV, index=False, encoding="utf-8")

        # --- Option : remettre les PDFs traités dans le dossier d’entrée
        if do_move:
            PDF_DIR.mkdir(parents=True, exist_ok=True)

            def _unique_dest(base_dir: Path, name: str) -> Path:
                """Évite d’écraser un fichier existant en ajoutant un suffixe _dupN."""
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
                # retire l’horodatage ajouté lors de l’archivage (avant "__")
                clean_name = (p.name.split("__")[0] + ".pdf") if "__" in p.name else p.name
                dst = _unique_dest(PDF_DIR, clean_name)
                try:
                    shutil.move(str(p), str(dst))
                except Exception as e:
                    st.warning(f"Impossible de déplacer {p.name} -> {dst.name} : {e}")

        st.success("Réinitialisation OK. Allez dans **Importer** pour revalider.")
