# app_classement_unique.py
# ------------------------------------------------------------
# Moteur CoronaMax : parsing PDF Winamax + journaux + classements
# ------------------------------------------------------------
from __future__ import annotations

import re
import csv
import shutil
import hashlib
import math
import json
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

import pandas as pd

# =========================
# Dossiers & fichiers
# =========================
BASE       = Path(__file__).parent
PDF_DIR    = BASE / "PDF_A_TRAITER"
PDF_DONE   = BASE / "PDF_Traites"
ARCHIVE    = BASE / "Archive"
SNAP_DIR   = ARCHIVE / "snapshots"

# Master (xlsx) et journaux (csv)
F_MASTER     = BASE / "GAINS_Wina.xlsx"
RESULTS_LOG  = ARCHIVE / "results_log.csv"   # lignes par joueur et par tournoi
JOURNAL_CSV  = ARCHIVE / "journal.csv"       # fichiers importés (anti-doublon, traçabilité)

# Utilisé par d’anciennes fonctions (laisser défini)
POPLER_PATH = None  # si besoin pour exports PDF→JPG ailleurs


# =========================
# Utilitaires généraux
# =========================
def ensure_dirs():
    for d in (PDF_DIR, PDF_DONE, ARCHIVE, SNAP_DIR):
        d.mkdir(parents=True, exist_ok=True)

def build_public_snapshot(dst: Path):
    """
    Construit un snapshot lecture-seule dans `dst` :
    - results_log.csv, master (XLSX/CSV si dispo)
    - dernier JPG du classement
    - index des PDFs archivés (archives.json)
    """
    from app_classement_unique import (
        RESULTS_LOG, F_MASTER, F_JPG_DIR, PDF_DONE, JOURNAL_CSV
    )

    dst = Path(dst)
    (dst / "data").mkdir(parents=True, exist_ok=True)

    # CSV/Excel
    if RESULTS_LOG.exists() and RESULTS_LOG.stat().st_size > 0:
        shutil.copy2(RESULTS_LOG, dst / "data" / RESULTS_LOG.name)
    if F_MASTER.exists():
        shutil.copy2(F_MASTER, dst / "data" / F_MASTER.name)
        # Bonus: exporter aussi en CSV lisible côté web
        try:
            pd.read_excel(F_MASTER).to_csv(dst / "data" / "master.csv", index=False)
        except Exception:
            pass
    if JOURNAL_CSV.exists():
        shutil.copy2(JOURNAL_CSV, dst / "data" / JOURNAL_CSV.name)

    # Dernier JPG
    try:
        jpgs = sorted(F_JPG_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if jpgs:
            shutil.copy2(jpgs[0], dst / "data" / jpgs[0].name)
    except Exception:
        pass

    # Index des PDFs (on ne copie pas les PDFs ici, on liste seulement)
    pdf_index = []
    for p in sorted(PDF_DONE.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        pdf_index.append({
            "name": p.name,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        })
    (dst / "data" / "archives.json").write_text(
        json.dumps({"pdfs": pdf_index}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Accepte : 1 234,56  |  1 234,56  |  1234.56  |  -12,3  |  12,3 €  |  "35.1€"
_NUM_RE = re.compile(r"[-+]?[\d]+(?:[ \u00A0]?\d{3})*(?:[.,]\d+)?")
_num_re = _NUM_RE  # alias pour compatibilité avec d’anciens appels

def _parse_money(x, default: float = 0.0) -> float:
    """Convertit toute écriture monétaire FR/EN en float (35,10 € -> 35.10)."""
    if x is None:
        return default
    # gère NaN éventuel (float)
    try:
        if isinstance(x, float) and math.isnan(x):
            return default
    except Exception:
        pass
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    # normalisations de base
    s = (s
         .replace("\u00A0", " ")   # espace insécable
         .replace("€", "")
         .replace("−", "-"))       # signe moins unicode -> ASCII

    # essaie d’isoler la partie numérique
    m = _NUM_RE.search(s)
    if not m:
        # fallback brut (enlève espaces, remplace virgule par point)
        s2 = s.replace(" ", "").replace(",", ".")
        try:
            return float(s2)
        except Exception:
            return default

    num = m.group(0)
    num = num.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    try:
        return float(num)
    except Exception:
        return default

def parse_money(x, default: float = 0.0) -> float:
    """Alias public (utilisé par l’UI)."""
    return _parse_money(x, default)

def euro(x) -> str:
    """Formate une valeur en 'xx,yy €' (FR)."""
    return f"{_parse_money(x):.2f} €".replace(".", ",")

def current_season_bounds(today: date | None = None) -> tuple[date, date]:
    """
    Saison = du 01/08/N au 31/07/N+1.
    Retourne (date_debut, date_fin) pour la saison couvrant 'today'.
    """
    if today is None:
        today = date.today()
    year = today.year
    s0 = date(year, 8, 1)
    if today >= s0:
        # saison courant: [01/08/year .. 31/07/year+1]
        return (date(year, 8, 1), date(year + 1, 7, 31))
    else:
        # saison précédente
        return (date(year - 1, 8, 1), date(year, 7, 31))

def classement_df_to_jpg(df: pd.DataFrame, out_path: Path, title: str | None = None) -> Path:
    """
    Render a classement DataFrame to a JPG image.
    - No seaborn; plain matplotlib only.
    - Styles header (grey/bold), Pseudo column (orange/bold),
      and applies a soft green→red background on 'Bénéfices' if present.
    """
    # Lazy imports to avoid importing matplotlib when unused
    import matplotlib.pyplot as plt

    # Work on a copy; ensure strings for rendering
    data = df.copy()
    data = data.astype(str)

    n_rows, n_cols = data.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Tableau vide : rien à exporter.")

    # Figure size (dynamic)
    col_w   = 1.1   # inches per column
    row_h   = 0.42  # inches per row
    width   = max(8.0, min(20.0, col_w * n_cols + 1.0))
    height  = max(2.5, row_h * (n_rows + 1) + (0.6 if title else 0.2))

    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    ax.axis("off")

    # Build the table
    table = ax.table(
        cellText=data.values,
        colLabels=list(data.columns),
        loc="center",
        cellLoc="center"
    )

    # Fonts / sizes
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    # Slightly taller rows for readability
    try:
        table.scale(1.0, 1.25)
    except Exception:
        pass

    # Helper: column index by name (or None)
    def col_idx(name: str) -> int | None:
        try:
            return list(data.columns).index(name)
        except ValueError:
            return None

    pseudo_col = col_idx("Pseudo")
    benef_col  = col_idx("Bénéfices")

    # Header style (row 0 in mpl-table is header)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="#000000")
            cell.set_facecolor("#e6e6e6")

    # Pseudo column: orange background + bold
    if pseudo_col is not None:
        for r in range(1, n_rows + 1):  # body rows start at 1
            cell = table[(r, pseudo_col)]
            cell.set_facecolor("#f7b329")
            cell.set_text_props(weight="bold", color="#000000")

    # Green→red gradient on Bénéfices if present (and numeric)
    if benef_col is not None:
        # numeric values (try to reuse your existing parse_money if present)
        try:
            from app_classement_unique import parse_money as _parse_money  # type: ignore
        except Exception:
            def _parse_money(x):
                s = str(x).strip().replace("€","").replace("\u00a0"," ").replace(" ","").replace(",",".")
                try:
                    return float(s)
                except Exception:
                    return 0.0

        vals = [ _parse_money(v) for v in df.iloc[:, benef_col] ]
        vmin = min(vals) if vals else 0.0
        vmax = max(vals) if vals else 1.0
        span = (vmax - vmin) if vmax != vmin else 1.0

        # Simple lerp between red and green
        def lerp(a, b, t): return a + (b - a) * t
        def color_for(v):
            t = (v - vmin) / span
            r = int(lerp(220,  80, t))
            g = int(lerp( 80, 200, t))
            b = int(lerp( 80,  90, t))
            return (r/255.0, g/255.0, b/255.0)

        for r in range(1, n_rows + 1):
            cell = table[(r, benef_col)]
            cell.set_facecolor(color_for(vals[r-1]))
            cell.set_text_props(weight="bold")

    # Draw a subtle outer border
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#000000")

    # Optional title
    if title:
        ax.set_title(title, fontsize=12, weight="bold", pad=8)

    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
    
    
# === SAISONS ===============================================================

def season_label(year_start: int) -> str:
    """Ex: 2024 -> '2024–25'"""
    return f"{year_start}\u2013{str((year_start+1) % 100).zfill(2)}"

def season_bounds_from_year(year_start: int) -> tuple[date, date]:
    """Bornes [incl, incl] d'une saison : 01/08/Y .. 31/07/Y+1."""
    from datetime import date
    d1 = date(year_start, 8, 1)
    d2 = date(year_start + 1, 7, 31)
    return d1, d2

def seasons_available(log: pd.DataFrame) -> list[tuple[str, date, date]]:
    """Liste des saisons présentes dans le log + la saison courante en dernier."""
    from datetime import date
    out: list[tuple[str, date, date]] = []
    if not log.empty and "start_time" in log.columns:
        years = []
        for dt in log["start_time"].dt.date:
            y = dt.year if dt.month >= 8 else dt.year - 1
            years.append(y)
        years = sorted(set(years))
        for y in years:
            d1, d2 = season_bounds_from_year(y)
            out.append((season_label(y), d1, d2))

    # ajoute toujours la saison courante si absente
    today = date.today()
    cur_y = today.year if today.month >= 8 else today.year - 1
    if not out or out[-1][0] != season_label(cur_y):
        out = [x for x in out if x[0] != season_label(cur_y)]
        out.append((season_label(cur_y), *season_bounds_from_year(cur_y)))
    return out


# === ROLLBACK du dernier import ===========================================

def rollback_last_import() -> dict:
    """
    Supprime le DERNIER tournoi importé du results_log + journal,
    tente de remettre son PDF archivé dans PDF_DIR, puis reconstruit le master.
    Retourne un petit récap.
    """
    log = load_results_log()
    if log.empty:
        return {"ok": False, "msg": "Aucun tournoi à annuler."}

    # dernier tournoi = max(processed_at) par tournament_id
    grp = log.groupby("tournament_id")["processed_at"].max()
    last_tid = grp.idxmax()
    last_ts = grp.max()

    # Supprimer les lignes du log
    log2 = log[log["tournament_id"] != last_tid].copy()
    log2.to_csv(RESULTS_LOG, index=False)

    # Journal & PDF archivé -> retour dans PDF_DIR si possible
    j = load_journal()
    moved_pdf = None
    if not j.empty and "sha1" in j.columns:
        row = j[j["sha1"] == last_tid].tail(1)
        if not row.empty:
            fn = row["filename"].iloc[0]
            # chercher un PDF correspondant dans PDF_DONE
            # (nom horodaté type 'xxx__YYYYMMDD_HHMMSS.pdf' ou juste fn)
            stem = Path(fn).stem.split("__")[0]
            cand = None
            for p in PDF_DONE.glob("*.pdf"):
                if last_tid in p.name or Path(p).stem.split("__")[0] == stem:
                    cand = p
                    break
            if cand and cand.exists():
                dest = PDF_DIR / (stem + ".pdf")
                dest.write_bytes(cand.read_bytes())
                cand.unlink(missing_ok=True)
                moved_pdf = dest.name

        # enlever du journal
        j2 = j[j["sha1"] != last_tid].copy()
        save_journal(j2)

    # reconstruire le master
    rebuild_master_from_log()
    return {
        "ok": True,
        "msg": f"Tid {last_tid} annulé.",
        "pdf_back": moved_pdf,
        "processed_at": str(last_ts),
    }
    
# =========================
# PDF parsing (Winamax)
# =========================
@dataclass
class ParsedTournament:
    tournament_id: str
    tournament_name: str
    start_time: datetime
    buyin: float           # hors rake
    rake: float
    buyin_total: float     # buyin + rake
    is_ko: bool
    rows: pd.DataFrame     # colonnes: Position, Pseudo, GainCash, Bounty, Recaves


# --- remplace _read_pdf_text par ceci ---
def _read_pdf_text(pdf_path: Path) -> str:
    """Retourne le texte du PDF (pdfplumber, fallback PyPDF2) + normalisation légère."""
    txt = ""
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p in pdf.pages:
                txt += (p.extract_text() or "") + "\n"
    except Exception:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                txt += (page.extract_text() or "") + "\n"
        except Exception as e:
            raise RuntimeError(f"Impossible de lire le PDF: {e}")

    # normalisation douce
    txt = txt.replace("\u00a0", " ")          # espace insécable → espace
    txt = txt.replace("–", "-").replace("—", "-")  # tirets “longs” → '-'
    txt = txt.replace(" ", " ")               # espace fine insécable
    # compacte les espaces multiples
    txt = re.sub(r"[ \t]+", " ", txt)
    # on garde un découpage par lignes “propres”
    txt = "\n".join(line.strip() for line in txt.splitlines() if line.strip())
    return txt


def _parse_money(s: str | float | int, default: float = 0.0) -> float:
    """Extrait un nombre FR depuis une chaîne libre."""
    if isinstance(s, (int, float)):
        return float(s)
    if not s:
        return default
    s = str(s)
    m = _num_re.search(s)
    if not m:
        return default
    x = m.group(0).replace(" ", "").replace(",", ".")
    try:
        return float(x)
    except Exception:
        return default



def _find_header_name_datetime(text: str, filename_fallback: str | None = None) -> tuple[str, datetime]:
    """
    Cherche une ligne de type:
      'Tournoi de poker <NOM> du dd-mm-YYYY HH:MM en argent réel'
    Variantes gérées :
      - préfixe 'winamax.fr-'
      - séparateurs de date '-', '/' ou ' ' (ex '31-08-2025' ou '31/08/2025')
      - heure '21:15' ou '2115'
      - 'en argent réel' optionnel
    Fallback: tente de parser le nom de fichier si le texte échoue.
    """
    # liste de patrons “souples”
    patterns = [
        r"(?:winamax\.fr-)?\s*Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+du\s+(?P<d>\d{2}[-/\s]\d{2}[-/\s]\d{4})\s+(?P<h>\d{1,2}[:h]?\d{2})(?:\s+en\s+argent\s+r[eé]el)?",
        r"(?:winamax\.fr-)?\s*Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+-\s*(?P<after>.+?)\s+du\s+(?P<d>\d{2}[-/\s]\d{2}[-/\s]\d{4})\s+(?P<h>\d{1,2}[:h]?\d{2})(?:\s+en\s+argent\s+r[eé]el)?",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            name = m.group("name").strip()
            d = m.group("d").replace("/", "-").replace(" ", "-")
            h = m.group("h").replace("h", ":")
            if ":" not in h and len(h) in (3, 4):
                # ex: "915" → "09:15", "2115" → "21:15"
                h = h.zfill(4)
                h = f"{h[:2]}:{h[2:]}"
            when = datetime.strptime(f"{d} {h}", "%d-%m-%Y %H:%M")
            return name, when

    # ---- Fallback: tenter via le nom de fichier ----
    if filename_fallback:
        fn = filename_fallback
        # Exemple de fichier :
        # 'winamax.fr-Tournoi de poker CoronaMax VII - Filetonfric 01 du 31-08-2025 2115 en argent réel.pdf'
        fn = fn.replace("_", " ").replace("%20", " ")
        m = re.search(
            r"(?:winamax\.fr-)?\s*Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+du\s+(?P<d>\d{2}-\d{2}-\d{4})\s+(?P<h>\d{1,2}[:h]?\d{2})",
            fn, flags=re.IGNORECASE
        )
        if m:
            name = m.group("name").strip()
            d = m.group("d")
            h = m.group("h").replace("h", ":")
            if ":" not in h and len(h) in (3, 4):
                h = h.zfill(4); h = f"{h[:2]}:{h[2:]}"
            when = datetime.strptime(f"{d} {h}", "%d-%m-%Y %H:%M")
            return name, when

    raise RuntimeError("En-tête tournoi introuvable (nom/date/heure).")

def _find_buyin_rake(text: str) -> tuple[float, float]:
    """Détecte Buy-in & Rake, fallback (9,1)."""
    buy = 9.0
    rake = 1.0

    pats_buy = [
        r"Buy-?in\s*[:\-]?\s*([\d\s]+[.,]?\d*)\s*€",
        r"Frais d'inscription.*?Buy-?in\s*[:\-]?\s*([\d\s]+[.,]?\d*)\s*€",
    ]
    pats_rake = [
        r"Rake\s*[:\-]?\s*([\d\s]+[.,]?\d*)\s*€",
        r"Frais d'inscription.*?Rake\s*[:\-]?\s*([\d\s]+[.,]?\d*)\s*€",
        r"Commission\s*[:\-]?\s*([\d\s]+[.,]?\d*)\s*€",
    ]

    for p in pats_buy:
        m = re.search(p, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            buy = _parse_money(m.group(1), default=buy)
            break

    for p in pats_rake:
        m = re.search(p, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            rake = _parse_money(m.group(1), default=rake)
            break

    return buy, rake


def _looks_ko(name: str, text: str) -> bool:
    """KO si 'KO' dans le nom ou si 'Bounty' mentionné."""
    if re.search(r"\bKO\b", name, re.IGNORECASE):
        return True
    if re.search(r"\bBounty\b", text, re.IGNORECASE):
        return True
    return False


# --- remplace extract_from_pdf par ceci (juste le bloc début qui appelle _find_header_name_datetime) ---
def extract_from_pdf(pdf_path: Path) -> ParsedTournament:
    """
    Parse un PDF Winamax (KO ou non) → ParsedTournament.
    rows: Position, Pseudo, GainCash, Bounty, Recaves
    """
    text = _read_pdf_text(pdf_path)

    # nom & datetime (avec fallback sur nom de fichier)
    try:
        tname, when = _find_header_name_datetime(text, filename_fallback=pdf_path.name)
    except Exception as e:
        # aide au debug : montre un extrait de la 1ʳᵉ ligne “longue” contenant 'Tournoi de poker'
        lines = [ln for ln in text.splitlines() if "Tournoi de poker" in ln]
        hint = f"\nAperçu ligne: {lines[0][:180]}..." if lines else ""
        raise RuntimeError(f"En-tête introuvable dans le PDF ou le nom de fichier.{hint}") from e

    # buy-in / rake
    buy, rake = _find_buyin_rake(text)
    buy_total = round(buy + rake, 2)

    # KO ?
    is_ko = _looks_ko(tname, text)

    # on part de 'Résultats'
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start_idx = None
    for i, ln in enumerate(lines):
        if "Résultats" in ln:
            start_idx = i + 1
            break
    if start_idx is None:
        start_idx = 0

    pos_head     = re.compile(r"^\s*(\d+)\s+(.*)$")
    reentry_tail = re.compile(r"^(.*\S)\s+(\d+)\s*$")
    money_tail   = re.compile(r"^(.*\S)\s+([-]?\d[\d\s.,]*)(?:\s*€)?(?:\s+([-]?\d[\d\s.,]*)(?:\s*€)?)?\s*$")

    rows = []
    for ln in lines[start_idx:]:
        if "Règlement" in ln or "Historique" in ln:
            break
        m = pos_head.match(ln)
        if not m:
            continue

        pos  = int(m.group(1))
        rest = m.group(2).strip()

        recaves = 0
        m2 = reentry_tail.match(rest)
        if m2:
            rest = m2.group(1).strip()
            recaves = int(m2.group(2))

        pseudo = rest
        gain   = 0.0
        bounty = 0.0

        m3 = money_tail.match(rest)
        if m3:
            left = m3.group(1).strip()
            n1   = _parse_money(m3.group(2), 0.0)
            n2   = _parse_money(m3.group(3), 0.0) if m3.group(3) else None

            if is_ko and n2 is not None:
                gain, bounty = n1, n2
                pseudo = left
            else:
                gain, bounty = n1, 0.0
                pseudo = left

        pseudo = re.sub(r"\s*€\s*$", "", pseudo).strip()
        if pseudo:
            rows.append({
                "Position": pos,
                "Pseudo": pseudo,
                "GainCash": float(gain),
                "Bounty": float(bounty),
                "Recaves": int(recaves)
            })

    df = pd.DataFrame(rows, columns=["Position", "Pseudo", "GainCash", "Bounty", "Recaves"])
    if df.empty:
        raise RuntimeError(f"{pdf_path.name} : 0 ligne détectée (format inattendu).")

    tid = sha1_of_file(pdf_path)
    return ParsedTournament(
        tournament_id=tid,
        tournament_name=tname,
        start_time=when,
        buyin=buy,
        rake=rake,
        buyin_total=buy_total,
        is_ko=is_ko,
        rows=df
    )

def build_rows_for_log(parsed: ParsedTournament) -> pd.DataFrame:
    """
    Prépare les lignes à insérer dans results_log (éditables dans l’UI).
    Colonnes: Pseudo, Position, GainCash, Bounty, Recaves, buyin_total, start_time, tournament_id, tournament_name
    """
    df = parsed.rows.copy()
    if df.empty:
        return df
    df["buyin_total"]     = float(parsed.buyin_total)
    df["start_time"]      = parsed.start_time
    df["tournament_id"]   = parsed.tournament_id
    df["tournament_name"] = parsed.tournament_name
    # types propres
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce").fillna(0).astype(int)
    df["GainCash"] = pd.to_numeric(df["GainCash"], errors="coerce").fillna(0.0)
    df["Bounty"]   = pd.to_numeric(df["Bounty"], errors="coerce").fillna(0.0)
    df["Recaves"]  = pd.to_numeric(df["Recaves"], errors="coerce").fillna(0).astype(int)
    return df[["Pseudo","Position","GainCash","Bounty","Recaves","buyin_total","start_time","tournament_id","tournament_name"]]


# =========================
# Journaux
# =========================
def load_results_log() -> pd.DataFrame:
    if RESULTS_LOG.exists() and RESULTS_LOG.stat().st_size > 0:
        df = pd.read_csv(RESULTS_LOG)
        # normaliser colonnes temps
        if "processed_at" in df.columns:
            df["processed_at"] = pd.to_datetime(df["processed_at"], errors="coerce")
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        # assure les deux alias
        if "processed_at" not in df.columns and "start_time" in df.columns:
            df["processed_at"] = df["start_time"]
        if "start_time" not in df.columns and "processed_at" in df.columns:
            df["start_time"] = df["processed_at"]
        return df
    # sinon, schéma vide propre
    return pd.DataFrame(columns=[
        "tournament_id","tournament_name","start_time","processed_at","filename",
        "Pseudo","Position","GainCash","Bounty","Reentry","buyin_total"
    ])

def append_results_log(df_rows: pd.DataFrame):
    """Ajoute des lignes au results_log.csv (crée si besoin)."""
    ensure_dirs()
    df = load_results_log()
    # normaliser colonnes
    needed = {"Pseudo","Position","GainCash","Bounty","Recaves","buyin_total","start_time","tournament_id","tournament_name"}
    missing = needed - set(df_rows.columns)
    if missing:
        # ajoute colonnes manquantes à 0/""/now
        for c in missing:
            if c in ("Pseudo","tournament_id","tournament_name"):
                df_rows[c] = ""
            elif c == "start_time":
                df_rows[c] = datetime.now()
            else:
                df_rows[c] = 0
    # types
    df_rows = df_rows.copy()
    df_rows["start_time"] = pd.to_datetime(df_rows["start_time"])
    # concat et sauvegarde
    out = pd.concat([df, df_rows], ignore_index=True)
    out.to_csv(RESULTS_LOG, index=False)


def load_journal() -> pd.DataFrame:
    ensure_dirs()
    if JOURNAL_CSV.exists() and JOURNAL_CSV.stat().st_size > 0:
        return pd.read_csv(JOURNAL_CSV, parse_dates=["processed_at"])
    return pd.DataFrame(columns=["sha1","filename","processed_at"])

def save_journal(df: pd.DataFrame):
    ensure_dirs()
    df = df.copy()
    if "processed_at" in df.columns:
        df["processed_at"] = pd.to_datetime(df["processed_at"])
    df.to_csv(JOURNAL_CSV, index=False)


def archive_pdf(src: Path) -> Path:
    """Déplace le PDF traité dans PDF_Traites en l’horodatant."""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = PDF_DONE / f"{src.stem}__{ts}.pdf"
    shutil.move(str(src), str(dst))
    return dst


# =========================
# Classement (depuis results_log)
# =========================
MASTER_COLUMNS = [
    "Place","Pseudo","Parties","Victoires","ITM","% ITM",
    "Recaves","Recaves en €","Bulles","Buy in","Frais","Gains","Bénéfices"
]

def standings_from_log(log_df: pd.DataFrame, season_only: bool = True) -> pd.DataFrame:
    """
    Agrège un log détaillé en classement :
      - Parties = nb lignes par Pseudo
      - Victoires = nb Position==1
      - ITM = nb GainCash>0 (Bounty ignoré pour ITM)
      - % ITM = ITM/Parties
      - Recaves = somme Recaves
      - Recaves en € = somme (Recaves * buyin_total)
      - Buy in = somme (buyin_total) (1 buy-in par tournoi joué)
      - Frais = Buy in + Recaves en €
      - Gains = somme (GainCash + Bounty)
      - Bénéfices = Gains - Frais
      - Winamax (ligne 1) : Parties = nb tournois, Gains = 10% des Frais des autres, Frais = 0
    """
    ensure_dirs()
    if log_df is None or log_df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    g = log_df.copy()

    # nettoyage types
    g["Position"]    = pd.to_numeric(g.get("Position", 0), errors="coerce").fillna(0).astype(int)
    g["GainCash"]    = pd.to_numeric(g.get("GainCash", 0.0), errors="coerce").fillna(0.0)
    g["Bounty"]      = pd.to_numeric(g.get("Bounty", 0.0), errors="coerce").fillna(0.0)
    g["Recaves"]     = pd.to_numeric(g.get("Recaves", 0), errors="coerce").fillna(0).astype(int)
    g["buyin_total"] = pd.to_numeric(g.get("buyin_total", 0.0), errors="coerce").fillna(0.0)
    g["start_time"]  = pd.to_datetime(g.get("start_time", datetime.now()))

    # borne saison si demandé (normalement déjà filtré côté UI)
    if season_only:
        s0, s1 = current_season_bounds()
        g = g[(g["start_time"].dt.date >= s0) & (g["start_time"].dt.date <= s1)]

    if g.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    # agrégats
    # Recaves en € par ligne
    g["recaves_euro_row"] = g["Recaves"] * g["buyin_total"]
    g["gains_tot_row"]    = g["GainCash"] + g["Bounty"]
    g["itm_row"]          = (g["GainCash"] > 0).astype(int)
    g["victoire_row"]     = (g["Position"] == 1).astype(int)

    agg = g.groupby("Pseudo").agg(
        Parties       = ("Pseudo", "count"),
        Victoires     = ("victoire_row", "sum"),
        ITM           = ("itm_row", "sum"),
        Recaves       = ("Recaves", "sum"),
        _recaves_euro = ("recaves_euro_row", "sum"),
        Buy_in        = ("buyin_total", "sum"),
        Gains         = ("gains_tot_row", "sum"),
    ).reset_index()

    agg = agg.rename(columns={"_recaves_euro": "Recaves en €"})
    agg["% ITM"]     = agg.apply(lambda r: f"{(r['ITM']/r['Parties']):.0%}" if r['Parties'] > 0 else "0%", axis=1)
    agg["Bulles"]    = 0  # (peut être rempli plus tard si on journalise explicitement la bulle)
    agg["Frais"]     = agg["Buy_in"] + agg["Recaves en €"]
    agg["Bénéfices"] = agg["Gains"] - agg["Frais"]

    # Winamax
    nb_tournois = g["tournament_id"].nunique() if "tournament_id" in g.columns else int(g["start_time"].dt.date.nunique())
    total_frais_autres = float(agg["Frais"].sum())
    winamax = pd.DataFrame([{
        "Pseudo": "Winamax",
        "Parties": int(nb_tournois),
        "Victoires": 0, "ITM": 0, "% ITM": "0%",
        "Recaves": 0, "Recaves en €": 0.0,
        "Bulles": 0, "Buy_in": 0.0, "Frais": 0.0,
        "Gains": round(total_frais_autres * 0.10, 2),
        "Bénéfices": round(total_frais_autres * 0.10, 2),
    }])

    out = pd.concat([winamax, agg], ignore_index=True)

    # tri: Winamax en tête, puis Bénéfices desc
    out_others = out[out["Pseudo"].str.lower() != "winamax"].copy()
    out_others = out_others.sort_values("Bénéfices", ascending=False, kind="mergesort")
    out = pd.concat([out[out["Pseudo"].str.lower() == "winamax"], out_others], ignore_index=True)

    # place (Winamax = 0, puis 1..)
    places = []
    rank = 0
    for i, row in out.iterrows():
        if str(row["Pseudo"]).strip().lower() == "winamax":
            places.append(0)
        else:
            rank += 1
            places.append(rank)
    out["Place"] = places

    # réordonner colonnes
    out = out[["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €","Bulles","Buy_in","Frais","Gains","Bénéfices"]]
    # uniformiser nom 'Buy in'
    out = out.rename(columns={"Buy_in": "Buy in"})
    return out


def order_master_with_winamax_first(df: pd.DataFrame) -> pd.DataFrame:
    """Utilitaire: Winamax en première ligne, puis Bénéfices décroissants."""
    if df is None or df.empty or "Pseudo" not in df.columns:
        return df
    win = df[df["Pseudo"].astype(str).str.lower() == "winamax"]
    oth = df[df["Pseudo"].astype(str).str.lower() != "winamax"]
    if "Bénéfices" in df.columns:
        oth = oth.sort_values("Bénéfices", ascending=False, kind="mergesort")
    out = pd.concat([win, oth], ignore_index=True)
    # recalcul des places
    places = []
    rank = 0
    for _, r in out.iterrows():
        if str(r["Pseudo"]).strip().lower() == "winamax":
            places.append(0)
        else:
            rank += 1
            places.append(rank)
    out["Place"] = places
    return out


# =========================
# Master (xlsx)
# =========================
def _empty_master() -> pd.DataFrame:
    return pd.DataFrame(columns=MASTER_COLUMNS)

def load_master_df() -> pd.DataFrame:
    """Charge le master xlsx, sinon crée un DataFrame vide avec les bonnes colonnes."""
    ensure_dirs()
    if F_MASTER.exists():
        try:
            df = pd.read_excel(F_MASTER)
            # garantir colonnes
            for c in MASTER_COLUMNS:
                if c not in df.columns:
                    df[c] = []  # ajoute vide
            return df[MASTER_COLUMNS]
        except Exception:
            pass
    return _empty_master()

def save_master_df(df: pd.DataFrame):
    """Écrit le master au format xlsx (openpyxl)."""
    ensure_dirs()
    # normaliser ordre de colonnes
    d = df.copy()
    for c in MASTER_COLUMNS:
        if c not in d.columns:
            d[c] = []
    d = d[MASTER_COLUMNS]
    with pd.ExcelWriter(F_MASTER, engine="openpyxl") as xw:
        d.to_excel(xw, index=False)


def snapshot_master(df: pd.DataFrame):
    """Sauvegarde un snapshot daté du master (CSV) pour Archives."""
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = SNAP_DIR / f"classement_{ts}.csv"
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)


def rebuild_master_from_log():
    """
    Recalcule le master à partir de results_log.csv (saison courante) et le sauve.
    """
    log = load_results_log()
    if log.empty:
        save_master_df(_empty_master())
        return
    # filtre saison courante
    s0, s1 = current_season_bounds()
    log = log[(log["start_time"].dt.date >= s0) & (log["start_time"].dt.date <= s1)].copy()
    master = standings_from_log(log, season_only=False)
    master = order_master_with_winamax_first(master)
    save_master_df(master)
    snapshot_master(master)
