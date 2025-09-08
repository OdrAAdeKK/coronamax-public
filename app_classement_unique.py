# app_classement_unique.py
from __future__ import annotations
import os, re, io, shutil, hashlib, json, time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional
import unicodedata
import base64, requests

from pdfminer.high_level import extract_text as _pdfminer_extract_text
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


import numpy as np
import pandas as pd

# ==============
# Chemins
# ==============
BASE = Path(__file__).parent.resolve()
ARCHIVE = BASE / "ARCHIVE"
PDF_DIR = BASE / "PDF_A_TRAITER"
PDF_DONE = ARCHIVE / "PDF_TRAITES"
SNAP_DIR = BASE / "SNAPSHOTS"
DATA_DIR = BASE / "data"           # snapshot public

F_MASTER = BASE / "GAINS_Wina.xlsx"  # facultatif (fallback local)
RESULTS_LOG = ARCHIVE / "results_log.csv"
JOURNAL_CSV = ARCHIVE / "journal.csv"

for d in (ARCHIVE, PDF_DIR, PDF_DONE, SNAP_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

IS_PUBLIC = os.getenv("CMX_MODE", "local").lower() == "public"


# ==============
# Schémas (UNIFIÉS)
# ==============
RESULTS_LOG_COLUMNS = [
    "tournament_id", "tournament_name",
    "start_time", "processed_at",
    "Pseudo", "Position",
    "GainsCash", "Bounty",
    "Reentry", "buyin_total",
]
JOURNAL_COLUMNS = ["sha1", "filename", "processed_at"]



# ==============
# Utilitaires argent / saison
# ==============
_money_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
def parse_money(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = _money_re.search(s.replace(" ", ""))
    if not m: return 0.0
    return float(m.group(0).replace(",", "."))

def euro(v: float) -> str:
    return f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

def current_season_bounds(today: Optional[date] = None) -> tuple[date, date]:
    """Saison = 01/08 -> 31/07"""
    today = today or date.today()
    year = today.year
    season_start = date(year if today >= date(year,8,1) else year-1, 8, 1)
    season_end   = date(season_start.year+1, 7, 31)
    return season_start, season_end

def _github_upsert_files(repo: str, token: str, branch: str, files: dict[str, bytes], folder: str = "data") -> list[str]:
    """
    Crée/maj des fichiers dans un dépôt GitHub via l'API Contents.
    files: {"nom.csv": b"...", ...}
    Retourne la liste des chemins upsertés.
    """
    api = "https://api.github.com"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    touched = []
    for name, blob in files.items():
        path = f"{folder}/{name}"
        # récupère le SHA s'il existe déjà
        r = requests.get(f"{api}/repos/{repo}/contents/{path}?ref={branch}", headers=headers, timeout=30)
        sha = r.json().get("sha") if r.status_code == 200 else None

        payload = {
            "message": f"CoronaMax: update {path}",
            "content": base64.b64encode(blob).decode("ascii"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        r = requests.put(f"{api}/repos/{repo}/contents/{path}", headers=headers, json=payload, timeout=60)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Push GitHub échoué pour {path}: {r.status_code} {r.text}")
        touched.append(path)
    return touched


def _build_public_snapshot_files() -> dict[str, bytes]:
    """
    Construit les blobs à publier dans data/ :
      - results_log.csv, journal.csv, latest_master.csv, points_table.csv
      - et les PDFs ARCHIVE/PDF_TRAITES copiés dans data/PDF_Traites/
    """
    log = load_results_log_any()
    journal = load_journal_any()
    try:
        classement = standings_from_log(log, season_only=False)
    except Exception:
        classement = pd.DataFrame()

    # points sur toute la période dispo
    if not log.empty:
        d1 = log["start_time"].min().date()
        d2 = log["start_time"].max().date()
        points = compute_points_table(log, d1, d2)
    else:
        points = pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

    files: dict[str, bytes] = {}
    files["results_log.csv"]   = _normalize_results_log(log).to_csv(index=False).encode("utf-8")
    files["journal.csv"]       = _normalize_journal(journal).to_csv(index=False).encode("utf-8")
    files["latest_master.csv"] = classement.to_csv(index=False).encode("utf-8")
    files["points_table.csv"]  = points.to_csv(index=False).encode("utf-8")

    # PDFs -> sous-dossier data/PDF_Traites
    pdf_dir = DATA_DIR / "PDF_Traites"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for p in PDF_DONE.glob("*.pdf"):
        # on injecte aussi leur contenu dans le dict avec un nom de type "PDF_Traites/nom.pdf"
        files[f"PDF_Traites/{p.name}"] = p.read_bytes()

    return files

# ==============
# Normalisation CSV (robuste aux variantes)
# ==============
def _normalize_results_log(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # alias
    alias = {
        "GainCash": "GainsCash",
        "Recaves": "Reentry",
        "buyin": "buyin_total",
        "buyin_total_ttc": "buyin_total",
    }
    d = d.rename(columns={k:v for k,v in alias.items() if k in d.columns})

    # colonnes manquantes
    for c in RESULTS_LOG_COLUMNS:
        if c not in d.columns:
            d[c] = 0 if c in ("Position","GainsCash","Bounty","Reentry","buyin_total") else ""

    # types
    d["start_time"]   = pd.to_datetime(d["start_time"], errors="coerce")
    d["processed_at"] = pd.to_datetime(d["processed_at"], errors="coerce")
    for c in ("Position","Reentry"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    for c in ("GainsCash","Bounty","buyin_total"):
        d[c] = d[c].apply(parse_money).astype(float)
    d["Pseudo"] = d["Pseudo"].astype(str)

    return d[RESULTS_LOG_COLUMNS]

def _normalize_journal(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in JOURNAL_COLUMNS:
        if c not in d.columns: d[c] = ""
    d["processed_at"] = pd.to_datetime(d["processed_at"], errors="coerce")
    return d[JOURNAL_COLUMNS]

def safe_unlink(p: Path, retries: int = 5, delay: float = 0.2) -> None:
    """Supprime un fichier en réessayant si Windows signale qu'il est occupé."""
    for _ in range(retries):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
    # Dernier essai : on abandonne silencieusement (ou log si tu préfères)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

# ==============
# Loaders "any" (public ou local)
# ==============
def _choose_public_or_local(public_rel: str, local_path: Path) -> Path:
    """En mode PUBLIC on lit data/<file>, sinon on lit toujours le local."""
    if IS_PUBLIC:
        pp = DATA_DIR / public_rel
        return pp if pp.exists() else local_path
    else:
        return local_path


def load_results_log_any() -> pd.DataFrame:
    p = _choose_public_or_local("results_log.csv", RESULTS_LOG)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=RESULTS_LOG_COLUMNS)
    return _normalize_results_log(pd.read_csv(p))

def load_journal_any() -> pd.DataFrame:
    p = _choose_public_or_local("journal.csv", JOURNAL_CSV)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    return _normalize_journal(pd.read_csv(p))

def load_latest_master_any() -> pd.DataFrame:
    p = DATA_DIR / "latest_master.csv"
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
        if "Buy_in" in df.columns and "Buy in" not in df.columns:
            df = df.rename(columns={"Buy_in":"Buy in"})
        return df
    # fallback local vide
    return pd.DataFrame(columns=["Place","Pseudo","Parties","Victoires","ITM","% ITM",
                                 "Recaves","Recaves en €","Bulles","Buy in","Frais","Gains","Bénéfices"])

# Loaders / Savers locaux
def load_results_log() -> pd.DataFrame:
    """Toujours le fichier local (utilisé par l’app de traitement)."""
    p = RESULTS_LOG
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=RESULTS_LOG_COLUMNS)
    return _normalize_results_log(pd.read_csv(p))

def append_results_log(df_rows: pd.DataFrame) -> None:
    """Append rows to RESULTS_LOG (schéma normalisé)."""
    cur = load_results_log_any()
    add = _normalize_results_log(df_rows)
    out = pd.concat([cur, add], ignore_index=True)
    out.to_csv(RESULTS_LOG, index=False, encoding="utf-8")

def load_journal() -> pd.DataFrame:
    """Toujours le fichier local (utilisé par l’app de traitement)."""
    p = JOURNAL_CSV
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    return _normalize_journal(pd.read_csv(p))

def save_journal(df: pd.DataFrame) -> None:
    _normalize_journal(df).to_csv(JOURNAL_CSV, index=False, encoding="utf-8")

def save_master_df(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    """Sauvegarde CSV pour snapshot interne (Excel facultatif chez toi)."""
    p = path or (DATA_DIR / "latest_master.csv")
    df.to_csv(p, index=False, encoding="utf-8")

# ==============
# Extractions PDF Winamax (regex robustes)
# ==============
HeaderRe = re.compile(
    r"Tournoi de poker\s+(?P<name>.*?)\s+du\s+(?P<date>\d{2}[-/]\d{2}[-/]\d{4})\s+(?P<time>\d{1,2}[:h]\d{2})\s+en argent réel",
    re.IGNORECASE | re.DOTALL
)
MoneyRe = re.compile(r"(\d+[.,]\d+|\d+)\s*€")

@dataclass
class ParsedTournament:
    tournament_id: str
    tournament_name: str
    start_time: datetime
    buyin_total: float
    rows: pd.DataFrame  # colonnes: Pseudo, Position, GainsCash, Bounty, Reentry


_WS = re.compile(r"[ \t\u00A0]+")  # espace normal + insécable

def _norm(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("–", "-").replace("—", "-")
    s = s.replace("\r", "\n")
    s = re.sub(r"\n+", "\n", s)
    s = _WS.sub(" ", s)
    return s.strip()

def _parse_dt(d: str, t: str) -> datetime:
    # d: '31-08-2025' ; t: '21:15', '21h15' ou '2115'
    d = d.replace("/", "-")
    if re.fullmatch(r"\d{4}", t):
        t = f"{t[:2]}:{t[2:]}"
    t = t.replace("h", ":")
    return datetime.strptime(f"{d} {t}", "%d-%m-%Y %H:%M")

def _parse_header_from_text(text: str):
    """
    Retourne (name, datetime) en lisant le texte du PDF, sinon (None, None).
    """
    text = _norm(text)
    patterns = [
        r"tournoi de poker (?P<name>.+?) du (?P<date>\d{2}[-/]\d{2}[-/]\d{4}) (?P<time>\d{1,2}[:h]?\d{2}) en argent reel",
        r"tournoi de poker (?P<name>.+?) du (?P<date>\d{2}[-/]\d{2}[-/]\d{4}) (?P<time>\d{1,2}[:h]?\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            name = m.group("name").strip()
            dt = _parse_dt(m.group("date"), m.group("time"))
            name = re.sub(r"^winamax\.fr\s*-\s*", "", name, flags=re.I)
            return name, dt
    return None, None

def _parse_header_from_filename(fname: str):
    """
    Secours: parse le nom de fichier si le texte PDF est foireux.
    """
    base = Path(fname).stem
    s = _norm(base)
    m = re.search(
        r"(?P<name>tournoi de poker .+?) du (?P<date>\d{2}-\d{2}-\d{4}) (?P<time>(\d{2}:\d{2}|\d{4}|\d{2}h\d{2}))",
        s, flags=re.I
    )
    if m:
        name = m.group("name").strip()
        dt = _parse_dt(m.group("date"), m.group("time"))
        name = re.sub(r"^winamax\.fr\s*-\s*", "", name, flags=re.I)
        return name, dt
    return None, None

def _extract_header(pdf_text: str, pdf_path: Path):
    """
    Renvoie (tournament_name, start_time) ou lève ValueError si introuvable.
    """
    name, dt = _parse_header_from_text(pdf_text)
    if not name:
        name, dt = _parse_header_from_filename(pdf_path.name)
    if not name or not dt:
        raise ValueError("En-tete tournoi introuvable (nom/date/heure).")
    return name, dt


def _pdf_text(p: Path) -> str:
    # Lecture en mémoire pour éviter tout lock Windows
    try:
        import PyPDF2, io
        data = p.read_bytes()
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        pass
    # Fallback pdfminer (si installé)
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(p))
    except Exception:
        return ""


def extract_from_pdf(pdf_path: Path) -> ParsedTournament:
    """
    Parse un PDF Winamax 'résultats de tournoi' :
      - nom du tournoi + date/heure (gère plusieurs variantes d'en-tête)
      - buy-in + rake  => buyin_total
      - table Résultats : Position, Pseudo, GainsCash, Bounty (si KO), Reentry (recaves)
    Retourne ParsedTournament(name, start_time, buyin_total, rows: DataFrame)
    """
    import re, hashlib
    from datetime import datetime
    import pandas as pd

    txt = _pdf_text(pdf_path)

    # --- Normalisation douce (espaces insécables, tirets longs, etc.)
    norm = txt.replace("\xa0", " ").replace("—", "-")
    norm = re.sub(r"[ \t]+", " ", norm)

    # ---------- 1) En-tête = nom + date/heure ----------
    # Cas A (classique) : "Tournoi de poker <NAME> du 07-09-2025 21:15 en argent réel"
    patA = re.compile(
        r"Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+du\s+"
        r"(?P<date>\d{2}[/-]\d{2}[/-]\d{4})\s+"
        r"(?P<time>\d{2}(?::|h)?\d{2})\s+en\s+argent\s+r[ée]el",
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Cas B (variantes récentes Winamax) :
    # "... Tournoi de poker <NAME> Buy-in : 9,00 € Rake : 1,00 € ... Début du tournoi : - 07/09/2025 21:15"
    patB_name = re.compile(
        r"Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+Buy-?in",
        flags=re.IGNORECASE | re.DOTALL,
    )
    patB_dt = re.compile(
        r"Début\s+du\s+tournoi\s*:\s*-?\s*(?P<date>\d{2}[/-]\d{2}[/-]\d{4})\s+(?P<time>\d{2}(?::|h)?\d{2})",
        flags=re.IGNORECASE,
    )

    mA = patA.search(norm)
    if mA:
        name = mA.group("name").strip()
        dstr = mA.group("date").replace("/", "-")
        tstr = mA.group("time").replace("h", ":")
    else:
        mN = patB_name.search(norm)
        mD = patB_dt.search(norm)
        if not (mN and mD):
            raise RuntimeError("En-tête tournoi introuvable (nom/date/heure).")
        name = mN.group("name").strip()
        dstr = mD.group("date").replace("/", "-")
        tstr = mD.group("time").replace("h", ":")

    # Gérer l’heure écrite "2115" (sans séparateur)
    if re.fullmatch(r"\d{4}", tstr):
        tstr = f"{tstr[:2]}:{tstr[2:]}"
    start = datetime.strptime(f"{dstr} {tstr}", "%d-%m-%Y %H:%M")

    # ---------- 2) Buy-in + Rake ----------
    buy_rake_re = re.compile(
        r"Buy-?in\s*:\s*(?P<bi>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*€"
        r".{0,120}?Rake\s*:\s*(?P<rk>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*€",
        flags=re.IGNORECASE | re.DOTALL,
    )
    mbr = buy_rake_re.search(norm)
    if mbr:
        bi = parse_money(mbr.group("bi") + " €")
        rk = parse_money(mbr.group("rk") + " €")
        buyin_total = float(bi + rk)
    else:
        # Fallback : on cherche vers "Buy-in" deux montants €
        buyin_total = 0.0
        try:
            i = norm.lower().index("buy-in")
            window = norm[max(0, i - 200): i + 300]
            euros = re.findall(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*€", window)
            vals = [parse_money(v + " €") for v in euros[:2]]
            if len(vals) == 2:
                buyin_total = float(vals[0] + vals[1])
        except Exception:
            pass
        if buyin_total <= 0:
            buyin_total = 10.0  # défaut raisonnable si introuvable

    # ---------- 3) Bloc Résultats ----------
    # On repart du texte original (retours ligne utiles pour repérer les lignes)
    lines = [L.strip() for L in txt.splitlines() if L.strip()]

    # On coupe après "Résultats"
    start_idx = 0
    for i, L in enumerate(lines):
        if "Résultats" in L or "Resultats" in L:
            start_idx = i + 1
            break
    lines = lines[start_idx:]

    # Helpers regex
    money_pat = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*€")
    head_int = re.compile(r"^\s*(\d+)\b")
    tail_int = re.compile(r"(\d+)\s*$")
    reentry_inline = re.compile(r"Re-?entry\s*:?[\s=]*(\d+)", re.IGNORECASE)

    def parse_result_line(L: str):
        mpos = head_int.search(L)
        if not mpos:
            return None
        pos = int(mpos.group(1))
        rest = L[mpos.end():].strip()

        # Tous les montants de la ligne
        monies = list(money_pat.finditer(rest))
        if not monies:
            return None

        gains_span = monies[0].span()
        gains_val = parse_money(monies[0].group(1) + " €")

        bounty_val = 0.0
        if len(monies) >= 2:
            # Si KO, il y a un 2e montant qui correspond au bounty
            bounty_val = parse_money(monies[1].group(1) + " €")

        # Reentry : priorité à "Re-entry : <n>", sinon entier final
        m_inline = reentry_inline.search(rest)
        if m_inline:
            reentry_val = int(m_inline.group(1))
            core = re.sub(reentry_inline, "", rest).strip()
        else:
            reentry_val = 0
            mtail = tail_int.search(rest)
            if mtail and (not monies or mtail.start() > monies[-1].end()):
                reentry_val = int(mtail.group(1))
                core = rest[:mtail.start()].rstrip()
            else:
                core = rest

        pseudo = core[:gains_span[0]].strip()
        if not pseudo:
            return None

        return {
            "Position": pos,
            "Pseudo": pseudo,
            "GainsCash": gains_val,
            "Bounty": bounty_val,
            "Reentry": reentry_val,
        }

    rows = []
    for L in lines:
        r = parse_result_line(L)
        if r:
            rows.append(r)

    df_rows = pd.DataFrame(rows, columns=["Position", "Pseudo", "GainsCash", "Bounty", "Reentry"])
    if df_rows.empty:
        raise RuntimeError("0 ligne détectée dans la table 'Résultats'.")
    df_rows = df_rows.sort_values("Position", kind="mergesort").reset_index(drop=True)

    # ---------- 4) ID de tournoi déterministe ----------
    try:
        sha_src = (pdf_path.read_bytes() if pdf_path.exists() else b"") + \
                  (name + start.isoformat()).encode("utf-8")
    except Exception:
        sha_src = (name + start.isoformat() + str(pdf_path)).encode("utf-8")
    sha = hashlib.sha1(sha_src).hexdigest()

    return ParsedTournament(
        tournament_id=sha,
        tournament_name=name,
        start_time=start,
        buyin_total=buyin_total,
        rows=df_rows,
    )




# ==============
# Construction lignes log
# ==============
def build_rows_for_log(parsed: ParsedTournament) -> pd.DataFrame:
    d = parsed.rows.copy()
    d["tournament_id"]   = parsed.tournament_id
    d["tournament_name"] = parsed.tournament_name
    d["start_time"]      = parsed.start_time
    d["processed_at"]    = datetime.now()
    d["buyin_total"]     = parsed.buyin_total
    # ordre normalisé
    d = d[["tournament_id","tournament_name","start_time","processed_at",
           "Pseudo","Position","GainsCash","Bounty","Reentry","buyin_total"]]
    return _normalize_results_log(d)

# ==============
# Agrégations
# ==============
def _order_master_with_winamax_first(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Pseudo" in d.columns:
        is_w = d["Pseudo"].str.lower().eq("winamax")
        top = d[is_w]
        rest = d[~is_w]
        # tri par bénéfices
        if "Bénéfices" in rest.columns:
            rest = rest.sort_values("Bénéfices", ascending=False)
        d = pd.concat([top, rest], ignore_index=True)
        d.insert(0, "Place", range(len(d)))
    return d

def standings_from_log(log: pd.DataFrame, season_only: bool = False) -> pd.DataFrame:
    """
    Construit le classement "gains" à partir du RESULTS_LOG.
    Colonnes attendues (robustes) dans `log` :
      - tournament_id, Pseudo, Position, GainsCash, Bounty, Reentry, buyin_total, start_time

    Règles :
      - ITM = nb de lignes avec GainsCash > 0 (bounty ignoré)
      - Bulles = par tournoi, 1er joueur (plus petite Position) avec GainsCash <= 0
      - Frais = Buy-in (1 par tournoi et par joueur) + Recaves * buyin_total
      - Gains = GainsCash + Bounty
      - Bénéfices = Gains - Frais
      - Winamax = 10% des Frais de tous les autres, Parties = nb de tournois
    """
    import numpy as np

    # Colonnes de sortie “propres”
    OUT_COLS = ["Place","Pseudo","Parties","Victoires","ITM","% ITM",
                "Recaves","Recaves en €","Bulles","Buy in","Frais",
                "Gains","Bénéfices"]

    if log is None or log.empty:
        return pd.DataFrame(columns=OUT_COLS)

    df = log.copy()

    # --- helpers robustes
    def _pick_series(d: pd.DataFrame, candidates: list[str], default):
        for c in candidates:
            if c in d.columns:
                return d[c]
        return pd.Series([default] * len(d), index=d.index)

    # conversions de base
    pseudo   = _pick_series(df, ["Pseudo"], "").astype(str)
    pos      = pd.to_numeric(_pick_series(df, ["Position","Place","Rank"], 0), errors="coerce").fillna(0).astype(int)
    gcash    = _pick_series(df, ["GainsCash","GainCash"], 0.0).apply(parse_money).fillna(0.0)
    bounty   = _pick_series(df, ["Bounty"], 0.0).apply(parse_money).fillna(0.0)
    reentry  = pd.to_numeric(_pick_series(df, ["Reentry","Re-entry"], 0), errors="coerce").fillna(0).astype(int)
    buyin_t  = _pick_series(df, ["buyin_total","Buy in total","buy_in_total"], 0.0).apply(parse_money).fillna(0.0)
    tid      = _pick_series(df, ["tournament_id"], "").astype(str)
    stime    = pd.to_datetime(_pick_series(df, ["start_time"], pd.NaT), errors="coerce")

    # Filtre saison si demandé
    if season_only and not stime.isna().all():
        s0, s1 = current_season_bounds()
        m = (stime.dt.date >= s0) & (stime.dt.date <= s1)
        pseudo, pos, gcash, bounty, reentry, buyin_t, tid = \
            pseudo[m], pos[m], gcash[m], bounty[m], reentry[m], buyin_t[m], tid[m]

    # Table de travail
    work = pd.DataFrame({
        "Pseudo": pseudo,
        "Position": pos,
        "GainsCash": gcash,
        "Bounty": bounty,
        "Reentry": reentry,
        "buyin_total": buyin_t,
        "tournament_id": tid,
    }).reset_index(drop=True)

    if work.empty:
        return pd.DataFrame(columns=OUT_COLS)

    # ---- Bulles : par tournoi, 1er sans cash
    bubbles_list = []
    for t, grp in work.groupby("tournament_id"):
        g = grp.sort_values("Position")
        no_paid = g[g["GainsCash"] <= 0.0]
        if not no_paid.empty:
            bubbles_list.append(no_paid.iloc[0]["Pseudo"])
    bulles_count = pd.Series(bubbles_list).value_counts() if bubbles_list else pd.Series(dtype=int)

    # ---- Agrégats de base par joueur
    agg = work.groupby("Pseudo", as_index=False).agg(
        Parties   = ("Pseudo", "count"),
        Victoires = ("Position", lambda s: int((s == 1).sum())),
        ITM       = ("GainsCash", lambda s: int((s > 0).sum())),
        Recaves   = ("Reentry", "sum"),
        Buy_in    = ("buyin_total", "sum"),  # 1 buy-in par tournoi et par joueur
    )

    # Recaves en € = somme(Reentry * buyin_total)
    recaves_e = work.assign(reu=work["Reentry"] * work["buyin_total"]) \
                    .groupby("Pseudo")["reu"].sum()
    agg["Recaves en €"] = agg["Pseudo"].map(recaves_e).fillna(0.0)

    # Gains = GainsCash + Bounty
    gains_tot = work.assign(gt=work["GainsCash"] + work["Bounty"]) \
                    .groupby("Pseudo")["gt"].sum()
    agg["Gains"] = agg["Pseudo"].map(gains_tot).fillna(0.0)

    # Frais & Bénéfices
    agg["Frais"] = agg["Buy_in"].fillna(0.0) + agg["Recaves en €"].fillna(0.0)
    agg["Bénéfices"] = agg["Gains"].fillna(0.0) - agg["Frais"].fillna(0.0)

    # Bulles mappées
    agg["Bulles"] = agg["Pseudo"].map(bulles_count).fillna(0).astype(int)

    # % ITM
    agg["% ITM"] = agg.apply(
        lambda r: f"{int(round(100 * r['ITM'] / r['Parties']))}%"
        if r["Parties"] else "0%",
        axis=1
    )

    # Renommage/ordre
    agg = agg.rename(columns={"Buy_in": "Buy in"})
    agg = agg[["Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
               "Bulles","Buy in","Frais","Gains","Bénéfices"]]

    # ---- Ajout WINAMAX
    total_frais_autres = float(agg["Frais"].sum())
    n_tourneys = work["tournament_id"].nunique()
    wina = pd.DataFrame([{
        "Pseudo": "WINAMAX",
        "Parties": int(n_tourneys),
        "Victoires": 0,
        "ITM": 0,
        "% ITM": "0%",
        "Recaves": 0,
        "Recaves en €": 0.0,
        "Bulles": 0,
        "Buy in": 0.0,
        "Frais": 0.0,
        "Gains": round(total_frais_autres * 0.10, 2),
        "Bénéfices": round(total_frais_autres * 0.10, 2),
    }])

    out = pd.concat([wina, agg[agg["Pseudo"].str.lower() != "winamax"]], ignore_index=True)

    # Tri : Winamax en haut, puis bénéfices décroissants
    others = out[out["Pseudo"].str.lower() != "winamax"].sort_values("Bénéfices", ascending=False)
    out = pd.concat([out[out["Pseudo"].str.lower() == "winamax"], others], ignore_index=True)

    # Place (Winamax = 0, joueurs = 1..N)
    places = [0] + list(range(1, len(out)))
    out.insert(0, "Place", places)

    # Types propres
    for c in ["Parties","Victoires","ITM","Recaves","Bulles","Place"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    for c in ["Recaves en €","Buy in","Frais","Gains","Bénéfices"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out[OUT_COLS]


def compute_points_table(log: pd.DataFrame, d1: date, d2: date) -> pd.DataFrame:
    if log is None or log.empty:
        return pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])
    g = _normalize_results_log(log).copy()
    g = g[(g["start_time"].dt.date >= d1) & (g["start_time"].dt.date <= d2)]
    if g.empty:
        return pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

    n_by_t = g.groupby("tournament_id")["Pseudo"].transform("size")
    points = (n_by_t - g["Position"] + 1).clip(lower=1)
    itm    = (g["GainsCash"] > 0).astype(int)
    vic    = (g["Position"] == 1).astype(int)

    tmp = pd.DataFrame({"Pseudo":g["Pseudo"],"Points":points,"ITM":itm,"Victoires":vic,"Parties":1})
    tmp = tmp[~tmp["Pseudo"].str.lower().eq("winamax")]  # pas de Winamax ici
    agg = tmp.groupby("Pseudo", as_index=False).sum(numeric_only=True)
    agg = agg.sort_values(["Points","Victoires","ITM","Parties"], ascending=[False,False,False,False])
    agg.insert(0, "Place", range(1, len(agg)+1))
    return agg[["Place","Pseudo","Parties","ITM","Victoires","Points"]]


# --- Utilitaire: bulle à partir d’un DF (Positions / GainCash) ---
def compute_bubble_from_rows(df_rows: pd.DataFrame) -> str | None:
    if df_rows is None or len(df_rows) == 0:
        return None
    d = df_rows.copy()

    def as_series(df, name, default):
        if name in df.columns:
            s = df[name]
            return s if hasattr(s, "shape") else pd.Series([default] * len(df), index=df.index)
        return pd.Series([default] * len(df), index=df.index)

    pos = pd.to_numeric(as_series(d, "Position", 0), errors="coerce").fillna(0).astype(int)
    gains = pd.to_numeric(as_series(d, "GainsCash", 0.0), errors="coerce").fillna(0.0)
    pseudo = as_series(d, "Pseudo", "").astype(str)

    tmp = pd.DataFrame({"Position": pos, "GainsCash": gains, "Pseudo": pseudo}).sort_values("Position")
    no_paid = tmp.loc[tmp["GainsCash"] <= 0.0]
    return None if no_paid.empty else str(no_paid.iloc[0]["Pseudo"])

def _player_stats_from_log(log: pd.DataFrame, who: str) -> dict:
    sub = log[log["Pseudo"].astype(str) == who].copy()
    if sub.empty:
        return {}

    sub["Position"]   = pd.to_numeric(sub["Position"], errors="coerce").fillna(0).astype(int)
    sub["GainsCash"]  = pd.to_numeric(sub.get("GainsCash", 0.0).apply(parse_money), errors="coerce").fillna(0.0)
    sub["Bounty"]     = pd.to_numeric(sub.get("Bounty", 0.0).apply(parse_money), errors="coerce").fillna(0.0)
    sub["buyin_total"]= pd.to_numeric(sub.get("buyin_total", 0.0), errors="coerce").fillna(0.0)
    sub["Reentry"]    = pd.to_numeric(sub.get("Reentry", 0), errors="coerce").fillna(0).astype(int)

    n = len(sub)
    wins = int((sub["Position"] == 1).sum())
    itm  = int((sub["GainsCash"] > 0.0).sum())
    bubbles = 0
    # bulles: “premier non payé” par tournoi (gainscash==0 & position==min_position_non_payé)
    for tid, g in sub.groupby("tournament_id"):
        g = g.sort_values("Position")
        first_no_cash = g[g["GainsCash"] <= 0.0]
        if not first_no_cash.empty and first_no_cash.iloc[0]["Pseudo"] == who:
            bubbles += 1

    last_pos = int(sub["Position"].max()) if n else 0
    last_place = int((sub["Position"] == last_pos).sum()) if last_pos else 0

    fees = (sub["buyin_total"] * (1 + sub["Reentry"])).sum()
    gains = (sub["GainsCash"] + sub["Bounty"]).sum()
    roi = (gains - fees) / fees if fees > 0 else 0.0

    return {
        "n": n, "wins": wins, "itm": itm, "itm_rate": (itm/n if n else 0.0),
        "bubbles": bubbles, "last_place": last_place,
        "avg_pos": float(sub["Position"].mean()) if n else 0.0,
        "fees": fees, "gains": gains, "benef": gains - fees, "roi": roi,
        "table": sub.sort_values("start_time")
    }


def render_player_details(log: pd.DataFrame):
    if log.empty:
        st.info("Aucun historique pour l’instant.")
        return

    who = st.selectbox("Choisir un joueur", sorted(log["Pseudo"].astype(str).unique().tolist()))
    Z = _player_stats_from_log(log, who)
    if not Z:
        st.info("Pas de données pour ce joueur.")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Parties", Z["n"])
    c2.metric("Victoires", Z["wins"])
    c3.metric("ITM", f"{Z['itm']} ({int(round(Z['itm_rate']*100))}%)")
    c4.metric("Bulles", Z["bubbles"])
    c5.metric("Dernières places", Z["last_place"])
    c6.metric("Position moy.", f"{Z['avg_pos']:.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Frais", euro(Z["fees"]))
    c2.metric("Gains", euro(Z["gains"]))
    c3.metric("Bénéfices", euro(Z["benef"]))

    # courbe cumulé frais/gains/bénéfices
    sub = Z["table"].copy()
    sub["Frais"] = sub["buyin_total"] * (1 + sub["Reentry"])
    sub["GainsTot"] = sub["GainsCash"] + sub["Bounty"]
    sub["Bénéfices"] = sub["GainsTot"] - sub["Frais"]
    sub = sub.sort_values("start_time")

    import altair as alt
    series = sub[["start_time","Frais","GainsTot","Bénéfices"]].rename(columns={"start_time":"Date"})
    st.altair_chart(
        alt.Chart(series.melt("Date", var_name="Type", value_name="Valeur"))
        .mark_line(point=True)
        .encode(x="Date:T", y="Valeur:Q", color="Type:N")
        .properties(height=320),
        use_container_width=True
    )

    # répartition des positions
    pos_count = sub["Position"].value_counts().sort_index().reset_index()
    pos_count.columns = ["Position","Count"]
    st.altair_chart(
        alt.Chart(pos_count).mark_bar().encode(x="Position:O", y="Count:Q").properties(height=200),
        use_container_width=True
    )

    # tableau détaillé (lisible)
    show = sub[["start_time","tournament_name","Position","Reentry","buyin_total","Frais","GainsCash","Bounty","GainsTot","Bénéfices"]].copy()
    for c in ["buyin_total","Frais","GainsCash","Bounty","GainsTot","Bénéfices"]:
        show[c] = show[c].apply(euro)
    show = show.rename(columns={"start_time":"Date","tournament_name":"Tournoi"})
    st.dataframe(show, use_container_width=True, hide_index=True)


# ==============
# Exports JPG (classement)
# ==============
def classement_df_to_jpg(df: pd.DataFrame, out_path: Path, dpi: int = 220):
    """
    Rend un JPG propre du classement :
    - en-têtes gris
    - Pseudo: WINAMAX rouge/blanc League Gothic, autres orange/noir
    - Dégradé fort sur Bénéfices
    - Bordures noires
    """
    import matplotlib.pyplot as plt
    import colorsys
    import numpy as np

    # --- util: HSL (CSS) -> RGB (0..1) pour matplotlib
    def hsl_to_rgb_tuple(h_deg: float, s_pct: float, l_pct: float):
        # colorsys prend H,L,S normalisés; CSS c'est H,S,L
        r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l_pct / 100.0, s_pct / 100.0)
        return (float(r), float(g), float(b))  # matplotlib accepte (r,g,b) floats

    # Colonnes dans l'ordre attendu si elles existent
    cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
            "Bulles","Buy in","Frais","Gains","Bénéfices"]
    cols = [c for c in cols if c in df.columns]
    data = df[cols].copy()

    # Valeurs affichées
    def _fmt_money(x): return euro(parse_money(x))
    def _fmt_pct(x):
        try:
            v = float(x)
            return f"{int(round(v*100, 0))}%"
        except Exception:
            return str(x)

    shown = data.copy()
    if "% ITM" in shown.columns:
        shown["% ITM"] = shown["% ITM"].apply(_fmt_pct)
    for c in ["Recaves en €","Buy in","Frais","Gains","Bénéfices"]:
        if c in shown.columns:
            shown[c] = shown[c].apply(_fmt_money)

    # Styles cellule par cellule
    nrows, ncols = shown.shape
    cell_colours = [["#ffffff"] * ncols for _ in range(nrows)]
    text_colors  = [["#000000"] * ncols for _ in range(nrows)]

    # Dégradé Bénéfices (HSL -> RGB)
    if "Bénéfices" in data.columns:
        vals = pd.to_numeric(data["Bénéfices"].apply(parse_money), errors="coerce").fillna(0.0)
        vmin, vmax = float(vals.min()), float(vals.max())
        span = (vmax - vmin) or 1.0
        j = shown.columns.get_loc("Bénéfices")
        for i, v in enumerate(vals):
            t = (float(v) - vmin) / span  # 0..1
            hue = 120.0 * t               # 0 rouge -> 120 vert
            rgb = hsl_to_rgb_tuple(hue, 80.0, 78.0)
            cell_colours[i][j] = rgb
            text_colors[i][j]  = "#000000"

    # Pseudo : Winamax & co
    if "Pseudo" in shown.columns:
        j = shown.columns.get_loc("Pseudo")
        for i, name in enumerate(shown["Pseudo"].astype(str)):
            if name.strip().lower() == "winamax":
                cell_colours[i][j] = "#c62828"
                text_colors[i][j]  = "#ffffff"
                shown.iloc[i, j]   = "WINAMAX"
            else:
                cell_colours[i][j] = "#f7b329"
                text_colors[i][j]  = "#000000"

    # Figure
    base_row_h = 0.4
    fig_h = max(3.0, base_row_h * (nrows + 2))
    fig_w = 1.1 + 0.95 * ncols
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")

    header_color = "#e6e6e6"
    header_text  = "#000000"

    the_table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellColours=cell_colours,
        colColours=[header_color]*ncols,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    # Bordures + styles
    for (r, c), cell in the_table.get_celld().items():
        cell.set_edgecolor("#222")
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_text_props(color=header_text, weight="bold")
        else:
            cell.get_text().set_color(text_colors[r-1][c])
            if shown.columns[c] == "Pseudo" and str(shown.iloc[r-1, c]).upper() == "WINAMAX":
                cell.get_text().set_fontfamily("League Gothic")
                cell.get_text().set_fontweight("heavy")

    # Pointillés SOUS Winamax (une seule ligne horizontale, pas tous les bords)
    try:
        # r = index de la ligne Winamax dans la table matplotlib (0 = en-tête)
        w_idx = data.index[data["Pseudo"].str.lower() == "winamax"][0]
        r = w_idx + 1  # +1 car la ligne 0 est l'en-tête

        # Coordonnées des cellules extrêmes de cette ligne (en unités d'axes)
        left_cell  = the_table[(r, 0)]
        right_cell = the_table[(r, ncols - 1)]
        x0, y0 = left_cell.xy                      # bas-gauche de la cellule de gauche
        x1 = right_cell.xy[0] + right_cell.get_width()  # bord droit de la cellule droite
        y  = y0  # bord inférieur de la ligne Winamax

        # Trace une ligne horizontale en pointillés SOUS la ligne Winamax
        ax.add_line(plt.Line2D(
            [x0, x1], [y, y],
            transform=ax.transAxes,
            linestyle=(0, (6, 4)),
            color="#222",
            linewidth=2.0
        ))
    except Exception:
        pass


    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def classement_points_df_to_jpg(df: pd.DataFrame, out_path: Path) -> None:
    """
    Rend un JPG du tableau 'classement par points'.
    - largeur/hauteur calculées dynamiquement pour éviter tout rognage,
    - fond orange sur la colonne 'Pseudo',
    - en-têtes gris, bordures foncées.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not isinstance(out_path, Path):
        out_path = Path(out_path)

    # Sécurise l'ordre des colonnes si possible
    wanted = ["Place", "Pseudo", "Parties", "ITM", "Victoires", "Points"]
    cols = [c for c in wanted if c in df.columns] or df.columns.tolist()
    d = df[cols].copy()

    # --------- Dimensions auto ----------
    n_rows, n_cols = d.shape
    # largeur de base par colonne (un peu plus large pour 'Pseudo')
    base_w = 1.8
    extra = 1.2 if "Pseudo" in d.columns else 0.0
    fig_w = max(14.0, base_w * n_cols + extra)        # >= 14 pouces
    row_h = 0.62                                      # hauteur par ligne
    head_h = 0.85
    fig_h = max(5.0, head_h + row_h * (n_rows))       # >= 5 pouces

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.axis("off")

    # Convertit en str (pour éviter les accents/format age bizarres)
    data = d.astype(str).values.tolist()
    headers = [str(c) for c in d.columns.tolist()]

    # Table
    tbl = ax.table(cellText=data, colLabels=headers, cellLoc="center",
                   loc="upper left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    # Augmente la hauteur des lignes (évite la densité trop forte)
    tbl.scale(1, 1.35)

    # Largeurs automatiques (toutes colonnes)
    try:
        tbl.auto_set_column_width(list(range(n_cols)))
    except Exception:
        pass

    # Styles en-têtes
    for j in range(n_cols):
        cell = tbl[0, j]  # ligne d'en-tête = 0
        cell.set_facecolor("#e6e6e6")
        cell.set_edgecolor("#222222")
        cell.set_linewidth(1.2)
        cell.set_text_props(weight="bold", color="#000000")

    # Styles cellules + fonds de la colonne Pseudo
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_edgecolor("#222222")
            cell.set_linewidth(0.8)
            if headers[j].lower() == "pseudo":
                cell.set_facecolor("#f7b329")
                cell.set_text_props(color="#000000", weight="bold")

    # Sauvegarde sans rognage
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)

# ==============
# Archive PDF & rollback
# ==============
def archive_pdf(src: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = PDF_DONE / f"{src.stem}__{ts}.pdf"
    PDF_DONE.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest

def rollback_last_import() -> dict:
    """Supprime les dernières lignes du log (dernier tournoi) et remet le PDF en entrée si possible."""
    log = load_results_log_any()
    if log.empty:
        return {"ok": False, "msg": "Log vide."}
    # dernier tournoi par processed_at
    last_ts = log["processed_at"].max()
    last_ids = log.loc[log["processed_at"] == last_ts, "tournament_id"].unique().tolist()
    if not last_ids:
        return {"ok": False, "msg": "Aucun import récent."}
    last_id = last_ids[-1]
    # supprime ces lignes
    log2 = log[log["tournament_id"] != last_id].copy()
    log2.to_csv(RESULTS_LOG, index=False, encoding="utf-8")

    # tente de retrouver le PDF archivé le plus récent
    back_pdf = ""
    pdfs = sorted(PDF_DONE.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pdfs:
        src = pdfs[0]
        dst = PDF_DIR / (src.name.split("__")[0] + ".pdf")
        shutil.move(str(src), str(dst))
        back_pdf = dst.name
    return {"ok": True, "msg": "Dernier import annulé.", "pdf_back": back_pdf}

# ==============
# Snapshot public (écrit dans data/)
# ==============
def publish_public_snapshot(push_to_github: bool = False, message: str = "CoronaMax: snapshot") -> tuple[bool, str]:
    """
    1) Génère le snapshot local dans BASE/data/ (toujours).
    2) Si push_to_github=True et variables GIT_PUBLIC_REPO / GIT_TOKEN présentes,
       pousse aussi vers GitHub (branche GIT_BRANCH, défaut 'main') via l'API.
    """
    files = _build_public_snapshot_files()

    # Écriture locale dans BASE/data/
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        path = DATA_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)

    # Option GitHub
    if push_to_github:
        repo   = os.getenv("GIT_PUBLIC_REPO", "").strip()   # ex: "OdrAAdeKK/coronamax-public"
        token  = os.getenv("GIT_TOKEN", "").strip()
        branch = os.getenv("GIT_BRANCH", "main").strip()
        if not repo or not token:
            return False, "Variables GIT_PUBLIC_REPO et/ou GIT_TOKEN manquantes : aucun push GitHub."

        # On pousse *tout* sous data/ (CSV + PDFs)
        try:
            touched = _github_upsert_files(repo, token, branch, files, folder="data")
            return True, f"Snapshot locale écrite + {len(touched)} fichier(s) poussé(s) vers {repo}@{branch}."
        except Exception as e:
            return False, f"Snapshot locale OK mais push GitHub KO : {e}"

    # Pas de push : snapshot uniquement local
    return True, "Snapshot locale générée (aucun push GitHub demandé)."
