# app_classement_unique.py
from __future__ import annotations

import os, re, io, shutil, hashlib, json, time, sys, subprocess
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional
import unicodedata
from urllib.parse import quote

import numpy as np
import pandas as pd

# -- streamlit est optionnel ici : on protège l'accès à secrets
try:
    import streamlit as st
except Exception:
    st = None  # type: ignore


# =============================================================================
# Bases & secrets → env (marche en local, en exe, et en Streamlit Cloud)
# =============================================================================

def _app_base() -> Path:
    """
    Racine de l'app :
    - si CMX_BASE_DIR est défini -> utilise ce dossier (EXE portable)
    - sinon si bundlé (sys.frozen) -> dossier de l'exe
    - sinon -> dossier du script courant
    """
    if os.getenv("CMX_BASE_DIR"):
        return Path(os.getenv("CMX_BASE_DIR") or ".").resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.resolve()


def _secrets_to_env() -> None:
    """
    Si streamlit.secrets existe, copie quelques clés vers os.environ.
    Tolère l'absence de streamlit en EXE/CLI.
    """
    keys = ("GIT_PUBLIC_REPO", "GIT_TOKEN", "GIT_BRANCH", "GIT_AUTHOR", "ADMIN_KEY")
    try:
        if st and hasattr(st, "secrets"):
            for k in keys:
                try:
                    v = str(st.secrets.get(k, "")).strip()
                    if v:
                        os.environ[k] = v
                except Exception:
                    pass
    except Exception:
        pass


BASE = _app_base()
_secrets_to_env()

ARCHIVE   = BASE / "ARCHIVE"
PDF_DIR   = BASE / "PDF_A_TRAITER"
PDF_DONE  = ARCHIVE / "PDF_TRAITES"
SNAP_DIR  = BASE / "SNAPSHOTS"
DATA_DIR  = BASE / "data"           # snapshot public

F_MASTER   = BASE / "GAINS_Wina.xlsx"  # facultatif (fallback local)
RESULTS_LOG = ARCHIVE / "results_log.csv"
JOURNAL_CSV = ARCHIVE / "journal.csv"

for d in (ARCHIVE, PDF_DIR, PDF_DONE, SNAP_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

IS_PUBLIC = os.getenv("CMX_MODE", "local").lower() == "public"


def _normalize_repo_url(repo_url: str) -> str:
    """
    Accepte:
      - 'https://github.com/owner/repo.git'
      - 'https://github.com/owner/repo'
      - 'github.com/owner/repo'
      - 'owner/repo'
    Et renvoie toujours une URL https complète, avec .git.
    """
    u = (repo_url or "").strip()
    if not u:
        return ""
    if u.startswith("https://") or u.startswith("http://"):
        # Normalise le host github + force .git
        if "github.com/" in u and not u.endswith(".git"):
            u += ".git"
        return u
    if u.startswith("github.com/"):
        v = u.split("github.com/", 1)[1]
        if not v.endswith(".git"):
            v += ".git"
        return f"https://github.com/{v}"
    # Format court: owner/repo(.git)
    if re.match(r"^[\w.-]+/[\w.-]+(?:\.git)?$", u):
        if not u.endswith(".git"):
            u += ".git"
        return f"https://github.com/{u}"
    return u


# =============================================================================
# Schémas / argent / saison
# =============================================================================

RESULTS_LOG_COLUMNS = [
    "tournament_id", "tournament_name",
    "start_time", "processed_at",
    "Pseudo", "Position",
    "GainsCash", "Bounty",
    "Reentry", "buyin_total",
]
JOURNAL_COLUMNS = ["sha1", "filename", "processed_at"]

_money_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def parse_money(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = _money_re.search(s.replace(" ", ""))
    if not m:
        return 0.0
    return float(m.group(0).replace(",", "."))

def euro(v: float) -> str:
    return f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

def current_season_bounds(today: Optional[date] = None) -> tuple[date, date]:
    """Saison = 01/08 -> 31/07"""
    today = today or date.today()
    year = today.year
    s0 = date(year if today >= date(year,8,1) else year-1, 8, 1)
    s1 = date(s0.year+1, 7, 31)
    return s0, s1


# =============================================================================
# Normalisation CSV & I/O
# =============================================================================

def _normalize_results_log(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    alias = {
        "GainCash": "GainsCash",
        "Recaves": "Reentry",
        "buyin": "buyin_total",
        "buyin_total_ttc": "buyin_total",
    }
    d = d.rename(columns={k: v for k, v in alias.items() if k in d.columns})

    for c in RESULTS_LOG_COLUMNS:
        if c not in d.columns:
            d[c] = 0 if c in ("Position","GainsCash","Bounty","Reentry","buyin_total") else ""

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
        if c not in d.columns:
            d[c] = ""
    d["processed_at"] = pd.to_datetime(d["processed_at"], errors="coerce")
    return d[JOURNAL_COLUMNS]

def safe_unlink(p: Path, retries: int = 5, delay: float = 0.2) -> None:
    for _ in range(retries):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _choose_public_or_local(public_rel: str, local_path: Path) -> Path:
    pp = DATA_DIR / public_rel
    return pp if pp.exists() else local_path

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
    return pd.DataFrame(columns=["Place","Pseudo","Parties","Victoires","ITM","% ITM",
                                 "Recaves","Recaves en €","Bulles","Buy in","Frais","Gains","Bénéfices"])

def load_journal() -> pd.DataFrame:
    """
    Charge ARCHIVE/journal.csv.
    Si absent ou vide, renvoie un DataFrame vide avec le bon schéma.
    """
    p = JOURNAL_CSV
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    try:
        df = pd.read_csv(p)
    except Exception:
        # fichier corrompu / encodage : repart sur un DF vide mais typé
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    return _normalize_journal(df)


def save_journal(df: pd.DataFrame) -> None:
    _normalize_journal(df).to_csv(JOURNAL_CSV, index=False, encoding="utf-8")

def append_results_log(df_rows: pd.DataFrame) -> None:
    cur = load_results_log_any()
    add = _normalize_results_log(df_rows)
    out = pd.concat([cur, add], ignore_index=True)
    out.to_csv(RESULTS_LOG, index=False, encoding="utf-8")

def save_master_df(df: pd.DataFrame, path: Optional[Path] = None) -> None:
    p = path or (DATA_DIR / "latest_master.csv")
    df.to_csv(p, index=False, encoding="utf-8")


# =============================================================================
# Extraction PDF Winamax (robuste)
# =============================================================================

@dataclass
class ParsedTournament:
    tournament_id: str
    tournament_name: str
    start_time: datetime
    buyin_total: float
    rows: pd.DataFrame  # Position, Pseudo, GainsCash, Bounty, Reentry


def _pdf_text(p: Path) -> str:
    # 1) PyPDF2 (mémoire) ; fallback pdfminer
    try:
        import PyPDF2
        data = p.read_bytes()
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(p))
    except Exception:
        return ""


def extract_from_pdf(pdf_path: Path) -> ParsedTournament:
    """
    Parse un PDF 'résultats de tournoi' Winamax :
      - nom + date/heure (plusieurs variantes)
      - Buy-in + Rake => buyin_total
      - table Résultats (Position, Pseudo, GainsCash, Bounty, Reentry)
    """
    txt = _pdf_text(pdf_path)

    # Normalisation
    norm = (txt or "").replace("\xa0", " ").replace("—", "-")
    norm = re.sub(r"[ \t]+", " ", norm)

    # ---------- En-tête ----------
    patA = re.compile(
        r"Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+du\s+"
        r"(?P<date>\d{2}[/-]\d{2}[/-]\d{4})\s+"
        r"(?P<time>\d{2}(?::|h)?\d{2})\s+en\s+argent\s+r[ée]el",
        flags=re.IGNORECASE | re.DOTALL,
    )
    patB_name = re.compile(r"Tournoi\s+de\s+poker\s+(?P<name>.+?)\s+Buy-?in", re.IGNORECASE|re.DOTALL)
    patB_dt   = re.compile(r"Début\s+du\s+tournoi\s*:\s*-?\s*(?P<date>\d{2}[/-]\d{2}[/-]\d{4})\s+(?P<time>\d{2}(?::|h)?\d{2})", re.IGNORECASE)

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

    if re.fullmatch(r"\d{4}", tstr):
        tstr = f"{tstr[:2]}:{tstr[2:]}"
    start = datetime.strptime(f"{dstr} {tstr}", "%d-%m-%Y %H:%M")

    # ---------- Buy-in + Rake ----------
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
            buyin_total = 10.0

    # ---------- Résultats ----------
    lines = [L.strip() for L in (txt or "").splitlines() if L.strip()]

    # couper après "Résultats"
    start_idx = 0
    for i, L in enumerate(lines):
        if "Résultats" in L or "Resultats" in L:
            start_idx = i + 1
            break
    lines = lines[start_idx:]

    money_pat = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*€")
    head_int  = re.compile(r"^\s*(\d+)\b")
    tail_int  = re.compile(r"(\d+)\s*$")
    reentry_inline = re.compile(r"Re-?entry\s*:?[\s=]*(\d+)", re.IGNORECASE)

    def parse_line(L: str):
        mpos = head_int.search(L)
        if not mpos:
            return None
        pos = int(mpos.group(1))
        rest = L[mpos.end():].strip()

        monies = list(money_pat.finditer(rest))
        if not monies:
            return None

        gains_span = monies[0].span()
        gains_val  = parse_money(monies[0].group(1) + " €")
        bounty_val = parse_money(monies[1].group(1) + " €") if len(monies) >= 2 else 0.0

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
        r = parse_line(L)
        if r:
            rows.append(r)

    df_rows = pd.DataFrame(rows, columns=["Position","Pseudo","GainsCash","Bounty","Reentry"])
    if df_rows.empty:
        raise RuntimeError("0 ligne détectée dans la table 'Résultats'.")
    df_rows = df_rows.sort_values("Position", kind="mergesort").reset_index(drop=True)

    # ---------- ID déterministe ----------
    try:
        sha_src = (pdf_path.read_bytes() if pdf_path.exists() else b"") + (name + start.isoformat()).encode("utf-8")
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


def build_rows_for_log(parsed: ParsedTournament) -> pd.DataFrame:
    d = parsed.rows.copy()
    d["tournament_id"]   = parsed.tournament_id
    d["tournament_name"] = parsed.tournament_name
    d["start_time"]      = parsed.start_time
    d["processed_at"]    = datetime.now()
    d["buyin_total"]     = parsed.buyin_total
    d = d[["tournament_id","tournament_name","start_time","processed_at",
           "Pseudo","Position","GainsCash","Bounty","Reentry","buyin_total"]]
    return _normalize_results_log(d)


# =============================================================================
# Agrégations
# =============================================================================

def standings_from_log(log: pd.DataFrame, season_only: bool = False) -> pd.DataFrame:
    """
    Classement “gains” :
      ITM = GainsCash > 0
      Bulles = par tournoi, 1er joueur non payé
      Frais = buyin_total * (1 + Reentry)
      Gains = GainsCash + Bounty
      Bénéfices = Gains - Frais
      Winamax = 10% des frais de tous les autres ; Parties = nb tournois
    """
    OUT_COLS = ["Place","Pseudo","Parties","Victoires","ITM","% ITM",
                "Recaves","Recaves en €","Bulles","Buy in","Frais",
                "Gains","Bénéfices"]

    if log is None or log.empty:
        return pd.DataFrame(columns=OUT_COLS)

    df = log.copy()

    def _pick_series(d: pd.DataFrame, candidates: list[str], default):
        for c in candidates:
            if c in d.columns:
                return d[c]
        return pd.Series([default] * len(d), index=d.index)

    pseudo   = _pick_series(df, ["Pseudo"], "").astype(str)
    pos      = pd.to_numeric(_pick_series(df, ["Position","Place","Rank"], 0), errors="coerce").fillna(0).astype(int)
    gcash    = _pick_series(df, ["GainsCash","GainCash"], 0.0).apply(parse_money).fillna(0.0)
    bounty   = _pick_series(df, ["Bounty"], 0.0).apply(parse_money).fillna(0.0)
    reentry  = pd.to_numeric(_pick_series(df, ["Reentry","Re-entry"], 0), errors="coerce").fillna(0).astype(int)
    buyin_t  = _pick_series(df, ["buyin_total","Buy in total","buy_in_total"], 0.0).apply(parse_money).fillna(0.0)
    tid      = _pick_series(df, ["tournament_id"], "").astype(str)
    stime    = pd.to_datetime(_pick_series(df, ["start_time"], pd.NaT), errors="coerce")

    if season_only and not stime.isna().all():
        s0, s1 = current_season_bounds()
        m = (stime.dt.date >= s0) & (stime.dt.date <= s1)
        pseudo, pos, gcash, bounty, reentry, buyin_t, tid = \
            pseudo[m], pos[m], gcash[m], bounty[m], reentry[m], buyin_t[m], tid[m]

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

    # Bulles
    bubbles_list = []
    for t, grp in work.groupby("tournament_id"):
        g = grp.sort_values("Position")
        no_paid = g[g["GainsCash"] <= 0.0]
        if not no_paid.empty:
            bubbles_list.append(no_paid.iloc[0]["Pseudo"])
    bulles_count = pd.Series(bubbles_list).value_counts() if bubbles_list else pd.Series(dtype=int)

    agg = work.groupby("Pseudo", as_index=False).agg(
        Parties   = ("Pseudo", "count"),
        Victoires = ("Position", lambda s: int((s == 1).sum())),
        ITM       = ("GainsCash", lambda s: int((s > 0).sum())),
        Recaves   = ("Reentry", "sum"),
        Buy_in    = ("buyin_total", "sum"),
    )

    recaves_e = work.assign(reu=work["Reentry"] * work["buyin_total"]).groupby("Pseudo")["reu"].sum()
    agg["Recaves en €"] = agg["Pseudo"].map(recaves_e).fillna(0.0)

    gains_tot = work.assign(gt=work["GainsCash"] + work["Bounty"]).groupby("Pseudo")["gt"].sum()
    agg["Gains"] = agg["Pseudo"].map(gains_tot).fillna(0.0)

    agg["Frais"] = agg["Buy_in"].fillna(0.0) + agg["Recaves en €"].fillna(0.0)
    agg["Bénéfices"] = agg["Gains"].fillna(0.0) - agg["Frais"].fillna(0.0)
    agg["Bulles"] = agg["Pseudo"].map(bulles_count).fillna(0).astype(int)
    agg["% ITM"] = agg.apply(lambda r: f"{int(round(100 * r['ITM'] / r['Parties']))}%" if r["Parties"] else "0%", axis=1)

    agg = agg.rename(columns={"Buy_in": "Buy in"})
    agg = agg[["Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
               "Bulles","Buy in","Frais","Gains","Bénéfices"]]

    total_frais_autres = float(agg["Frais"].sum())
    n_tourneys = work["tournament_id"].nunique()
    wina = pd.DataFrame([{
        "Pseudo": "WINAMAX",
        "Parties": int(n_tourneys),
        "Victoires": 0, "ITM": 0, "% ITM": "0%",
        "Recaves": 0, "Recaves en €": 0.0, "Bulles": 0,
        "Buy in": 0.0, "Frais": 0.0,
        "Gains": round(total_frais_autres * 0.10, 2),
        "Bénéfices": round(total_frais_autres * 0.10, 2),
    }])

    out = pd.concat([wina, agg[agg["Pseudo"].str.lower() != "winamax"]], ignore_index=True)
    others = out[out["Pseudo"].str.lower() != "winamax"].sort_values("Bénéfices", ascending=False)
    out = pd.concat([out[out["Pseudo"].str.lower() == "winamax"], others], ignore_index=True)

    places = [0] + list(range(1, len(out)))
    out.insert(0, "Place", places)

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
    tmp = tmp[~tmp["Pseudo"].str.lower().eq("winamax")]
    agg = tmp.groupby("Pseudo", as_index=False).sum(numeric_only=True)
    agg = agg.sort_values(["Points","Victoires","ITM","Parties"], ascending=[False,False,False,False])
    agg.insert(0, "Place", range(1, len(agg)+1))
    return agg[["Place","Pseudo","Parties","ITM","Victoires","Points"]]


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
    """
    Statistiques d'un joueur 'who' sur le DataFrame 'log' (déjà filtré si besoin).
    Correction bulle: on calcule la bulle par tournoi sur le log COMPLET,
    puis on compte uniquement ceux où le pseudo bulle == who.
    """
    if log is None or log.empty:
        return {}

    # Normalisation robuste
    g = _normalize_results_log(log).copy()
    g["Position"]    = pd.to_numeric(g["Position"], errors="coerce").fillna(0).astype(int)
    g["GainsCash"]   = pd.to_numeric(g["GainsCash"], errors="coerce").fillna(0.0)
    g["Bounty"]      = pd.to_numeric(g.get("Bounty", 0.0), errors="coerce").fillna(0.0)
    g["buyin_total"] = pd.to_numeric(g.get("buyin_total", 0.0), errors="coerce").fillna(0.0)
    g["Reentry"]     = pd.to_numeric(g.get("Reentry", 0), errors="coerce").fillna(0).astype(int)
    g["Pseudo"]      = g["Pseudo"].astype(str)
    who = str(who)

    # Sous-ensemble du joueur (pour le reste des stats)
    sub = g[g["Pseudo"] == who].copy()
    if sub.empty:
        return {}

    # --- Bulles: par tournoi = 1er joueur (position la plus haute) avec GainsCash <= 0
    bubbles_by_tid = {}
    for tid, grp in g.groupby("tournament_id"):
        grp = grp.sort_values("Position")
        no_paid = grp[grp["GainsCash"] <= 0.0]
        if not no_paid.empty:
            bubbles_by_tid[str(tid)] = str(no_paid.iloc[0]["Pseudo"])
    player_tids = set(sub["tournament_id"].astype(str))
    bubbles = sum(1 for tid in player_tids if bubbles_by_tid.get(tid) == who)

    # --- Agrégats classiques
    n     = int(len(sub))
    wins  = int((sub["Position"] == 1).sum())
    itm   = int((sub["GainsCash"] > 0.0).sum())
    fees  = float((sub["buyin_total"] * (1 + sub["Reentry"])).sum())
    gains = float((sub["GainsCash"] + sub["Bounty"]).sum())
    benef = gains - fees
    roi   = (benef / fees) if fees > 0 else 0.0
    avgpos = float(sub["Position"].mean()) if n else 0.0

    return {
        "n": n,
        "wins": wins,
        "itm": itm,
        "itm_rate": (itm / n if n else 0.0),
        "bubbles": int(bubbles),
        "avg_pos": avgpos,
        "fees": fees,
        "gains": gains,
        "benef": benef,
        "roi": roi,                 # ROI sur gains totaux (cash + bounty), inchangé
        "table": sub.sort_values("start_time"),
    }



# =============================================================================
# Exports JPG (classement)
# =============================================================================

def _hsl_to_rgb_tuple(h: float, s: float, l: float):
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h/360.0, l/100.0, s/100.0)
    return float(r), float(g), float(b)

def classement_df_to_jpg(df: pd.DataFrame, out_path: Path, dpi: int = 220, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    cols = ["Place","Pseudo","Parties","Victoires","ITM","% ITM","Recaves","Recaves en €",
            "Bulles","Buy in","Frais","Gains","Bénéfices"]
    cols = [c for c in cols if c in df.columns]
    data = df[cols].copy()

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

    nrows, ncols = shown.shape
    cell_colours = [["#ffffff"] * ncols for _ in range(nrows)]
    text_colors  = [["#000000"] * ncols for _ in range(nrows)]

    if "Bénéfices" in data.columns:
        vals = pd.to_numeric(data["Bénéfices"].apply(parse_money), errors="coerce").fillna(0.0)
        vmin, vmax = float(vals.min()), float(vals.max())
        span = (vmax - vmin) or 1.0
        j = shown.columns.get_loc("Bénéfices")
        for i, v in enumerate(vals):
            t = (float(v) - vmin) / span
            hue = 120.0 * t
            rgb = _hsl_to_rgb_tuple(hue, 80.0, 78.0)
            cell_colours[i][j] = rgb
            text_colors[i][j]  = "#000000"

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

    base_row_h = 0.4
    fig_h = max(3.0, base_row_h * (nrows + 2))
    fig_w = 1.1 + 0.95 * ncols

    import matplotlib
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=14)

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

    for (r, c), cell in the_table.get_celld().items():
        cell.set_edgecolor("#222")
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_text_props(color=header_text, weight="bold")
        else:
            cell.get_text().set_color(text_colors[r-1][c])

    try:
        w_idx = data.index[data["Pseudo"].str.lower() == "winamax"][0]
        r = w_idx + 1
        left_cell  = the_table[(r, 0)]
        right_cell = the_table[(r, ncols - 1)]
        x0, y0 = left_cell.xy
        x1 = right_cell.xy[0] + right_cell.get_width()
        y  = y0
        ax.add_line(plt.Line2D([x0, x1], [y, y], transform=ax.transAxes,
                               linestyle=(0, (6, 4)), color="#222", linewidth=2.0))
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def classement_points_df_to_jpg(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    wanted = ["Place", "Pseudo", "Parties", "ITM", "Victoires", "Points"]
    cols = [c for c in wanted if c in df.columns] or df.columns.tolist()
    d = df[cols].copy()

    n_rows, n_cols = d.shape
    base_w = 1.8
    extra = 1.2 if "Pseudo" in d.columns else 0.0
    fig_w = max(14.0, base_w * n_cols + extra)
    row_h = 0.62
    head_h = 0.85
    fig_h = max(5.0, head_h + row_h * (n_rows))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.axis("off")

    data = d.astype(str).values.tolist()
    headers = [str(c) for c in d.columns.tolist()]

    tbl = ax.table(cellText=data, colLabels=headers, cellLoc="center", loc="upper left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.35)

    try:
        tbl.auto_set_column_width(list(range(n_cols)))
    except Exception:
        pass

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#e6e6e6")
        cell.set_edgecolor("#222222")
        cell.set_linewidth(1.2)
        cell.set_text_props(weight="bold", color="#000000")

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_edgecolor("#222222")
            cell.set_linewidth(0.8)
            if headers[j].lower() == "pseudo":
                cell.set_facecolor("#f7b329")
                cell.set_text_props(color="#000000", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)


# =============================================================================
# Archive PDF & rollback
# =============================================================================

def archive_pdf(src: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = PDF_DONE / f"{src.stem}__{ts}.pdf"
    PDF_DONE.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest

def rollback_last_import() -> dict:
    log = load_results_log_any()
    if log.empty:
        return {"ok": False, "msg": "Log vide."}
    last_ts = log["processed_at"].max()
    last_ids = log.loc[log["processed_at"] == last_ts, "tournament_id"].unique().tolist()
    if not last_ids:
        return {"ok": False, "msg": "Aucun import récent."}
    last_id = last_ids[-1]

    log2 = log[log["tournament_id"] != last_id].copy()
    log2.to_csv(RESULTS_LOG, index=False, encoding="utf-8")

    back_pdf = ""
    pdfs = sorted(PDF_DONE.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pdfs:
        src = pdfs[0]
        dst = PDF_DIR / (src.name.split("__")[0] + ".pdf")
        shutil.move(str(src), str(dst))
        back_pdf = dst.name
    return {"ok": True, "msg": "Dernier import annulé.", "pdf_back": back_pdf}


# =============================================================================
# Publication snapshot (local + push Git)
# =============================================================================

def _build_public_snapshot_files() -> dict[str, bytes]:
    log = load_results_log_any()
    journal = load_journal_any()

    try:
        classement = standings_from_log(log, season_only=False)
    except Exception:
        classement = pd.DataFrame()

    if not log.empty:
        d1 = log["start_time"].min().date()
        d2 = log["start_time"].max().date()
        points = compute_points_table(log, d1, d2)
    else:
        points = pd.DataFrame(columns=["Place","Pseudo","Parties","ITM","Victoires","Points"])

    files: dict[str, bytes] = {}
    files["results_log.csv"]  = _normalize_results_log(log).to_csv(index=False).encode("utf-8")
    files["journal.csv"]      = _normalize_journal(journal).to_csv(index=False).encode("utf-8")
    files["latest_master.csv"]= classement.to_csv(index=False).encode("utf-8")
    files["points_table.csv"] = points.to_csv(index=False).encode("utf-8")

    # PDFs archivés (miroir)
    for p in PDF_DONE.glob("*.pdf"):
        rel = Path("PDF_Traites") / p.name
        files[str(rel).replace("\\", "/")] = p.read_bytes()

    return files


def _git_run(cmd: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), env=env or os.environ.copy(),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout.strip()
    except Exception as e:
        return 1, f"EXC:{e}"

def _ensure_pub_repo(repo_url: str, token: str, branch: str, repo_dir: Path) -> tuple[bool, str]:
    """
    Prépare le repo de travail dans repo_dir :
      - si déjà cloné : remet l'URL, nettoie d’éventuels extraheaders persistants, fetch/checkout/reset.
      - sinon : clone proprement la branche demandée.
    NB : on n’utilise PAS le token ici (le push ajoute l’Authorization via GIT_HTTP_EXTRAHEADER).
    """
    repo_url = _normalize_repo_url(repo_url)
    repo_dir = Path(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    outlog = []

    def _try_unset_extraheader():
        # On enlève toute config http.extraheader persistante qui pourrait créer des doublons.
        _git_run(["git", "config", "--unset-all", "http.https://github.com/.extraheader"], cwd=repo_dir)
        _git_run(["git", "config", "--unset-all", "http.https://github.com/.extraheader"], cwd=repo_dir.parent)

    if (repo_dir / ".git").exists():
        # Repo déjà présent : resynchronisation propre
        _try_unset_extraheader()

        rc, out = _git_run(["git", "remote", "set-url", "origin", repo_url], cwd=repo_dir)
        outlog.append(f"git remote set-url origin {repo_url} rc={rc}")
        if rc != 0:
            return False, "prepare: remote set-url KO"

        rc, out = _git_run(["git", "fetch", "origin", branch], cwd=repo_dir)
        outlog.append(f"git fetch origin {branch} rc={rc}")
        if rc != 0:
            return False, "prepare: fetch KO"

        rc, out = _git_run(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=repo_dir)
        outlog.append(f"git checkout -B {branch} origin/{branch} rc={rc}")
        if rc != 0:
            return False, "prepare: checkout KO"

        rc, out = _git_run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_dir)
        outlog.append(f"git reset --hard origin/{branch} rc={rc}")
        if rc != 0:
            return False, "prepare: reset KO"

        return True, "prepare: repo prêt"
    else:
        # Pas encore cloné : on (re)clone proprement
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

        rc, out = _git_run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)], cwd=repo_dir.parent)
        outlog.append(f"git clone --branch {branch} {repo_url} {repo_dir} rc={rc}")
        if rc != 0:
            return False, f"prepare: clone KO: {out}"

        _try_unset_extraheader()
        return True, "prepare: repo prêt"




def publish_public_snapshot(push_to_github: bool = False, message: str = "CoronaMax: snapshot") -> tuple[bool, str]:
    """
    1) Génère le snapshot local dans BASE/data/.
    2) Si push_to_github=True, pousse vers le dépôt (branche GIT_BRANCH) en injectant
       l'Authorization *uniquement* pour le 'git push' via GIT_HTTP_EXTRAHEADER.
    """
    files = _build_public_snapshot_files()

    # --- Écriture locale (toujours)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        path = DATA_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)

    if not push_to_github:
        return True, "Snapshot locale générée (aucun push GitHub demandé)."

    repo_url = os.getenv("GIT_PUBLIC_REPO", "").strip()
    token    = os.getenv("GIT_TOKEN", "").strip()
    branch   = os.getenv("GIT_BRANCH", "main").strip()
    author   = os.getenv("GIT_AUTHOR", "CoronaBot <bot@example.com>").strip()
    
    repo_url = _normalize_repo_url(repo_url)

    if not repo_url or not token:
        return False, "Variables GIT_PUBLIC_REPO et/ou GIT_TOKEN manquantes : aucun push GitHub."

    # --- Prépare le repo de travail
    work = BASE / ".pubpush"
    repo = work / "repo"
    work.mkdir(parents=True, exist_ok=True)

    ok, msg = _ensure_pub_repo(repo_url, token, branch, repo)
    outlog = [msg]
    if not ok:
        return False, "\n".join(outlog)

    # --- Copier DATA_DIR -> repo/data
    (repo / "data").mkdir(parents=True, exist_ok=True)
    for p in DATA_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(DATA_DIR)
            dst = (repo / "data" / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

    # --- Identité commit
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"]     = (author.split("<")[0].strip() or "CoronaBot")
    env["GIT_AUTHOR_EMAIL"]    = (author.split("<")[-1].strip(">") or "bot@example.com")
    env["GIT_COMMITTER_NAME"]  = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]

    # --- add / commit
    rc, out = _git_run(["git", "add", "-A"], cwd=repo, env=env)
    outlog.append(f"git add: rc={rc} {out}")
    if rc != 0:
        return False, "\n".join(outlog)

    rc, out = _git_run(["git", "commit", "-m", message], cwd=repo, env=env)
    outlog.append(f"git commit: rc={rc} {out}")
    # rc peut être 1 avec "nothing to commit" -> on continue quand même

    # --- Push avec Authorization injecté *uniquement* via l'env
    import base64
    auth_b64 = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")

    env2 = env.copy()
    env2["GIT_HTTP_EXTRAHEADER"] = f"AUTHORIZATION: Basic {auth_b64}"

    rc, out = _git_run(["git", "push", "-u", "origin", branch], cwd=repo, env=env2)
    outlog.append(f"git push: rc={rc} {out}")
    if rc != 0:
        return False, "\n".join(outlog)

    return True, "\n".join(outlog)
