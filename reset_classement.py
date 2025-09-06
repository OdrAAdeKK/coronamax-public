# reset_classement.py
from datetime import datetime
from pathlib import Path
import pandas as pd

from app_classement_unique import (
    PDF_DIR, PDF_DONE, ARCHIVE, F_MASTER, RESULTS_LOG, JOURNAL_CSV,
    load_master_df, save_master_df
)

def list_files_sorted(folder: Path, pattern="*.pdf"):
    return sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (ARCHIVE / "BACKUPS").mkdir(parents=True, exist_ok=True)

    # backups
    if Path(RESULTS_LOG).exists():
        (ARCHIVE / "BACKUPS" / f"results_log_{ts}.csv").write_bytes(Path(RESULTS_LOG).read_bytes())
    if Path(JOURNAL_CSV).exists():
        (ARCHIVE / "BACKUPS" / f"journal_{ts}.csv").write_bytes(Path(JOURNAL_CSV).read_bytes())
    if Path(F_MASTER).exists():
        (ARCHIVE / "BACKUPS" / f"MASTER_{ts}.csv").write_bytes(Path(F_MASTER).read_bytes())

    # master vide
    save_master_df(pd.DataFrame(columns=load_master_df().columns))

    # vider journal / results_log
    Path(JOURNAL_CSV).write_text("sha1,filename,processed_at\n", encoding="utf-8")
    Path(RESULTS_LOG).write_text("tournament_id,tournament_name,start_time,buyin_total,is_KO,Pseudo,Position,GainsCash,Bounty,Reentry,processed_at\n", encoding="utf-8")

    # remettre les PDFs archivés dans PDF_A_TRAITER (enlevant l’horodatage)
    for p in list_files_sorted(PDF_DONE, "*.pdf"):
        name = p.name.split("__")[0] + ".pdf" if "__" in p.name else p.name
        dst = PDF_DIR / name
        dst.write_bytes(p.read_bytes())
        p.unlink(missing_ok=True)

    print("Reset terminé. Ouvre l’app Streamlit et va dans l’onglet Importer pour rejouer les tournois.")

if __name__ == "__main__":
    main()
