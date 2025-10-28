# App.py — Wrapper robuste + SAFE MODE pour diagnostiquer le boot Cloud
from __future__ import annotations
import os, sys, importlib, subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

# Page config (ignore si déjà appelée ailleurs)
try:
    st.set_page_config(page_title="CoronaMax", page_icon="🂡", layout="wide")
except Exception:
    pass

# Lecture des query params compatible anciennes/nouvelles versions
try:
    qp = st.query_params  # ≥1.29
    get_q = lambda k, d=None: qp.get(k, [d])[0] if isinstance(qp.get(k), list) else qp.get(k, d)
except Exception:
    qp = st.experimental_get_query_params()  # legacy
    get_q = lambda k, d=None: (qp.get(k, [d]) or [d])[0]

SAFE = str(get_q("safe", "0")).lower() in ("1", "true", "yes")

def _env_table():
    rows = []
    for name in ["streamlit","pandas","numpy","altair","Pillow","matplotlib","pdfplumber","PyPDF2"]:
        try:
            m = importlib.import_module(name)
            rows.append((name, "OK", getattr(m, "__version__", "?")))
        except Exception as e:
            rows.append((name, "ERROR", str(e)))
    df = pd.DataFrame(rows, columns=["package","status","version/error"])
    return df

if SAFE:
    st.title("CoronaMax — Safe mode")
    st.caption("Ce mode évite d’importer l’app principale, vérifie l’environnement et permet de lancer app_core à la demande.")

    base = Path(__file__).resolve().parent
    st.write("BASE:", str(base))
    try:
        listing = [p.as_posix() for p in base.iterdir()]
        st.write("Listing racine :", listing[:200])
    except Exception as e:
        st.warning(f"Listing KO: {e}")

    st.subheader("Packages")
    st.dataframe(_env_table(), use_container_width=True, hide_index=True)

    with st.expander("pip freeze"):
        try:
            txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30)
            st.code(txt)
        except Exception as e:
            st.write(f"pip freeze KO: {e}")

    st.divider()
    if st.button("Lancer l’app principale (import app_core)"):
        try:
            import app_core  # ton App.py d’origine déplacé ici
        except Exception as e:
            st.error("🚨 Exception au démarrage de app_core :")
            st.exception(e)
            st.stop()

    st.info("Pour quitter le Safe mode, enlève '?safe=1' de l’URL.")
else:
    # Chemin normal : on essaye d’exécuter l’app métier
    try:
        import app_core
    except Exception as e:
        st.error("🚨 Erreur au démarrage de l’application (stack ci-dessous)")
        st.exception(e)
        st.stop()
