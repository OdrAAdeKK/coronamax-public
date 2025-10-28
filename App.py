# App.py — wrapper de démarrage robuste (ne touche pas à ton code métier)

import streamlit as st

# Important : config très tôt (sinon pas de page si ça crashe)
st.set_page_config(page_title="CoronaMax", page_icon="🂡", layout="wide")
st.write("")  # force un premier rendu

try:
    # ⚠️ Ton VRAI code vit maintenant dans app_core.py
    import app_core  # exécute tout le code existant (ex-App.py)
except Exception as e:
    st.error("🚨 Erreur au démarrage de l’application (trace détaillée ci-dessous)")
    st.exception(e)
    # journalisation simple pour post-mortem
    try:
        from pathlib import Path
        Path("boot_error.log").write_text(str(e))
    except Exception:
        pass
    st.stop()
