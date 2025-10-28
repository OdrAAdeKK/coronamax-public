# App.py — wrapper robuste qui affiche toute erreur de boot

import streamlit as st

# On essaye la config très tôt, mais on ignore si elle est déjà faite dans app_core
try:
    st.set_page_config(page_title="CoronaMax", page_icon="🂡", layout="wide")
except Exception:
    pass

st.write("")  # force un premier rendu pour éviter l'écran vide

try:
    # ⚠️ Ton vrai code (celui d'avant) doit vivre dans app_core.py
    import app_core  # exécute tout ton code existant
except Exception as e:
    st.error("🚨 Erreur au démarrage de l’application (trace détaillée ci-dessous)")
    st.exception(e)
    try:
        from pathlib import Path
        Path("boot_error.log").write_text(repr(e))
    except Exception:
        pass
    st.stop()
