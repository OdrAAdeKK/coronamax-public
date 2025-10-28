# streamlit_app.py
# -*- coding: utf-8 -*-
import os, sys, importlib, traceback
import streamlit as st

st.set_page_config(page_title="CoronaMax – Safe loader", page_icon="🛠️", layout="wide")

qs = st.query_params
safe = qs.get("safe", "0") in ("1", "true", "yes")

st.markdown("## CoronaMax — chargeur sécurisé")
st.caption("Ce lanceur n’importe **rien** d’autre que Streamlit au démarrage.")

with st.expander("Environnement (diag rapide)", expanded=True):
    st.write("Python:", sys.version)
    try:
        import importlib.metadata as md
        ver = md.version("streamlit")
    except Exception:
        ver = "?"
    st.write("Streamlit:", ver)
    st.write("CWD:", os.getcwd())
    st.write("Fichiers présents:", sorted(os.listdir("."))[:50])

col1, col2 = st.columns(2)
with col1:
    st.write("Mode sûr :", "✅ activé" if safe else "⛔ désactivé (normal)")
with col2:
    st.caption("Astuce : ajoute `?safe=1` à l’URL pour rester en mode diag.")

st.divider()

def launch_app():
    """
    Charge App.py sous try/except pour voir la stacktrace exacte SI ça plante.
    App.py exécute l’app au moment de l’import, donc on ne l’appelle pas;
    on se contente de l'importer.
    """
    try:
        # Assure-toi qu'on recharge la version du repo, pas un cache
        if "App" in sys.modules:
            del sys.modules["App"]
        importlib.invalidate_caches()

        # 👉 App.py sera exécuté à l'import (comme d'habitude)
        importlib.import_module("App")
        st.success("App importée sans exception. Si rien ne s'affiche, c'est que App.py a déjà rendu l'UI.")
    except Exception:
        st.error("Exception au chargement de App.py :")
        st.exception(traceback.format_exc())

if safe:
    st.warning("Mode sûr actif : App.py ne sera chargé que si tu cliques sur le bouton ci-dessous.")
    if st.button("🚀 Lancer App.py maintenant"):
        launch_app()
else:
    # Mode normal : on tente directement, mais on capture l'erreur pour l'afficher
    launch_app()
