# App.py — lanceur sûr + diagnostic
from __future__ import annotations
import os, sys, subprocess, importlib, traceback
import streamlit as st

st.set_page_config(page_title="CoronaMax – bootstrap", layout="wide")

def _get_query_params():
    try:
        # Streamlit récents
        return dict(st.query_params)
    except Exception:
        # Compat anciens
        return {k: v[0] if isinstance(v, list) and v else v
                for k, v in st.experimental_get_query_params().items()}

def diag():
    st.title("🔎 Diagnostic CoronaMax")
    # Python
    st.write("**Python:**", sys.version)
    # Versions clés
    pkgs = ["streamlit", "pandas", "numpy", "altair", "fitz", "PIL", "opencv-python-headless"]
    ok = []
    ko = []
    for p in pkgs:
        try:
            m = importlib.import_module(p if p != "PIL" else "PIL.Image")
            ver = getattr(m, "__version__", getattr(m, "VERSION", "n/a"))
            ok.append((p, str(ver)))
        except Exception as e:
            ko.append((p, repr(e)))
    if ok:
        st.subheader("✅ Imports OK")
        st.table({"package": [x[0] for x in ok], "version": [x[1] for x in ok]})
    if ko:
        st.subheader("❌ Imports en échec")
        st.table({"package": [x[0] for x in ko], "error": [x[1] for x in ko]})

    # pip freeze (utile pour les versions exactes installées)
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30)
        with st.expander("pip freeze", expanded=False):
            st.code(out)
    except Exception as e:
        st.write("pip freeze indisponible:", e)

    st.info("Si un import est en échec ci-dessus, c’est la cause du ‘Oh no.’")

def run_app():
    # On importe ton app réelle (inchangée) : app_main.py
    import app_main  # noqa: F401

def main():
    q = _get_query_params()
    if "safe" in q:
        diag()
        return
    try:
        run_app()
    except Exception as e:
        st.title("💥 Erreur au démarrage")
        st.error("L’application a levé une exception au boot (voir trace ci-dessous).")
        st.exception(e)
        with st.expander("Traceback complet", expanded=True):
            st.code(traceback.format_exc())
        st.link_button("Lancer en mode diagnostic", url="?safe=1", type="secondary")

if __name__ == "__main__":
    main()
