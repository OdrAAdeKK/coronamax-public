# streamlit_app.py
# Exécute App.py sous try/except à CHAQUE run (et pas seulement à l'import)

import streamlit as st
import runpy
import sys

def show_diag(e):
    st.title("Diagnostic CoronaMax")
    st.write("Exception au chargement/exécution de **App.py** :")
    st.exception(e)
    # Petit état des libs utiles
    try:
        import pandas, numpy, altair, PIL, fitz, streamlit as stlib
        st.caption(
            f"Python {sys.version.split()[0]} | "
            f"streamlit {stlib.__version__} | "
            f"pandas {pandas.__version__} | "
            f"numpy {numpy.__version__} | "
            f"altair {altair.__version__} | "
            f"pillow {PIL.__version__} | "
            f"fitz {fitz.__version__}"
        )
    except Exception:
        pass

def launch():
    try:
        # Exécute le module comme un script, ainsi toute exception de page est capturée
        runpy.run_module("App", run_name="__main__", alter_sys=True)
    except Exception as e:
        show_diag(e)

if __name__ == "__main__":
    launch()
