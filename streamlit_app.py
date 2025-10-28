# streamlit_app.py — bootstrap minimal, pas de set_page_config ici
import importlib
import streamlit as st
import traceback

def launch():
    try:
        # App.py gère toute la config de page et l'UI
        importlib.import_module("App")
    except Exception as e:
        # Pas de set_page_config ici (sinon double appel)
        st.title("Diagnostic CoronaMax")
        st.write("Exception au chargement de App.py :")
        st.exception(e)
        # Optionnel : un petit état des versions utiles
        try:
            import sys, pandas, numpy, altair, PIL, fitz
            st.caption(
                f"Python: {sys.version.split()[0]} | "
                f"streamlit={st.__version__} | pandas={pandas.__version__} | "
                f"numpy={numpy.__version__} | altair={altair.__version__} | "
                f"Pillow={PIL.__version__} | fitz={fitz.__version__}"
            )
        except Exception:
            pass

if __name__ == "__main__":
    launch()
