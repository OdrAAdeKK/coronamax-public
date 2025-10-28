# streamlit_app.py – bootstrap minimal, pas de set_page_config ici
import importlib
import streamlit as st
import traceback

def launch():
    try:
        # App.py gère toute la page et l’UI
        importlib.import_module("App")
    except Exception as e:
        st.title("Diagnostic CoronaMax")
        st.error("Erreur au chargement de App.py")
        st.code("".join(traceback.format_exception(e)), language="python")

        # (optionnel) versions utiles
        try:
            import sys, pandas, numpy, altair, PIL, fitz
            st.caption(
                f"Python {sys.version.split()[0]} | "
                f"streamlit {st.__version__} | pandas {pandas.__version__} | "
                f"numpy {numpy.__version__} | altair {altair.__version__} | "
                f"pillow {PIL.__version__} | fitz {fitz.__version__}"
            )
        except Exception:
            pass

if __name__ == "__main__":
    launch()
