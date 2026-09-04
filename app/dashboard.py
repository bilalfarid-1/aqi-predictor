import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="Pearls AQI • AeroSense Precision",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu, header, footer, .stDeployButton { display: none !important; visibility: hidden !important; height: 0 !important; }
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    iframe {
        width: 100% !important;
        height: 100vh !important;
        min-height: 1050px !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

curr_dir = Path(__file__).resolve().parent
candidates = [
    curr_dir / "standalone.html",
    curr_dir / "app" / "static" / "standalone.html",
    curr_dir.parent / "standalone.html",
    curr_dir.parent.parent / "standalone.html",
    curr_dir.parent.parent / "app" / "static" / "standalone.html",
]

html_file = None
for c in candidates:
    if c.exists():
        html_file = c
        break

if html_file and html_file.exists():
    html_text = html_file.read_text(encoding="utf-8")
    components.html(html_text, height=1050, scrolling=True)
else:
    st.error("UI bundle not found. Please verify repository files.")
