import streamlit as st

# Set page config for wide layout
st.set_page_config(
    page_title="Bias Detection Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Navigation setup
st.sidebar.markdown("<h2 class='sidebar-title'>Bias Detection Dashboard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Navigate through the analysis steps below:")

# Theme switcher in sidebar
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], help="Switch between light and dark themes.")

# Apply theme
if theme == "Dark":
    st.markdown("""
        <style>
            body { background: #1F2937; color: #D1D5DB; }
            .stApp { background: #1F2937; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<script>document.body.classList.add("dark");</script>', unsafe_allow_html=True)
else:
    st.markdown('<script>document.body.classList.remove("dark");</script>', unsafe_allow_html=True)

# Main app navigation handled by Streamlit's multi-page setup
st.markdown("<h1 class='fadeIn'>Welcome to Bias Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Use the sidebar to navigate through the analysis steps. Start by uploading your dataset.")