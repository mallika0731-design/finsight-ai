"""
app.py
------
FinSight — MSME Financial Document Intelligence
Main Streamlit entry point.

Run with:
    streamlit run app.py

Or in Colab:
    !streamlit run app.py &
    from pyngrok import ngrok
    print(ngrok.connect(8501))
"""

import streamlit as st

# ── Page config — MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="FinSight",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS (injected once) ────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* App background */
    .stApp, .main { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1e293b; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 12px; }
    [data-testid="stMetricValue"] { color: #e2e8f0; font-size: 22px; font-weight: 600; }

    /* Buttons */
    .stButton > button {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 8px;
        font-size: 12px;
    }
    .stButton > button:hover { border-color: #818cf8; color: #a5b4fc; }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #4f46e5;
        border-color: #4f46e5;
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:hover { background: #6366f1; }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Text */
    h1, h2, h3, h4 { color: #e2e8f0; }
    p, li, .stMarkdown { color: #cbd5e1; }
    code { background: #1e293b; color: #a5b4fc; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── View imports (after page config) ─────────────────────────────────────────
from ui.upload_view import run_upload_view
from ui.qa_view     import run_qa_view
from ui.insight_view import run_insight_view

# ── Session state defaults ────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "Upload"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 💡 FinSight")
    st.markdown("**MSME Financial Intelligence**")
    st.markdown("---")

    doc = st.session_state.get("uploaded_filename")
    if doc:
        st.success(f"📄 {doc}")
    else:
        st.info("No document loaded yet")

    st.markdown("### Navigate")

    nav_items = [
        ("📤 Upload",               "Upload"),
        ("💬 Ask Questions",        "QA"),
        ("📊 Insights & Loan Score","Insights"),
    ]
    for label, page_key in nav_items:
        # Highlight the active page
        is_active = st.session_state["page"] == page_key
        btn_label = f"**{label}**" if is_active else label
        if st.button(btn_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state["page"] = page_key
            st.rerun()

    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Upload a financial PDF\n"
        "2. Ask natural-language questions\n"
        "3. Every answer gets a faithfulness score\n"
        "4. Get an automatic loan eligibility score"
    )
    st.markdown("---")
    st.caption("All models run locally. No data leaves your machine.")

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state.get("page", "Upload")

if page == "Upload":
    run_upload_view()
elif page == "QA":
    run_qa_view()
elif page == "Insights":
    run_insight_view()
else:
    st.error(f"Unknown page: {page}")
    st.session_state["page"] = "Upload"
    st.rerun()
