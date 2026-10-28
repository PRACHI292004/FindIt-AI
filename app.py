"""
app.py - Streamlit interface for LinkdIt AI.

All matching logic lives in matcher.py. This file draws the pages, caches the
slow things (dataset, model, embeddings) and turns errors into friendly messages.

Run locally:   python -m streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st
from supabase import create_client

import matcher

st.set_page_config(
    page_title="LinkdIt AI | Lost and Found Matching",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)



st.markdown("""
<style>

/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700;800&display=swap');

/* ===== GLOBAL FONT ===== */
html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif;
}


/* ===== RICH VIOLET BACKGROUND ===== */
.stApp,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(139, 92, 246, 0.24),
            transparent 35%
        ),
        radial-gradient(
            circle at 95% 45%,
            rgba(91, 75, 210, 0.16),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #DDD5FF 0%,
            #E6E1FF 48%,
            #D8E8FA 100%
        ) !important;
}

/* ===== MAIN CONTENT SPACING ===== */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ===== HEADINGS ===== */
h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
    color: #191A35;
    letter-spacing: -0.6px;
}

h1 {
    font-weight: 800 !important;
}

h2 {
    font-weight: 700 !important;
}

h3 {
    font-weight: 600 !important;
}

/* ===== BUTTONS ===== */
.stButton > button,
.stFormSubmitButton > button {
    font-family: 'Inter', sans-serif;
    border-radius: 12px;
    border: 1px solid #7568F5;
    background: linear-gradient(
        135deg,
        #7568F5,
        #6254DF
    );
    color: white;
    font-weight: 700;
    padding: 0.55rem 1rem;
    transition: all 0.2s ease;
}

/* ===== BUTTON HOVER EFFECT ===== */
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(
        135deg,
        #6254DF,
        #5143C6
    );
    color: white;
    border-color: #5143C6;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(117, 104, 245, 0.25);
}

/* ===== TEXT INPUTS ===== */
.stTextInput input,
.stTextArea textarea,
.stDateInput input {
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
}

/* ===== SELECT BOXES ===== */
.stSelectbox [data-baseweb="select"] {
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}


/* ===== PREMIUM BRAND FONT ===== */
.fi-brand {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -1px;
}

.linked-hero h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800 !important;
    letter-spacing: -1.5px;
}
.fi-brand span {
    color: #7568F5;
}

/* ===== METRIC CARDS ===== */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.88);
    padding: 18px;
    border: 1px solid #E0E3F2;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(30, 35, 80, 0.06);
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #EFEDFF,
        #EAF8F7
    );
}

/* ===== HOME PAGE HERO ===== */
.linked-hero {
    padding: 22px 0;
}

.linked-eyebrow {
    color: #7568F5;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

/* ===== HERO TITLE ===== */
.linked-hero h1 {
    font-family: 'Poppins', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800 !important;
    line-height: 1.05;
    margin-bottom: 12px;
}

.linked-hero h1 span {
    background: linear-gradient(
        90deg,
        #7568F5,
        #27B9A5
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

/* ===== HERO TAGLINE ===== */
.linked-hero h3 {
    color: #FFFFFF;
    font-family: 'Poppins', sans-serif;
    font-size: clamp(1.3rem, 3vw, 2rem);
    font-weight: 700 !important;
    margin-bottom: 14px;
}

/* ===== HERO DESCRIPTION ===== */
.linked-hero p {
    color: #E2E8FF;
    font-family: 'Inter', sans-serif;
    font-size: 1.08rem;
    line-height: 1.8;
    max-width: 650px;
}

/* ===== RESPONSIVE ADJUSTMENTS ===== */
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .linked-hero {
        padding: 12px 0;
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Overall typography and page background */
html, body, [class*="stApp"] {
    font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left,
        rgba(117, 104, 245, 0.12), transparent 45%),
        #F5F6FC;
}

/* Main content spacing */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1, h2, h3 {
    color: #191A35;
    letter-spacing: -0.6px;
}

h1 {
    font-weight: 800 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px;
    border: 1px solid #7568F5;
    background: linear-gradient(135deg, #7568F5, #6254DF);
    color: white;
    font-weight: 700;
    padding: 0.55rem 1rem;
    transition: all 0.2s ease;
}

/* Button hover */
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #6254DF, #5143C6);
    color: white;
    border-color: #5143C6;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(117, 104, 245, 0.25);
}

/* Input fields */
.stTextInput input,
.stTextArea textarea,
.stDateInput input,
.stSelectbox [data-baseweb="select"] {
    border-radius: 10px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
}

/* Navigation brand */
.fi-brand {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* Hero area */
div[data-testid="stVerticalBlock"]:has(
    > div[data-testid="stVerticalBlockBorderWrapper"]
) {
    border-radius: 18px;
}

/* Smooth visual feel */
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #E7E8F3;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(30, 35, 80, 0.04);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F0F0FC;
}


/* ===== LINKDIT AI VISUAL OVERRIDES ===== */

/* Stronger pastel background */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #EDE9FF 0%,
        #F0F5FF 50%,
        #E0F8F2 100%
    ) !important;
}

/* Modern font throughout the app */
.stApp,
.stApp p,
.stApp label,
.stApp input,
.stApp textarea,
.stApp button {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Modern headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    font-family: 'Poppins', 'Segoe UI', sans-serif !important;
    font-weight: 700 !important;
}

/* Make the tagline readable inside the navy hero */
.linked-hero h3 {
    color: #FFFFFF !important;
}

/* Improve hero description contrast */
.linked-hero p {
    color: #E2E8FF !important;
}

/* Give cards a clean, polished appearance */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid #DDDDF5 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(47, 42, 100, 0.08) !important;
}

/* Smooth, rounded buttons */
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
}

/* Subtle navbar background */
div[data-testid="stVerticalBlock"]:has(.fi-brand) {
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)

TAGLINE = "Lost something? Let AI help you find it."
NAV_PAGES = [
    "Home",
    "Find an Item",
    "Report an Item",
    "How It Works",
    "About",
]
EXAMPLE_TEXT = "I lost a black backpack with a blue water bottle and a notebook inside near the library."
PLACEHOLDER = "I lost a black backpack with a blue water bottle and a notebook inside..."
MODE_LIVE = "Today"
MODE_DEMO = "Newest report in the dataset"
ANY = "Any"

DEFAULT_SETTINGS = {
    "days": matcher.DEFAULT_WINDOW_DAYS,
    "top_k": matcher.DEFAULT_TOP_K,
    "ref": MODE_LIVE,
    "strong": matcher.DEFAULT_STRONG_THRESHOLD,
    "possible": matcher.DEFAULT_POSSIBLE_THRESHOLD,
}

# --------------------------------------------------------------------------
# Styling (palette: navy #172554, indigo #4F46E5, teal #14B8A6, off-white #F8FAFC)
# --------------------------------------------------------------------------
CSS = """
<style>
:root { --navy:#172554; --indigo:#4F46E5; --teal:#14B8A6; --bg:#F8FAFC; --card:#FFFFFF;
        --text:#0F172A; --muted:#64748B; --line:#E2E8F0; --indigo-soft:#EEF2FF; --teal-soft:#E6F7F5; }
html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif; }
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"] { display: none !important; }
.stApp { background: var(--bg); color: var(--text); }

[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    min-height: 2.8rem !important;
    line-height: 1.5 !important;
}

[data-testid="stExpander"] summary p {
    margin: 0 !important;
    line-height: 1.5 !important;
}

[data-testid="stNumberInput"],
[data-testid="stSelectbox"],
[data-testid="stRadio"] {
    margin-bottom: 0.75rem !important;
}
.block-container { max-width: 1100px; padding: 1rem 1.25rem 3rem; }
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.01em; }
p, li, .fi-card, .fi-hero, .fi-stat { overflow-wrap: anywhere; }
:focus-visible { outline: 3px solid var(--teal) !important; outline-offset: 2px; }

/* Navigation bar */
.st-key-navbar { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: .55rem 1rem; margin-bottom: 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,.06); }
.fi-brand { display:flex; align-items:center; gap:.6rem; font-weight:700; font-size:1.25rem; color:var(--navy); }
.fi-brand span { color: var(--indigo); }

/* Hero */
.st-key-hero { background: var(--navy); border-radius: 22px; padding: 2rem 2rem 1.6rem;
  margin-bottom: 1.25rem; box-shadow: 0 10px 28px rgba(23,37,84,.22); }
.st-key-hero h1 { color:#fff; font-size: 2.5rem; line-height: 1.15; margin: 0 0 .7rem; }
.st-key-hero p { color:#CBD5E1; font-size:1.08rem; line-height:1.55; margin: 0 0 1.1rem; }
.st-key-hero_find button { background: var(--teal); color: var(--navy); border: none; font-weight: 700; }
.st-key-hero_find button:hover { background:#2DD4BF; color: var(--navy); }
.st-key-hero_how button { background: transparent; color:#fff; border: 1px solid #94A3B8; font-weight: 600; }
.st-key-hero_how button:hover { border-color:#fff; color:#fff; background: rgba(255,255,255,.08); }
.fi-art { width:100%; max-width: 360px; display:block; margin: 0 auto; }

/* Cards and stats */
.fi-grid4 { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap:.8rem; margin:.4rem 0 1.2rem; }
.fi-stat { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:.9rem 1.1rem;
  box-shadow:0 1px 3px rgba(15,23,42,.05); }
.fi-stat-label { color:var(--muted); font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.fi-stat-value { color:var(--navy); font-size:1.7rem; font-weight:700; line-height:1.25; }
.fi-stat-sub { color:var(--muted); font-size:.82rem; }
.fi-steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap:1rem; margin: .5rem 0 1rem; }
.fi-step { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.2rem; box-shadow:0 2px 8px rgba(15,23,42,.06); }
.fi-step h3 { margin:.5rem 0 .3rem; font-size:1.15rem; }
.fi-step p { color:var(--muted); margin:0; line-height:1.5; }
.fi-icon { width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background:var(--indigo-soft); color:var(--indigo); font-weight:700; }
.fi-note { background:var(--teal-soft); border:1px solid #B7E8E2; border-radius:12px; padding:.75rem 1rem; color:#0F3D38; font-size:.93rem; line-height:1.5; }
.fi-warn { background:#FFF7ED; border:1px solid #FED7AA; border-radius:12px; padding:.75rem 1rem; color:#7C2D12; font-size:.93rem; }
.fi-section-title { margin: 1.4rem 0 .2rem; }
.fi-lead { color: var(--muted); margin-bottom: .6rem; }

/* Search panel */
.st-key-search_panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.3rem 1.4rem;
  box-shadow: 0 2px 10px rgba(15,23,42,.06);
  margin-bottom: 1rem;
}

/* Keep widget labels readable and properly spaced */
.st-key-search_panel label {
  position: static !important;
  display: block !important;
  line-height: 1.5 !important;
  margin-bottom: 0.35rem !important;
  transform: none !important;
}

/* Space the advanced options section */
.st-key-search_panel [data-testid="stExpander"] {
  margin-top: 0.75rem;
  margin-bottom: 0.75rem;
}

/* Search button */
[data-testid="stFormSubmitButton"] button {
  width: 100%;
  font-weight: 700;
  border-radius: 12px;
  padding: 0.6rem 1rem;
}

div.stButton > button {
  border-radius: 12px;
  font-weight: 600;
}

/* Results */
.fi-query { background:var(--indigo-soft); border-left:4px solid var(--indigo); border-radius:10px; padding:.8rem 1rem; color:var(--navy); }
.fi-card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.3rem; margin:.8rem 0;
  box-shadow:0 2px 10px rgba(15,23,42,.06); }
.fi-head { display:flex; flex-wrap:wrap; align-items:center; gap:.5rem; margin-bottom:.7rem; }
.fi-rank { background:var(--navy); color:#fff; border-radius:999px; padding:.15rem .75rem; font-weight:600; font-size:.85rem; }
.fi-cat { background:var(--indigo-soft); color:var(--indigo); border-radius:999px; padding:.15rem .75rem; font-weight:600; font-size:.85rem; }
.fi-badge { border-radius:999px; padding:.15rem .75rem; font-weight:700; font-size:.82rem; border:1px solid transparent; }
.fi-badge.strong { background:var(--teal-soft); color:#0F5F57; border-color:#9ADDD5; }
.fi-badge.possible { background:var(--indigo-soft); color:#3730A3; border-color:#C7D2FE; }
.fi-badge.low { background:#F1F5F9; color:#475569; border-color:var(--line); }
.fi-score { margin-left:auto; font-weight:700; color:var(--navy); }
.fi-bar { height:8px; background:#E2E8F0; border-radius:999px; overflow:hidden; margin-bottom:.9rem; }
.fi-fill { height:100%; border-radius:999px; background:var(--indigo); }
.fi-fill.strong { background:var(--teal); }
.fi-fill.low { background:#94A3B8; }
.fi-label { color:var(--muted); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; margin-bottom:.15rem; }
.fi-desc { font-size:1.05rem; line-height:1.5; color:var(--text); margin-bottom:.9rem; }
.fi-meta { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap:.7rem; }
.fi-value { font-weight:600; color:var(--text); }
.fi-none { color:var(--muted); font-style:italic; }
details.fi-contact { margin-top:.9rem; border-top:1px solid var(--line); padding-top:.7rem; }
details.fi-contact summary { cursor:pointer; color:var(--indigo); font-weight:600; }
details.fi-contact .fi-privacy { color:var(--muted); font-size:.85rem; margin:.4rem 0 0; }

/* Footer */
.fi-footer { margin-top:2.2rem; padding:1.3rem 0 0; border-top:1px solid var(--line); color:var(--muted); font-size:.88rem; line-height:1.55; }
.fi-footer b { color:var(--navy); }

@media (max-width: 640px) {
  .st-key-hero { padding:1.3rem 1.1rem; } .st-key-hero h1 { font-size:1.8rem; }
  .fi-score { margin-left:0; } .st-key-search_panel { padding:1rem; }
}
</style>
"""

LOGO_SVG = (
    '<svg width="34" height="34" viewBox="0 0 48 48" role="img" aria-label="LinkdIt AI logo" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="48" height="48" rx="12" fill="#172554"/>'
    '<circle cx="21" cy="21" r="9" fill="none" stroke="#14B8A6" stroke-width="4"/>'
    '<path d="M28 28 L37 37" stroke="#14B8A6" stroke-width="4" stroke-linecap="round"/>'
    '<circle cx="21" cy="21" r="3" fill="#FFFFFF"/></svg>'
)

HERO_ART_SVG = (
    '<svg class="fi-art" viewBox="0 0 360 280" role="img" '
    'aria-label="A magnifying glass over a backpack, illustrating AI-assisted item discovery" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="20" y="20" width="320" height="240" rx="24" fill="#1E3A8A"/>'
    '<path d="M130 80 q50 -50 100 0" fill="none" stroke="#818CF8" stroke-width="10" stroke-linecap="round"/>'
    '<rect x="105" y="78" width="150" height="150" rx="34" fill="#4F46E5"/>'
    '<rect x="135" y="150" width="90" height="56" rx="14" fill="#4338CA"/>'
    '<rect x="150" y="165" width="60" height="8" rx="4" fill="#A5B4FC"/>'
    '<path d="M180 98 v36" stroke="#A5B4FC" stroke-width="6" stroke-linecap="round"/>'
    '<circle cx="232" cy="150" r="44" fill="rgba(20,184,166,0.18)" stroke="#14B8A6" stroke-width="9"/>'
    '<path d="M264 182 L304 222" stroke="#14B8A6" stroke-width="12" stroke-linecap="round"/>'
    '<circle cx="62" cy="64" r="5" fill="#14B8A6"/><circle cx="296" cy="60" r="4" fill="#A5B4FC"/>'
    '<circle cx="300" cy="120" r="3" fill="#14B8A6"/><circle cx="52" cy="206" r="4" fill="#A5B4FC"/>'
    '<path d="M62 64 L100 110 M296 60 L270 104" stroke="#3B4CA8" stroke-width="2" stroke-dasharray="4 5"/></svg>'
)


def get_supabase_client():
    """Create one Supabase client for this Streamlit session."""
    if "_supabase_client" not in st.session_state:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        st.session_state["_supabase_client"] = create_client(url, key)

    return st.session_state["_supabase_client"]

def load_user_reports():
    """Load reports belonging to the currently signed-in user."""
    client = get_supabase_client()
    user_id = current_user()

    if not user_id:
        return []

    response = (
        client.table("item_reports")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    return response.data or []

def current_user():
    """Return the currently authenticated user, if any."""
    return st.session_state.get("findit_user")


def sign_out():
    """Clear the local authentication state."""
    st.session_state.pop("findit_user", None)
    st.session_state.pop("findit_email", None)
    st.session_state.pop("findit_name", None)
    st.session_state.pop("findit_access_token", None)


def page_auth():
    """Display the sign-in and account creation screen."""
    st.title("Welcome to LinkdIt AI")
    st.write("Sign in or create an account to continue.")

    st.markdown("**Lost it. Link it. Find it.**")
    def go_home():
        st.session_state["page"] = "Home"

    if st.button(
        "← Back to Home",
        key="back_to_home",
        on_click=go_home
    ):
        pass

    client = get_supabase_client()
    tab_signin, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_signin:
        with st.form("findit_signin_form"):
            email = st.text_input("Email address", key="signin_email")
            password = st.text_input(
                "Password", type="password", key="signin_password"
            )
            submitted = st.form_submit_button("Sign in")

        if submitted:
            try:
                response = client.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password}
                )
                user = response.user
                if user:
                    st.session_state["findit_user"] = user.id
                    st.session_state["findit_email"] = user.email
                    st.session_state["findit_access_token"] = (
                        response.session.access_token
                        if response.session else None
                    )
                    st.success("Signed in successfully!")
                    st.rerun()
                else:
                    st.error("Sign-in failed. Please check your details.")
            except Exception:
                st.error("Sign-in failed. Check your email and password.")

    with tab_signup:
        with st.form("findit_signup_form"):
            full_name = st.text_input("Full name",key="signup_full_name")
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input(
                "Create password", type="password", key="signup_password"
            )
            confirm_password = st.text_input(
                "Confirm password", type="password", key="signup_confirm"
            )
            create_account = st.form_submit_button("Create account")

        if create_account:
            if not new_email.strip() or not new_password:
                st.error("Please enter your email and password.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            elif len(new_password) < 8:
                st.error("Use a password with at least 8 characters.")
            else:
                try:
                    response = client.auth.sign_up(
    {
        "email": new_email.strip(),
        "password": new_password,
    },
    options={
        "data": {
            "full_name": full_name.strip()
        },
        "email_redirect_to": "https://YOUR-APP-NAME.streamlit.app"
    }
)
                    if response.session and response.user:
                        st.session_state["findit_user"] = response.user.id
                        st.session_state["findit_email"] = response.user.email
                        st.session_state["findit_name"] = (user.user_metadata.get("full_name", "")if user.user_metadata else "")
                        st.session_state["findit_access_token"] = (
                            response.session.access_token
                        )
                        st.success("Account created!")
                        st.rerun()
                    else:
                        st.success(
                            "Account request submitted. Check your email "
                            "for the confirmation link before signing in."
                        )
                except Exception:
                    st.error(
                        "Account creation failed. Check your Supabase "
                        "settings or whether this email is already registered."
                    )

def authentication_gate() -> bool:
    """Show sign-up or sign-in and allow access only after login."""
    try:
        client = get_supabase_client()
    except Exception:
        st.error(
            "Supabase connection could not be initialized. "
            "Check your .streamlit/secrets.toml settings."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.write(
                "Signed in as:",
                st.session_state.get("user_email", "User")
            )
            if st.button("Sign out"):
                try:
                    client.auth.sign_out()
                finally:
                    st.session_state["authenticated"] = False
                    st.session_state.pop("user_email", None)
                    st.rerun()
        return True

    st.title("🔎 LinkdIt AI")
    st.subheader("Secure account access")
    st.write("Sign in or create an account to continue.")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        with st.form("supabase_signin_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )
            login_clicked = st.form_submit_button(
                "Sign In", type="primary"
            )

        if login_clicked:
            if not email.strip() or not password:
                st.warning("Enter your email address and password.")
            else:
                try:
                    response = client.auth.sign_in_with_password({
                        "email": email.strip(),
                        "password": password,
                    })
                    if response.user:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = response.user.email
                        st.rerun()
                    else:
                        st.error("Sign-in was unsuccessful. Please try again.")
                except Exception:
                    st.error(
                        "Sign-in failed. Check your details and confirm "
                        "your email address if required."
                    )

    with tab_signup:
        with st.form("supabase_signup_form"):
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input(
                "Create a password",
                type="password",
                key="signup_password",
                help="Use a strong, unique password."
            )
            confirm_password = st.text_input(
                "Confirm password", type="password"
            )
            signup_clicked = st.form_submit_button(
                "Create Account", type="primary"
            )

        if signup_clicked:
            if not new_email.strip():
                st.warning("Enter your email address.")
            elif len(new_password) < 8:
                st.warning("Use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.warning("The passwords do not match.")
            else:
                try:
                    response = client.auth.sign_up({
                        "email": new_email.strip(),
                        "password": new_password,
                    })
                    if response.session and response.user:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = response.user.email
                        st.rerun()
                    else:
                        st.success(
                            "Account request submitted. Check your email "
                            "for a confirmation link before signing in."
                        )
                except Exception:
                    st.error(
                        "Account creation failed. The email may already "
                        "be registered, or Supabase may have rejected "
                        "the request. Check your Supabase settings."
                    )

    return False

def html(markup: str) -> None:
    """Render raw HTML. Lines are trimmed so Markdown never mistakes indentation for code."""
    compact = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


html(CSS)


# --------------------------------------------------------------------------
# Cached resources: each is built once and reused between interactions.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_dataset(path: str, signature: str):
    """`signature` (a hash of the CSV) is part of the cache key, so editing the file reloads it."""
    return matcher.load_dataset(path)


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    """Loads the SentenceTransformer model once per server process."""
    return matcher.build_sentence_transformer_embedder(model_name)


@st.cache_resource(show_spinner=False)
def get_embeddings(path: str, signature: str, model_name: str):
    """Embeddings of every report. Also cached on disk (see matcher.load_or_compute_embeddings),
    keyed by the descriptions and model name, so changed text or a new model rebuilds them."""
    df, _ = get_dataset(path, signature)
    return matcher.load_or_compute_embeddings(
        df["description_clean"].tolist(), get_embedder(model_name), model_name)


def load_data() -> dict:
    """Load the dataset without the AI model, so every page opens quickly."""
    path = matcher.DEFAULT_DATASET_PATH
    try:
        signature = matcher.file_signature(path)
        df, stats = get_dataset(path, signature)
        return {"ok": True, "path": path, "signature": signature, "df": df, "stats": stats}
    except matcher.DatasetError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception:  # never show a raw traceback to ordinary users
        return {"ok": False, "error": "The dataset could not be loaded because of an unexpected problem."}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def format_when(ts) -> str:
    """Show a time only if the report really has one (this CSV has dates only)."""
    if ts is None:
        return "Unavailable"
    if ts.hour == 0 and ts.minute == 0:
        return ts.strftime("%d %b %Y")
    return ts.strftime("%d %b %Y, %H:%M")


def clean_text(value) -> str:
    """Escape for HTML and keep line breaks harmless."""
    return escape(" ".join(str(value).split()))


def go_to(page: str) -> None:
    st.session_state["page"] = page


def use_example() -> None:
    st.session_state["query_text"] = EXAMPLE_TEXT


def clear_search() -> None:
    st.session_state["query_text"] = ""
    st.session_state.pop("saved", None)


def use_newest_reference() -> None:
    st.session_state["settings"]["ref"] = MODE_DEMO


def restore_page_if_deselected() -> None:
    """The nav control lets a selected option be un-selected; put the page back."""
    if st.session_state.get("page") is None:
        st.session_state["page"] = st.session_state.get("_last_page", "Home")


def reference_time(settings: dict, stats) -> datetime:
    if settings["ref"] == MODE_DEMO:
        return stats.newest.to_pydatetime()
    return datetime.now()


# --------------------------------------------------------------------------
# Reusable page pieces
# --------------------------------------------------------------------------
def render_nav() -> None:
    with st.container(key="navbar"):
        left, middle, right = st.columns([2, 5, 2], vertical_alignment="center")
        with left:
            html(f'<div class="fi-brand">{LOGO_SVG}<div>LinkdIt <span>AI</span></div></div>')
        with middle:
            st.segmented_control(
    "Main navigation",
    NAV_PAGES,
    selection_mode="single",
    key="main_navigation",
    label_visibility="collapsed",
    on_change=lambda: go_to(st.session_state["main_navigation"])
)
        with right:
            user_id = st.session_state.get("findit_user")

            if user_id:
                email = st.session_state.get("findit_email", "")
                display_name = st.session_state.get("findit_name") or (
                    email.split("@")[0] if email else "My Account"
                )

                with st.popover(f"👤 {display_name}"):
                    st.write(f"Signed in as: {email}")

                    if st.button("My Reports", key="nav_my_reports"):
                        go_to("My Reports")

                    if st.button("Sign Out", key="nav_sign_out"):
                        sign_out()
                        st.session_state["page"] = "Home"
                        st.rerun()
            else:
                st.button(
                    "Sign In",
                    type="primary",
                    key="nav_cta",
                    on_click=go_to,
                    args=("Sign In",),
                )


def render_stat_cards(data: dict, settings: dict) -> None:
    if not data["ok"]:
        return
    try:
        now = reference_time(settings, data["stats"])
        summary = matcher.summarize_dataset(data["df"], settings["days"], now)
    except Exception:
        return  # omit the cards rather than show wrong numbers
    cards = [
        ("Usable reports", f'{summary["total"]:,}', "after cleaning the dataset"),
        ("In search window", f'{summary["in_window"]:,}', f'last {settings["days"]:,} days'),
        ("Item categories", f'{len(summary["categories"])}', "found in the data" if summary["categories"] else "not available"),
        ("Latest report", format_when(summary["latest"]), "newest date in the dataset"),
    ]
    html('<div class="fi-grid4">' + "".join(
        f'<div class="fi-stat"><div class="fi-stat-label">{escape(a)}</div>'
        f'<div class="fi-stat-value">{escape(b)}</div><div class="fi-stat-sub">{escape(c)}</div></div>'
        for a, b, c in cards) + "</div>")


def render_steps() -> None:
    steps = [
        ("1", "Describe", "Enter the details of the missing item: colour, type, brand, size, marks and where you last had it."),
        ("2", "Analyze", "AI turns descriptions into semantic embeddings (lists of numbers that capture meaning) and compares their similarity."),
        ("3", "Discover", "Review the highest-ranked candidate reports, then check the details yourself."),
    ]
    html('<div class="fi-steps">' + "".join(
        f'<div class="fi-step"><div class="fi-icon" aria-hidden="true">{n}</div><h3>{t}</h3><p>{d}</p></div>'
        for n, t, d in steps) + "</div>")


def render_footer() -> None:
    html("""
    <div class="fi-footer"><b>LinkdIt AI</b> is a college project that suggests lost-and-found reports with similar
    wording to your description, using SentenceTransformers.<br>
    Similarity scores show how alike two pieces of text are. They do not confirm that an item is yours.<br>
    Please do not submit unnecessary sensitive personal information (ID numbers, passwords, financial details).</div>
    """)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home(data: dict, settings: dict) -> None:
    with st.container(key="hero"):
        text_col, art_col = st.columns([3, 2], vertical_alignment="center")
        
    with text_col:
        html(
    """
    <div class="linked-hero">
        <div class="linked-eyebrow">SMARTER LOST &amp; FOUND</div>
        <h1>LinkdIt <span>AI</span></h1>
        <h3>Lost it. Link it. Find it.</h3>
        <p>
            One smart place to report lost belongings and discover
            found items through intelligent matching.
        </p>
    </div>
    """
)

        b1, b2, _ = st.columns([1.3, 1.3, 1])
        b1.button("Find My Item", key="hero_find", on_click=go_to, args=("Find an Item",))
        b2.button("How It Works", key="hero_how", on_click=go_to, args=("How It Works",))
    with art_col:
        html(HERO_ART_SVG)

    if data["ok"]:
        render_stat_cards(data, settings)
    else:
        st.error(f"**Dataset problem:** {data['error']}")

    html('<h2 class="fi-section-title">How it works in three steps</h2>')
    render_steps()
    html('<div class="fi-note">LinkdIt AI helps you spot reports that may describe your item. It cannot guarantee '
         "recovery or verify ownership; always confirm details with the person who found it.</div>")


def render_results(saved: dict) -> None:
    result = saved["result"]
    settings = saved["settings"]
    st.subheader("Matching results")
    html('<div class="fi-label">Your description</div>'
         f'<div class="fi-query">{clean_text(saved["query"])}</div>')

    if result.status != "ok":
        suggestions = {
            "no_recent_records": "No reports fall inside the search window. Try a longer search period under "
                                 "Advanced options, or count days back from the newest report in the dataset.",
            "no_filtered_records": "Try setting Category and Location to Any, or choose a longer search period.",
            "no_matches": "Try a more detailed description or adjust the filters.",
        }
        st.info(f"{result.message} {suggestions.get(result.status, '')}")
        if result.status == "no_recent_records" and settings["ref"] == MODE_LIVE:
            st.button("Count days back from the newest report instead", on_click=use_newest_reference)
            st.caption("Then press Search for Matches again.")
        return

    count = len(result.matches)
    st.markdown(
        f"**{count} result{'s' if count != 1 else ''}**, ranked from highest to lowest similarity, "
        f"out of {result.eligible_count:,} eligible report(s) from the last {settings['days']:,} days "
        f"(counted back from {format_when(saved['now'])}).")
    html('<div class="fi-note"><b>About the percentage:</b> it is a transformed similarity score, '
         "((cosine similarity + 1) / 2) × 100, that compares the wording and meaning of two descriptions. "
         "It is <b>not</b> the probability that an item is yours or has been found, and unrelated descriptions "
         "can still score well above 0%. Labels are guides only: "
         f"Strong Match is {settings['strong']:g}% or more and Possible Match is {settings['possible']:g}% or more; "
         "labels never hide results.</div>")
    for _, row in result.matches.iterrows():
        html(result_card_html(row, settings["strong"], settings["possible"]))


def result_card_html(row, strong: float, possible: float) -> str:
    score = float(row["similarity_score"])
    label = matcher.score_label(score, strong, possible)
    tone = {"Strong Match": "strong", "Possible Match": "possible", "Low Similarity": "low"}[label]
    description = clean_text(str(row["description"]).strip().rstrip(" ,"))  # display-only tidy-up
    category = clean_text(row["category"]) or "Uncategorised"

    def field(name: str, value: str) -> str:
        shown = escape(value) if value else '<span class="fi-none">Not provided</span>'
        return f'<div><div class="fi-label">{name}</div><div class="fi-value">{shown}</div></div>'

    contact = str(row["contact"]).strip()
    if contact:
        contact_html = (
            '<details class="fi-contact"><summary>Show finder contact details</summary>'
            f'<div class="fi-value" style="margin-top:.4rem">{escape(contact)}</div>'
            '<p class="fi-privacy">Use this only to ask about this report. Ask the finder to describe details only the '
            "owner would know, and avoid sharing sensitive personal information.</p></details>")
    else:
        contact_html = '<div class="fi-none" style="margin-top:.8rem">No contact details were provided for this report.</div>'

    return f"""
    <div class="fi-card">
      <div class="fi-head">
        <span class="fi-rank">Match #{int(row["rank"])}</span>
        <span class="fi-cat">{category}</span>
        <span class="fi-badge {tone}">{label}</span>
        <span class="fi-score">{score:.1f}% similarity score</span>
      </div>
      <div class="fi-bar" role="progressbar" aria-valuenow="{score:.1f}" aria-valuemin="0" aria-valuemax="100"
           aria-label="Similarity score {score:.1f} percent"><div class="fi-fill {tone}" style="width:{score:.1f}%"></div></div>
      <div class="fi-label">Reported item description</div>
      <div class="fi-desc">{description}</div>
      <div class="fi-meta">{field("Location", str(row["location"]).strip())}{field("Date reported", format_when(row["datetime"]))}</div>
      {contact_html}
    </div>"""


def run_search(data: dict, query: str, settings: dict, category: str, location: str) -> None:
    """Load the model + embeddings (first time only), then search."""
    try:
        with st.status("Searching for matches...", expanded=True) as status:
            st.write("Loading the AI model (the first search can take a few minutes while it downloads)...")
            embedder = get_embedder(matcher.DEFAULT_MODEL_NAME)
            st.write("Preparing report embeddings...")
            embeddings = get_embeddings(data["path"], data["signature"], matcher.DEFAULT_MODEL_NAME)
            st.write("Comparing your description with eligible reports...")
            now = reference_time(settings, data["stats"])
            result = matcher.find_matches(
                query, data["df"], embeddings, embedder,
                days=settings["days"], top_k=settings["top_k"], now=now,
                category=None if category == ANY else category,
                location=None if location == ANY else location)
            status.update(label="Search complete", state="complete", expanded=False)
        st.session_state["saved"] = {"query": query.strip(), "result": result, "settings": dict(settings), "now": now}
    except matcher.ModelLoadError as exc:
        st.session_state.pop("saved", None)
        st.error(f"**The AI model could not be loaded.** {exc}")
    except Exception as exc:
        st.session_state.pop("saved", None)
        st.error(f"The search could not be completed ({type(exc).__name__}). Please try again.")


def page_find(data: dict, settings: dict) -> None:
    html('<h1 style="margin-bottom:.2rem">Find an Item</h1>'
         '<div class="fi-lead">Describe what you lost. The more detail, the better the comparison.</div>')
    if not data["ok"]:
        st.error(f"**Dataset problem:** {data['error']}")
        st.caption("Check that data/lost_found_dataset_1000_entries.csv exists, or set FINDIT_DATASET_PATH. See the README.")
        return

    render_stat_cards(data, settings)
    categories = sorted({v for v in data["df"]["category"] if v})
    locations = sorted({v for v in data["df"]["location"] if v})

    with st.container(key="search_panel"):
        html("<h3 style='margin:0 0 .5rem'>Tell us what you lost.</h3>")
        st.button("Try an example", on_click=use_example, key="example_btn")
        with st.form("search_form"):
            st.text_area("Describe your lost item", key="query_text", height=140, max_chars=1000,
                         placeholder=PLACEHOLDER,
                         help="Colour, type, brand, size, distinguishing marks, contents, and where you last had it.")
            col_a, col_b = st.columns(2)
            category = col_a.selectbox("Category (optional)", [ANY] + categories) if categories else ANY
            location = col_b.selectbox("Location (optional)", [ANY] + locations) if locations else ANY
            with st.container():
                st.markdown("### Advanced options")
                days = st.number_input("Search period (days back)", 1, 3650, int(settings["days"]),
                                       help="Only reports from this many days back are compared. Default: 730.")
                top_k = st.number_input("Maximum results", 1, 10, int(settings["top_k"]))
                ref = st.radio("Count days back from", [MODE_LIVE, MODE_DEMO],
                               index=[MODE_LIVE, MODE_DEMO].index(settings["ref"]),
                               help="'Today' uses the current date. The other option treats the newest report in the "
                                    "dataset as today, which is useful for older demo data.")
                strong = st.number_input("'Strong Match' label starts at (%)", 1.0, 100.0, float(settings["strong"]))
                possible = st.number_input("'Possible Match' label starts at (%)", 0.0, 99.0, float(settings["possible"]),
                                           help="Labels are display-only; they never hide results.")
            submitted = st.form_submit_button("Search for Matches", type="primary")

    if submitted:
        query = st.session_state.get("query_text", "")
        if not query.strip():
            st.warning("Please describe the item first, for example its colour, type, and where you lost it.")
        elif not any(ch.isalnum() for ch in query):
            st.warning("Your description needs some words or numbers. Please add details about the item.")
        elif strong <= possible:
            st.error("The 'Strong Match' threshold must be higher than the 'Possible Match' threshold.")
        else:
            settings.update({"days": int(days), "top_k": int(top_k), "ref": ref,
                             "strong": float(strong), "possible": float(possible)})
            run_search(data, query, settings, category, location)

    saved = st.session_state.get("saved")
    if saved:
        render_results(saved)
        st.button("Clear and start a new search", on_click=clear_search, key="clear_btn")
    else:
        html('<div class="fi-note">No search yet. Describe your item above and press <b>Search for Matches</b>. '
             "Your results will appear here.</div>")


def page_how() -> None:
    html('<h1 style="margin-bottom:.2rem">How It Works</h1>'
         '<div class="fi-lead">Three simple steps, powered by semantic similarity.</div>')
    render_steps()
    st.markdown(f"""
**What happens behind the scenes**

1. Your description is cleaned (lower-cased, extra spaces removed), exactly like the stored reports.
2. A **SentenceTransformers** model (`{matcher.DEFAULT_MODEL_NAME}`) converts each description into an *embedding*:
   a list of numbers that captures meaning, so "rucksack" and "backpack" end up close together.
3. **Cosine similarity** measures how closely two embeddings point in the same direction.
4. The reports in the search period are ranked from most to least similar, and the top results are shown.

**Reading the score.** The percentage is `((cosine similarity + 1) / 2) × 100`. It is a transformed similarity
score, not the probability that an item belongs to you, and it does not verify ownership.
""")
    html('<div class="fi-note">LinkdIt AI does not guarantee recovery of any item. Always confirm details with the finder.</div>')


def page_about() -> None:
    html('<h1 style="margin-bottom:.2rem">About LinkdIt AI</h1>'
         f'<div class="fi-lead">{escape(TAGLINE)}</div>')
    st.markdown(f"""
**The problem.** Lost-and-found notice boards and spreadsheets are hard to search. People describe the same
item in different words, so scrolling through lists is slow and easy to get wrong.

**Why semantic matching beats keywords.** A keyword search for "rucksack" misses a report that says "backpack".
Semantic matching compares *meaning*, so reports with different wording can still be found.

**The role of the AI model.** `{matcher.DEFAULT_MODEL_NAME}` (SentenceTransformers) turns text into embeddings.
Similar descriptions get similar embeddings, and cosine similarity ranks them. The model is used as it comes;
it has not been trained on this dataset.

**Limits of similarity scores**
- They compare text only. They cannot see the item, and two different items can be described alike.
- The scale `((cosine + 1) / 2) × 100` makes unrelated text still score above 0%, so compare results with each
  other instead of reading the number as a chance of success.
- Labels such as "Strong Match" use adjustable, hand-picked thresholds and are not validated accuracy measures.
- Duplicated or very generic reports can rank alike.

**Intended use.** A lost-and-found *assistance* tool for a college project demonstration. It suggests
candidates for a person to review; it does not verify ownership or replace contacting the finder.
""")


def page_report() -> None:
    st.header("Report a Lost or Found Item")
    st.write(
        "Submit a report to help connect lost items with the people "
        "who find them. A photo is optional."
    )

    with st.form("report_item_form", clear_on_submit=True):
        report_type = st.radio(
            "What are you reporting?",
            ["Lost Item", "Found Item"],
            horizontal=True,
        )

        category = st.selectbox(
            "Item category",
            [
                "Bag",
                "Electronics",
                "Wallet",
                "Keys",
                "Clothing",
                "Bottle",
                "Other",
            ],
        )

        description = st.text_area(
            "Describe the item",
            placeholder=(
                "Include colour, brand, distinctive marks, "
                "and other useful details."
            ),
            max_chars=1000,
        )

        location = st.text_input(
            "Where was it lost or found?"
        )

        report_date = st.date_input(
            "Date lost or found"
        )

        photo = st.file_uploader(
            "Upload an item photo (optional)",
            type=["jpg", "jpeg", "png"],
        )

        contact = st.text_input(
            "Contact information (keep this private)"
        )

        submitted = st.form_submit_button(
            "Submit Report",
            type="primary",
        )

    if submitted:
        if not description.strip():
            st.error("Please enter an item description.")
        elif not location.strip():
            st.error("Please enter the location.")
        else:
            report = {
                "type": report_type,
                "category": category,
                "description": description.strip(),
                "location": location.strip(),
                "date": report_date.isoformat(),
                "photo_name": photo.name if photo else None,
                "contact": contact.strip(),
            }

            # Temporary storage for testing this form.
            # A database will be needed for permanent reports.
            if "demo_reports" not in st.session_state:
                st.session_state["demo_reports"] = []

            st.session_state["demo_reports"].append(report)

            st.success(
                "Report accepted for this demo. "
                "Permanent storage and AI matching are not connected yet."
            )

    reports = st.session_state.get("demo_reports", [])

    if reports:
        st.subheader("Reports submitted in this session")
        for report in reversed(reports):
            st.markdown(
                f"**{report['type']} — {report['category']}**"
            )
            st.write(f"Description: {report['description']}")
            st.write(f"Location: {report['location']}")
            st.write(f"Date: {report['date']}")
            st.divider()

# --------------------------------------------------------------------------
# App entry
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# App entry
# --------------------------------------------------------------------------
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("query_text", "")
settings = st.session_state.setdefault("settings", dict(DEFAULT_SETTINGS))

current_page = st.session_state.get("page") or "Home"

# Show authentication only when the visitor clicks Sign In.
if current_page == "Sign In":
    if current_user():
        st.session_state["page"] = "Home"
        st.rerun()

    page_auth()
    st.stop()

# Show account information and sign-out for logged-in users.
if current_user():
    with st.sidebar:
        st.write("Signed in as:")
        st.write(st.session_state.get("findit_email", "Authenticated user"))

        if st.button("Sign out"):
            sign_out()
            st.session_state["page"] = "Home"
            st.rerun()

# Display the existing app and its original home page.
st.session_state["_last_page"] = current_page

render_nav()
dataset = load_data()


# Protect search and report pages from users who are not signed in.
if current_page in ("Find an Item", "Report an Item") and not current_user():
    st.warning("Please sign in or create an account to continue.")
    page_auth()
    st.stop()

if current_page == "Find an Item":
    page_find(dataset, settings)
elif current_page == "Report an Item":
    if not current_user():
        st.session_state["page"] = "Sign In"
        st.rerun()
    page_report()
elif current_page == "My Reports":
    if not current_user():
        st.session_state["page"] = "Sign In"
        st.rerun()

    st.header("My Reports")
    reports = load_user_reports()

    if reports:
        st.dataframe(reports, use_container_width=True)
    else:
        st.info("You have not submitted any reports yet.")
elif current_page == "How It Works":
    page_how()
elif current_page == "About":
    page_about()
else:
    page_home(dataset, settings)

render_footer()
