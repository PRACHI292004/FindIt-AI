"""
app.py - Streamlit interface for FindIt AI.

All matching logic lives in matcher.py. This file only draws the pages,
caches the slow things (model, dataset, embeddings) and shows friendly messages.

Run:  streamlit run app.py
"""

from datetime import datetime
from html import escape

import streamlit as st

import matcher

st.set_page_config(page_title="FindIt AI", page_icon="🔎", layout="wide")

TAGLINE = "Lost something? Let AI help you find it."
EXAMPLE_TEXT = "I lost my black backpack with a laptop compartment near the library."
PAGES = ["Find an Item", "How It Works", "About the Project"]
MODE_LIVE = "Today (live)"
MODE_DEMO = "Newest record in dataset (demo)"
ANY = "Any"

# --------------------------------------------------------------------------
# Styling: white background, navy text, blue accents, rounded cards
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
:root { --navy:#0B2545; --blue:#1D4ED8; --soft:#EFF4FF; --line:#D9E3F5; --muted:#51627E; }
html, body, [class*="st-"], .stMarkdown { font-family: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { max-width: 960px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { color: var(--navy); letter-spacing: -0.01em; }
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
.fi-brand { font-size: 1.5rem; font-weight: 700; color: var(--navy); margin-bottom: .1rem; }
.fi-brand span { color: var(--blue); }
.fi-sub { color: var(--muted); font-size: .9rem; margin-bottom: 1rem; }
.fi-hero { background: linear-gradient(135deg, #0B2545 0%, #1D4ED8 100%); color: #fff; border-radius: 20px;
           padding: 2rem 2rem 1.7rem; margin-bottom: 1.4rem; box-shadow: 0 8px 24px rgba(11,37,69,.18); }
.fi-hero h1 { color: #fff; margin: 0 0 .3rem; font-size: 2.3rem; }
.fi-hero p { margin: 0; font-size: 1.05rem; color: #DCE7FF; line-height: 1.5; }
.fi-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem;
           margin:.8rem 0; box-shadow: 0 2px 10px rgba(11,37,69,.07); }
.fi-head { display:flex; flex-wrap:wrap; align-items:center; gap:.6rem; margin-bottom:.6rem; }
.fi-rank { background: var(--navy); color:#fff; border-radius:999px; padding:.1rem .65rem; font-weight:600; font-size:.85rem; }
.fi-cat { background: var(--soft); color: var(--blue); border-radius:999px; padding:.1rem .7rem; font-weight:600; font-size:.85rem; }
.fi-score { margin-left:auto; color: var(--navy); font-weight:700; }
.fi-bar { height:10px; background:#E6EDFA; border-radius:999px; overflow:hidden; margin-bottom:.9rem; }
.fi-fill { height:100%; background: var(--blue); border-radius:999px; }
.fi-label { color: var(--muted); font-size:.8rem; margin-bottom:.15rem; }
.fi-desc { color: var(--navy); font-size:1.02rem; margin-bottom:.9rem; line-height:1.45; }
.fi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:.7rem; }
.fi-value { color: var(--navy); font-weight:600; overflow-wrap:anywhere; }
.fi-none { color: var(--muted); font-style: italic; font-weight:400; }
.fi-query { background: var(--soft); border-left: 4px solid var(--blue); border-radius: 10px; padding:.8rem 1rem; color: var(--navy); }
.fi-note { background:#F7F9FD; border:1px dashed var(--line); border-radius:12px; padding:.7rem 1rem; color: var(--muted); font-size:.92rem; }
.fi-step { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.9rem 1.1rem; margin:.55rem 0; box-shadow:0 2px 8px rgba(11,37,69,.05); }
.fi-step b { color: var(--navy); }
.fi-num { display:inline-block; background:var(--blue); color:#fff; border-radius:50%; width:1.7rem; height:1.7rem; text-align:center; line-height:1.7rem; margin-right:.6rem; font-weight:600; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius: 12px; font-weight: 600; }
@media (max-width: 640px) { .fi-hero h1 { font-size: 1.7rem; } .fi-hero { padding: 1.3rem; } .fi-score { margin-left: 0; } }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources: the model loads once, the index is rebuilt only if the
# dataset file (or model name) changes.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    return matcher.build_sentence_transformer_embedder(model_name)


@st.cache_resource(show_spinner=False)
def get_index(path: str, signature: str, model_name: str):
    """`signature` (hash of the CSV) is part of the cache key, so editing the CSV
    triggers a rebuild."""
    df, stats = matcher.load_dataset(path)
    embedder = get_embedder(model_name)
    embeddings = matcher.load_or_compute_embeddings(df["description_clean"].tolist(), embedder, model_name)
    return df, embeddings, stats


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def format_when(ts) -> str:
    """Show the time only if the record actually has one (this CSV has dates only)."""
    if ts.hour == 0 and ts.minute == 0:
        return ts.strftime("%d %b %Y")
    return ts.strftime("%d %b %Y, %H:%M")


def field_html(label: str, value: str) -> str:
    shown = escape(value) if value else '<span class="fi-none">Not provided</span>'
    return f'<div><div class="fi-label">{label}</div><div class="fi-value">{shown}</div></div>'


def result_card_html(row) -> str:
    score = float(row["similarity_score"])
    description = escape(str(row["description"]).strip().rstrip(" ,"))  # display-only tidy-up
    return f"""
<div class="fi-card">
  <div class="fi-head">
    <span class="fi-rank">#{int(row["rank"])}</span>
    <span class="fi-cat">{escape(row["category"]) or "Uncategorised"}</span>
    <span class="fi-score">Similarity {score:.1f} / 100</span>
  </div>
  <div class="fi-bar" role="progressbar" aria-valuenow="{score:.1f}" aria-valuemin="0" aria-valuemax="100"
       aria-label="Text similarity score"><div class="fi-fill" style="width:{score:.1f}%"></div></div>
  <div class="fi-label">Reported item description (from the found-items records)</div>
  <div class="fi-desc">{description}</div>
  <div class="fi-grid">
    {field_html("Location", row["location"])}
    {field_html("Date found", format_when(row["datetime"]))}
    {field_html("Contact", row["contact"])}
  </div>
</div>"""


def switch_to_demo_mode():
    st.session_state["ref_mode"] = MODE_DEMO


def use_example():
    st.session_state["query_text"] = EXAMPLE_TEXT


# --------------------------------------------------------------------------
# Sidebar: navigation + settings
# --------------------------------------------------------------------------
st.session_state.setdefault("query_text", "")
st.session_state.setdefault("ref_mode", MODE_LIVE)

with st.sidebar:
    st.markdown('<div class="fi-brand">FindIt <span>AI</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fi-sub">{TAGLINE}</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    with st.expander("Search settings"):
        window_days = st.number_input("Search window (days)", 1, 3650, matcher.DEFAULT_WINDOW_DAYS,
                                      help="Only reports from this many days back are compared. Default: 730.")
        top_k = st.number_input("Maximum results", 1, 10, matcher.DEFAULT_TOP_K)
        ref_mode = st.radio(
            "Count days back from", [MODE_LIVE, MODE_DEMO], key="ref_mode",
            help="Live mode uses today's date, as the original notebook did. Demo mode treats the "
                 "newest record in the CSV as 'today' so an older dataset can still be searched.")
        dataset_path = st.text_input("Dataset CSV path", matcher.DEFAULT_DATASET_PATH)


# --------------------------------------------------------------------------
# Page: Find an Item
# --------------------------------------------------------------------------
def render_results(saved: dict):
    result = saved["result"]
    st.subheader("Matching results")
    st.markdown("**Your description**")
    st.markdown(f'<div class="fi-query">{escape(saved["query"])}</div>', unsafe_allow_html=True)

    if result.status != "ok":
        st.info(result.message)
        return

    count = len(result.matches)
    st.markdown(
        f"**{count} result{'s' if count != 1 else ''} returned**, ranked from highest to lowest "
        f"similarity, out of {result.eligible_count} eligible report(s) "
        f"from the {saved['days']}-day window ending {format_when(saved['now'])}."
    )
    st.markdown(
        '<div class="fi-note"><b>How to read the score:</b> it measures how similar the <i>wording and meaning</i> '
        "of two descriptions are (0 to 100). It is not the probability that an item is yours, and a high score "
        "does not confirm a match. Check the details and contact the finder to verify ownership.</div>",
        unsafe_allow_html=True,
    )
    for _, row in result.matches.iterrows():
        st.markdown(result_card_html(row), unsafe_allow_html=True)


def page_find():
    st.markdown(
        f'<div class="fi-hero"><h1>FindIt AI</h1><p><b>{TAGLINE}</b><br>'
        "Describe your lost item in your own words. FindIt AI compares your description with recently "
        "reported found items by meaning, not just matching keywords, and shows the closest reports.</p></div>",
        unsafe_allow_html=True,
    )

    # Load (or reuse) the model + dataset index.
    try:
        signature = matcher.file_signature(dataset_path)
        with st.spinner("Preparing the AI model and found-item records (the first run can take a few minutes)..."):
            df, embeddings, stats = get_index(dataset_path, signature, matcher.DEFAULT_MODEL_NAME)
    except matcher.DatasetError as exc:
        st.error(f"**Dataset problem:** {exc}")
        st.stop()
    except matcher.ModelLoadError as exc:
        st.error(f"**Model problem:** {exc}")
        st.stop()
    except Exception as exc:  # last-resort guard so the app never shows a raw traceback
        st.error(f"Something unexpected went wrong while preparing the app: {exc}")
        st.stop()

    if stats.unreadable_dates or stats.empty_descriptions:
        st.caption(f"Skipped {stats.unreadable_dates} record(s) with unreadable dates and "
                   f"{stats.empty_descriptions} with empty descriptions.")

    # The date the window counts back from.
    if ref_mode == MODE_DEMO:
        now = stats.newest.to_pydatetime()
        st.info(f"Demo mode: {format_when(stats.newest)} (the newest record in the dataset) is treated as today.")
    else:
        now = datetime.now()

    categories = sorted({c for c in df["category"] if c})
    locations = sorted({l for l in df["location"] if l})

    st.markdown("### Describe your lost item")
    st.caption(f"Example: {EXAMPLE_TEXT}")
    st.button("Use the example", on_click=use_example)

    with st.form("search_form"):
        st.text_area("Item description", key="query_text", height=130,
                     placeholder="Colour, type, brand, size, distinguishing marks, where you last had it...")
        col_a, col_b = st.columns(2)
        category = col_a.selectbox("Category (optional)", [ANY] + categories) if categories else ANY
        location = col_b.selectbox("Location (optional)", [ANY] + locations) if locations else ANY
        submitted = st.form_submit_button("Find Matching Items", type="primary", use_container_width=True)

    if submitted:
        query = st.session_state["query_text"]
        if not query.strip():
            st.warning("Please describe the item first, for example its colour, type and where you lost it.")
        else:
            try:
                with st.spinner("Searching..."):
                    result = matcher.find_matches(
                        query, df, embeddings, get_embedder(matcher.DEFAULT_MODEL_NAME),
                        days=int(window_days), top_k=int(top_k), now=now,
                        category=None if category == ANY else category,
                        location=None if location == ANY else location)
                st.session_state["saved"] = {"query": query.strip(), "result": result,
                                             "days": int(window_days), "now": now}
            except Exception as exc:
                st.session_state.pop("saved", None)
                st.error(f"The search failed: {exc}")

    saved = st.session_state.get("saved")
    if saved:
        render_results(saved)
        result = saved["result"]
        if result.status == "no_recent_records":
            st.markdown(
                f"The dataset's reports run from **{format_when(stats.oldest)}** to **{format_when(stats.newest)}**, "
                "so none fall inside the current window. Nothing has been made up to fill the gap.")
            if ref_mode == MODE_LIVE:
                st.button("Switch to demo mode (count back from the newest record)", on_click=switch_to_demo_mode)
                st.caption("Then press Find Matching Items again.")
    else:
        st.markdown('<div class="fi-note">No search yet. Describe your item above and press '
                    "<b>Find Matching Items</b> to see the closest reports.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page: How It Works
# --------------------------------------------------------------------------
def page_how():
    st.title("How It Works")
    st.write("FindIt AI follows the same five steps every time you search.")
    steps = [
        ("You describe a lost item", "Type what you remember: colour, type, brand, size, marks, where you lost it."),
        ("The system preprocesses the description",
         "Your text is lower-cased and extra spaces are removed. The stored descriptions are cleaned in exactly the same way."),
        ("SentenceTransformer creates text embeddings",
         "The all-mpnet-base-v2 model turns each description into a list of numbers that captures its meaning, so "
         "\"rucksack\" and \"backpack\" end up close together."),
        ("Cosine similarity compares your description with eligible records",
         "Only reports from the recent window (730 days by default) are compared. Similarity is converted to a 0 to 100 scale."),
        ("The system ranks and displays the most similar records",
         "The top three are shown from highest to lowest score, with their location, date and contact details."),
    ]
    for number, (title, text) in enumerate(steps, start=1):
        st.markdown(f'<div class="fi-step"><span class="fi-num">{number}</span><b>{title}</b><br>{text}</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="fi-note">Scores come from text similarity only. They are a ranking aid, '
                "not proof of ownership.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page: About
# --------------------------------------------------------------------------
def page_about():
    st.title("About the Project")
    st.markdown("""
### The problem
Lost-and-found desks and notice boards rely on people scanning long lists and guessing which wording
matches their item. People describe the same thing differently ("rucksack", "backpack", "bag with a laptop
sleeve"), so plain keyword search misses good candidates.

### The AI technique
FindIt AI uses **semantic text similarity**. A SentenceTransformer model (*all-mpnet-base-v2*) converts each
description into an embedding vector. Descriptions with similar meaning have vectors pointing in similar
directions, which **cosine similarity** measures. The score shown is `((cosine + 1) / 2) × 100`.

### Why semantic similarity?
It lets a short, informal description find records written in different words, without hand-written rules.

### Current limitations
- It compares **text only**. There is no image recognition, and it cannot tell two similar items apart
  unless the descriptions differ.
- A high score is not a confirmed match and is not a probability of ownership. Ownership must be verified by a person.
- The score scale is `(cosine + 1) / 2`, so unrelated descriptions usually still score well above 0. Compare
  scores relative to each other rather than treating them as percentages.
- Only reports inside the date window are searched (730 days by default), so older reports are never shown.
- Quality depends on the descriptions in the dataset. Duplicate or very generic descriptions will rank similarly.
- Colours, sizes and brands are understood only as part of the text, not as exact filters.
""")


{"Find an Item": page_find, "How It Works": page_how, "About the Project": page_about}[page]()
