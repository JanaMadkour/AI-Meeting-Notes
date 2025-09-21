import streamlit as st
import pandas as pd
from src.meeting_ai import process, build_markdown, load_spacy

st.set_page_config(page_title="AI Meeting Notes & Action Items", layout="wide")
st.title("🧠 AI Meeting Notes & Action Items Extractor")

st.write(
    "Upload a transcript (.txt/.vtt/.srt) or paste text below. "
    "The app will generate a summary, extract action items (owner & due date), and flag decisions."
)

sample = ("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nJana: Let's finalize the API schema by Monday.\n"
          "Alex: I'll draft the PRD today and share it.\n"
          "We decided to use Postgres for v1.\n")

uploaded = st.file_uploader("Upload transcript file", type=["txt", "vtt", "srt"])
text = st.text_area("…or paste transcript text", value=sample, height=200)

colA, colB = st.columns([1,1])
with colA:
    summary_len = st.slider("Summary length (sentences)", min_value=3, max_value=10, value=5)
with colB:
    st.info("Tip: Sentences that sound like requests, todos, or 'we will…' often become action items/decisions.")

run = st.button("Run Extraction")

try:
    _ = load_spacy()
except Exception as e:
    st.error(str(e))
    st.stop()

if run:
    raw = ""
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8", errors="ignore")
    elif text.strip():
        raw = text
    else:
        st.warning("Please upload a file or paste text.")
        st.stop()

    with st.spinner("Analyzing transcript…"):
        result = process(raw, summary_len=summary_len)

    t1, t2, t3, t4 = st.tabs(["Summary", "Action Items", "Decisions", "Full Text"])

    with t1:
        if result.summary:
            for s in result.summary:
                st.markdown(f"- {s}")
        else:
            st.caption("No summary available.")
    with t2:
        if result.action_items.empty:
            st.caption("No action items detected.")
        else:
            st.dataframe(result.action_items, use_container_width=True)
    with t3:
        if result.decisions:
            for d in result.decisions:
                st.markdown(f"- {d}")
        else:
            st.caption("No decisions detected.")
    with t4:
        st.text(result.cleaned_text[:2000])

    md = build_markdown(result.summary, result.action_items, result.decisions)
    st.download_button("⬇️ Download Markdown Brief", md, file_name="meeting_brief.md", mime="text/markdown")
