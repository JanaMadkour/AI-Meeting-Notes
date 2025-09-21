# AI Meeting Notes & Action Items Extractor

Turn meeting transcripts into a concise **summary**, **action list** (owner & due), and **decision log**. Built with Python, spaCy, scikit-learn, and Streamlit.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py


## 6) PRD.md
```bash
cat > PRD.md << 'EOF'
# PRD — AI Meeting Notes & Action Items

## Vision
Help teams leave every meeting with clarity: what was decided, who owns what, and by when.

## Users
Students (group projects), early career PMs, interns, engineers.

## MVP Goals
- Ingest transcript (.txt/.vtt/.srt)
- Generate concise summary
- Extract action items (owner, due date)
- List explicit decisions
- Export 1-page brief (Markdown)

## Success Metrics
- Task success: user can list next steps in < 1 minute
- Precision@5 for action items ≥ 0.75 (on small labeled set)
- Time-to-brief ≤ 10s for 5–10 min transcripts

## Non-Goals (MVP)
Perfect diarization; live capture; enterprise auth; LLM fine-tuning.

## Future
Outlook/Teams integration; PDF export; optional LLM rewrite for clarity.
