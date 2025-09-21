from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z0-9_.-]{1,30}):\s+(.*)$")
TIMECODE_LINE_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(?:,\d{3})?\s+-->\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?")
TIMECODE_INLINE_RE = re.compile(r"\[?\(?\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3}|,\d{3})?\)?\]?")
SRT_COUNTER_RE = re.compile(r"^\d{1,6}$")
WEBVTT_HEADER_RE = re.compile(r"^\s*WEBVTT\s*$", re.IGNORECASE)

ACTION_VERBS = {
    "send","create","update","review","schedule","prepare","investigate","draft",
    "write","implement","fix","follow","check","summarize","compile","share","organize",
    "deploy","test","document","plan","set","collect","analyze","refactor","present"
}
ACTION_PATTERNS = [
    "let's ", "lets ", "please ", "need to ", "we need to ", "we should ",
    "make sure", "follow up", "action item", "todo", "to-do"
]
DECISION_PATTERNS = [
    "we decided", "we will go with", "we will use", "we're going to",
    "we chose", "we choose", "we selected", "decision is", "consensus is", "agreed to", "approved"
]

@dataclass
class ExtractResult:
    sentences: List[str]
    summary: List[str]
    action_items: pd.DataFrame
    decisions: List[str]
    cleaned_text: str

def load_spacy() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Run:\n"
            "  python -m spacy download en_core_web_sm"
        ) from e

def clean_transcript(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        l = line.strip()
        if not l:
            continue
        if WEBVTT_HEADER_RE.match(l) or SRT_COUNTER_RE.match(l) or TIMECODE_LINE_RE.match(l):
            continue
        l = TIMECODE_INLINE_RE.sub("", l)
        lines.append(l)
    return "\n".join(lines)

def split_sentences(nlp: spacy.language.Language, txt: str) -> List[Tuple[Optional[str], str]]:
    segments: List[Tuple[Optional[str], str]] = []
    current_speaker: Optional[str] = None
    buffer = []
    for raw_line in txt.splitlines():
        m = SPEAKER_RE.match(raw_line)
        if m:
            if buffer:
                buffer_text = " ".join(buffer).strip()
                if buffer_text:
                    doc = nlp(buffer_text)
                    for s in doc.sents:
                        segments.append((current_speaker, s.text.strip()))
                buffer = []
            current_speaker = m.group(1)
            remainder = m.group(2)
            if remainder:
                buffer.append(remainder)
        else:
            buffer.append(raw_line)
    if buffer:
        buffer_text = " ".join(buffer).strip()
        if buffer_text:
            doc = nlp(buffer_text)
            for s in doc.sents:
                segments.append((current_speaker, s.text.strip()))
    if not segments:
        doc = nlp(txt)
        segments = [(None, s.text.strip()) for s in doc.sents if s.text.strip()]
    seen = set()
    uniq: List[Tuple[Optional[str], str]] = []
    for spk, s in segments:
        key = (spk, s)
        if key not in seen:
            uniq.append((spk, s))
            seen.add(key)
    return uniq

def summarize(sentences: List[str], k: int = 5) -> List[str]:
    if not sentences:
        return []
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    k = max(1, min(k, len(sentences)))
    top_idx = scores.argsort()[::-1][:k]
    top_idx_sorted = sorted(top_idx)
    return [sentences[i] for i in top_idx_sorted]

def is_action_sentence(nlp_doc_sent: spacy.tokens.span.Span) -> bool:
    s = nlp_doc_sent.text.lower()
    if any(p in s for p in ACTION_PATTERNS):
        return True
    first = nlp_doc_sent[0]
    if first.pos_ == "VERB" and first.tag_ == "VB":
        return True
    if s.startswith("please "):
        return True
    return any(t.lemma_ in ACTION_VERBS and t.pos_ == "VERB" for t in nlp_doc_sent)

def extract_action_items(nlp: spacy.language.Language,
                         segments: List[Tuple[Optional[str], str]]) -> pd.DataFrame:
    rows = []
    for speaker, sent in segments:
        doc = nlp(sent)
        if not is_action_sentence(doc):
            continue
        owner = speaker or ""
        due = ""
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        if persons:
            owner = persons[0]
        if dates:
            due = dates[0]
        rows.append({"Owner": owner or "Unassigned", "Action": sent, "Due": due})
    if not rows:
        return pd.DataFrame(columns=["Owner", "Action", "Due"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["Action"])
    return df.reset_index(drop=True)

def extract_decisions(nlp: spacy.language.Language,
                      sentences: List[str]) -> List[str]:
    decisions = []
    for s in sentences:
        s_lower = s.lower()
        if any(p in s_lower for p in DECISION_PATTERNS):
            decisions.append(s)
    for s in sentences:
        if re.search(r"\bwe (?:will|shall|are going to)\b", s.lower()):
            decisions.append(s)
    seen = set()
    out = []
    for s in decisions:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def build_markdown(summary: List[str],
                   actions: pd.DataFrame,
                   decisions: List[str]) -> str:
    md = ["# Meeting Brief", "", "## Summary", ""]
    if summary:
        for s in summary:
            md.append(f"- {s}")
    else:
        md.append("_No summary available._")
    md += ["", "## Action Items", ""]
    if not actions.empty:
        for _, r in actions.iterrows():
            owner = r["Owner"] or "Unassigned"
            due = f" (Due: {r['Due']})" if r["Due"] else ""
            md.append(f"- **{owner}**: {r['Action']}{due}")
    else:
        md.append("_No action items detected._")
    md += ["", "## Decisions", ""]
    if decisions:
        for d in decisions:
            md.append(f"- {d}")
    else:
        md.append("_No decisions detected._")
    return "\n".join(md)

def process(raw_text: str, summary_len: int = 5) -> ExtractResult:
    nlp = load_spacy()
    cleaned = clean_transcript(raw_text)
    seg = split_sentences(nlp, cleaned)
    sentences = [s for _, s in seg]
    summary = summarize(sentences, k=summary_len)
    actions = extract_action_items(nlp, seg)
    decisions = extract_decisions(nlp, sentences)
    return ExtractResult(
        sentences=sentences,
        summary=summary,
        action_items=actions,
        decisions=decisions,
        cleaned_text=cleaned
    )
