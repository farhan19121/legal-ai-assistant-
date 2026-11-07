# AI Legal Study Assistant (India)

A Streamlit + LangGraph chatbot (Gemini) that helps junior corporate lawyers study
Indian consumer-protection and corporate-law cases. Answers are grounded in retrieved
sources and checked by a citation verifier.

**Study aid only. Not legal advice. Verify all citations and the current version of the law.**

## Features
- Research, Study (Socratic) and Draft modes
- Local legal database search (Chroma + Gemini embeddings)
- Consumer forum jurisdiction and limitation calculator
- Optional Indian Kanoon search (needs a token)
- Citation verifier with a verified badge and a sources panel

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
INDIAN_KANOON_TOKEN=optional
```

## Add legal documents
Put text PDFs or .txt files of statutes and judgments in `data/` (not included in this repo), then:
```bash
python ingest.py
```

## Run
```bash
python -m streamlit run frontend.py
```

## Files
- `frontend.py` - Streamlit UI
- `chatbot_backend.py` - LangGraph graph (agent, tools, verifier)
- `legal_tools.py` - legal search tools and calculator
- `ingest.py` - builds the vector database
