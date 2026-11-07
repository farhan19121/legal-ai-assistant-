"""Legal tools for the Indian corporate-law study chatbot.

Kept in a separate module (plain functions wrapped with @tool) so they can be
exposed over MCP later with almost no changes.
"""
import json
import os
from datetime import date

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

DB_DIR = os.getenv("LEGAL_DB_DIR", "legal_db")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

_vectorstore = None


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma

        _vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


# ---------------------------------------------------------------------------
# Tool 1: local statute + landmark-case search
# ---------------------------------------------------------------------------
@tool
def search_legal_db(query: str) -> str:
    """Search the local database of Indian statutes (Consumer Protection Act 2019,
    E-Commerce Rules, Companies Act, Contract Act, DPDP Act) and landmark cases.
    Use this FIRST for any legal question. Returns passages with source names and pages."""
    try:
        docs = _get_vectorstore().similarity_search(query, k=6)
    except Exception as exc:
        return json.dumps({"error": f"Legal database unavailable: {exc}. Run ingest.py first."})
    if not docs:
        return json.dumps({"results": [], "note": "No matching passages found."})
    return json.dumps(
        {
            "results": [
                {
                    "source": d.metadata.get("source_name", "unknown"),
                    "page": d.metadata.get("page", 0),
                    "text": d.page_content,
                }
                for d in docs
            ]
        }
    )


# ---------------------------------------------------------------------------
# Tool 2: Indian Kanoon search (only enabled if INDIAN_KANOON_TOKEN is set)
# ---------------------------------------------------------------------------
@tool
def search_indian_kanoon(query: str) -> str:
    """Search Indian court judgments on Indian Kanoon. Returns title, court and URL
    for the top results. Use to find case law not present in the local database."""
    token = os.getenv("INDIAN_KANOON_TOKEN")
    if not token:
        return json.dumps({"error": "INDIAN_KANOON_TOKEN not set."})
    try:
        r = requests.post(
            "https://api.indiankanoon.org/search/",
            headers={"Authorization": f"Token {token}"},
            data={"formInput": query, "pagenum": 0},
            timeout=20,
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])[:5]
    except Exception as exc:
        return json.dumps({"error": f"Indian Kanoon request failed: {exc}"})
    return json.dumps(
        {
            "results": [
                {
                    "title": d.get("title"),
                    "court": d.get("docsource"),
                    "url": f"https://indiankanoon.org/doc/{d.get('tid')}/",
                    "snippet": d.get("headline", ""),
                }
                for d in docs
            ]
        }
    )


# ---------------------------------------------------------------------------
# Tool 3: consumer forum jurisdiction + limitation calculator
# ---------------------------------------------------------------------------
@tool
def consumer_forum_calculator(consideration_paid: float, cause_of_action_date: str) -> str:
    """Find which consumer commission has pecuniary jurisdiction (based on the value of
    goods/services PAID as consideration, in rupees) and check the 2-year limitation
    period under Section 69, Consumer Protection Act 2019.
    cause_of_action_date format: YYYY-MM-DD."""
    # VERIFY against the current Consumer Protection (Jurisdiction of the District
    # Commission, the State Commission and the National Commission) Rules, 2021.
    if consideration_paid <= 50_00_000:
        forum = "District Commission"
    elif consideration_paid <= 2_00_00_000:
        forum = "State Commission"
    else:
        forum = "National Commission (NCDRC)"

    try:
        d = date.fromisoformat(cause_of_action_date)
    except ValueError:
        return json.dumps({"error": "Date must be in YYYY-MM-DD format."})
    try:
        deadline = d.replace(year=d.year + 2)
    except ValueError:  # 29 Feb
        deadline = d.replace(year=d.year + 2, day=28)

    return json.dumps(
        {
            "forum": forum,
            "consideration_paid": consideration_paid,
            "limitation_deadline": str(deadline),
            "within_limitation": date.today() <= deadline,
            "note": (
                "Section 69, CPA 2019: complaint within 2 years from the date the cause of "
                "action arose; delay may be condoned for sufficient cause. Pecuniary limits "
                "are per the 2021 Jurisdiction Rules and must be verified against the latest "
                "notification."
            ),
        }
    )


def get_legal_tools():
    tools = [search_legal_db, consumer_forum_calculator]
    if os.getenv("INDIAN_KANOON_TOKEN"):
        tools.append(search_indian_kanoon)
    return tools
