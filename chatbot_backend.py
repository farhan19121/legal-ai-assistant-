import asyncio
import json
import os
import threading
from typing import Annotated, Optional, TypedDict

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from legal_tools import get_legal_tools

load_dotenv()

# ------------------------------------------------------------------
# 0. Dedicated async loop for backend tasks (unchanged from your version)
# ------------------------------------------------------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# ------------------------------------------------------------------
# 1. LLM (Gemini)
# ------------------------------------------------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
verifier_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

# ------------------------------------------------------------------
# 2. Tools (see legal_tools.py)
# ------------------------------------------------------------------
tools = get_legal_tools()

# Optional MCP tools. Leave empty to skip.
MCP_SERVERS: dict = {}
if MCP_SERVERS:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        tools += run_async(MultiServerMCPClient(MCP_SERVERS).get_tools())
    except Exception as exc:
        print(f"[MCP] Failed to load tools: {exc!r}")

llm_with_tools = llm.bind_tools(tools)

# ------------------------------------------------------------------
# 3. State
# ------------------------------------------------------------------
class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: str                      # research | study | draft
    sources: list[dict]            # shown in the UI "Sources used" panel
    verified: Optional[bool]       # True / False / None (not checked)
    unsupported: list[str]         # claims the verifier couldn't find in sources


# ------------------------------------------------------------------
# 4. Prompts
# ------------------------------------------------------------------
BASE_PROMPT = """You are a legal study assistant for junior corporate lawyers in India, \
focusing on consumer protection law (customer complaints) and company/contract/data-protection disputes.

RULES:
1. Use the tools first. For any legal question call search_legal_db (and search_indian_kanoon if available).
2. NEVER cite a case name, citation, section or rule that did not appear in a tool result. If you are \
unsure, say so. Do not invent citations under any circumstances.
3. If the sources are insufficient, say so plainly and suggest what to search for.
4. After each claim add [Source: <source name>, p.<page>] taken from the tool result.
5. For consumer complaints use consumer_forum_calculator for forum and limitation once you know the \
amount paid and the date the cause of action arose. If either is missing, ask for it first.
6. Laws get amended (e.g. IPC/CrPC replaced by BNS/BNSS). Remind the user to verify the current version.
7. This is a study aid, not legal advice.
8. If the question is not about law, politely say you only handle legal study questions."""

MODE_PROMPTS = {
    "research": """MODE: RESEARCH.
Structure answers as: Issue -> Applicable Law -> Case Law -> Analysis -> Practical Steps.""",
    "study": """MODE: STUDY (Socratic tutor).
Do NOT give the full answer immediately. Ask ONE question to test the user's understanding, wait for \
their reply, then evaluate it against the retrieved sources and give a hint or the next question. \
If the user says they give up or asks for the answer, then explain fully with sources.""",
    "draft": """MODE: DRAFT.
Help draft legal notices, replies and written statements. Use clear placeholders like [CLIENT NAME] \
for unknown facts. Base legal references only on retrieved sources. Mark every draft 'DRAFT - for \
review by a qualified advocate'.""",
}


def build_system_prompt(mode: str) -> str:
    return BASE_PROMPT + "\n\n" + MODE_PROMPTS.get(mode, MODE_PROMPTS["research"])


# ------------------------------------------------------------------
# 5. Helpers
# ------------------------------------------------------------------
def _text(content) -> str:
    """Gemini may return a str or a list of content blocks; normalise to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
    return str(content)


def _current_turn_tool_messages(messages) -> list[ToolMessage]:
    """ToolMessages produced after the latest HumanMessage."""
    out = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage):
            out.append(m)
    return list(reversed(out))


def _sources_from_tool_message(m: ToolMessage) -> list[dict]:
    name = getattr(m, "name", None)
    if name not in ("search_legal_db", "search_indian_kanoon"):
        return []
    try:
        data = json.loads(_text(m.content))
    except Exception:
        return []
    found = []
    for r in data.get("results", []):
        if name == "search_legal_db":
            found.append({"label": r.get("source"), "detail": f"p.{r.get('page')}", "url": None})
        else:
            found.append({"label": r.get("title"), "detail": r.get("court"), "url": r.get("url")})
    return found


# ------------------------------------------------------------------
# 6. Nodes
# ------------------------------------------------------------------
async def agent_node(state: ChatState):
    """LLM node: answers or requests a tool call, using the mode's system prompt."""
    mode = state.get("mode", "research")
    msgs = [SystemMessage(content=build_system_prompt(mode))] + state["messages"]
    response = await llm_with_tools.ainvoke(msgs)
    return {"messages": [response]}


_base_tool_node = ToolNode(tools)


async def tools_node(state: ChatState, config: RunnableConfig):
    """Run tools, then record which sources were retrieved for the UI."""
    result = await _base_tool_node.ainvoke(state, config)
    new_sources = []
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            new_sources.extend(_sources_from_tool_message(m))
    return {"messages": result["messages"], "sources": state.get("sources", []) + new_sources}


class VerifyResult(BaseModel):
    unsupported: list[str] = Field(
        default_factory=list,
        description="Case names, citations, or statutory sections in the answer that do NOT appear in the sources.",
    )


async def verifier_node(state: ChatState):
    """Check the final answer's citations against what the tools actually returned."""
    if state.get("mode") == "study":
        return {"verified": None, "unsupported": []}

    answer = _text(state["messages"][-1].content)
    sources_text = "\n\n".join(_text(m.content) for m in _current_turn_tool_messages(state["messages"]))
    try:
        check = await verifier_llm.with_structured_output(VerifyResult).ainvoke(
            "You are a strict legal citation checker.\n"
            "List every case name, case citation, statute section or rule number mentioned in the ANSWER "
            "that does NOT appear in the SOURCES. Ignore general statements without citations. "
            "If everything is supported, return an empty list.\n\n"
            f"ANSWER:\n{answer}\n\nSOURCES:\n{sources_text[:30000] or '(no sources retrieved)'}"
        )
        unsupported = list(check.unsupported) if check else []
        return {"verified": len(unsupported) == 0, "unsupported": unsupported}
    except Exception as exc:
        print(f"[verifier] failed: {exc!r}")
        return {"verified": None, "unsupported": []}


# ------------------------------------------------------------------
# 7. Checkpointer
# ------------------------------------------------------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# ------------------------------------------------------------------
# 8. Graph:  START -> agent -> (tools -> agent)* -> verifier -> END
# ------------------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_node("verifier", verifier_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "verifier"})
graph.add_edge("tools", "agent")
graph.add_edge("verifier", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------
# 9. Helpers (unchanged)
# ------------------------------------------------------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
