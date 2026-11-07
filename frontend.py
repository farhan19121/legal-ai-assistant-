import queue
import uuid

import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads, run_async, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

st.set_page_config(page_title="AI Legal Study Assistant (India)", page_icon="⚖️")

SAMPLE_QUESTIONS = [
    "A customer bought a defective phone for ₹60,000 and the seller refuses to replace it. Which forum and what is the limitation period?",
    "What is the liability of an e-commerce platform for a seller's defective product?",
    "Explain deficiency in service under the Consumer Protection Act 2019.",
    "Is a builder's delay in handing over a flat a consumer dispute?",
]


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def extract_text(content) -> str:
    """Gemini may return a str OR a list of content blocks; normalize to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def render_meta(meta):
    """Verified badge + sources panel under an assistant answer."""
    if not meta:
        return
    verified = meta.get("verified")
    if verified is True:
        st.success("✅ Citations checked against retrieved sources")
    elif verified is False:
        st.warning(
            "⚠️ Needs checking. Not found in retrieved sources: "
            + "; ".join(meta.get("unsupported", []))
        )
    sources = meta.get("sources") or []
    if sources:
        with st.expander(f"Sources used ({len(sources)})"):
            for s in sources:
                line = f"- **{s.get('label')}**"
                if s.get("detail"):
                    line += f", {s['detail']}"
                if s.get("url"):
                    line += f" — [open]({s['url']})"
                st.markdown(line)


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "queued_question" not in st.session_state:
    st.session_state["queued_question"] = None

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("⚖️ AI Legal Study Assistant")

mode_label = st.sidebar.radio("Mode", ["Research", "Study", "Draft"])
mode = mode_label.lower()
st.sidebar.warning("Study aid only, not legal advice. Verify every citation and the current version of the law before relying on it.")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Sample questions")
for i, q in enumerate(SAMPLE_QUESTIONS):
    if st.sidebar.button(q[:55] + "…", key=f"sample_{i}"):
        st.session_state["queued_question"] = q

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                continue
            text = extract_text(msg.content)
            if not text.strip():
                continue
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": text})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================
st.title("AI Legal Study Assistant")
st.caption("Indian consumer protection and corporate law · answers grounded in retrieved sources")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_meta(message.get("meta"))

typed = st.chat_input("Ask a legal question…")
user_input = typed or st.session_state.pop("queued_question", None)

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Per-turn inputs reset sources/verification for this question
    turn_input = {
        "messages": [HumanMessage(content=user_input)],
        "mode": mode,
        "sources": [],
        "verified": None,
        "unsupported": [],
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        turn_input, config=CONFIG, stream_mode="messages"
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if isinstance(message_chunk, str) and message_chunk == "error":
                    raise metadata

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", None) or "tool"
                    label = f"🔧 Using `{tool_name}` …"
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(label, expanded=True)
                    else:
                        status_holder["box"].update(label=label, state="running", expanded=True)

                # Stream ONLY the agent's text (not the verifier's internal LLM call)
                if (
                    isinstance(message_chunk, AIMessage)
                    and metadata.get("langgraph_node") == "agent"
                ):
                    text = extract_text(message_chunk.content)
                    if text:
                        yield text

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        # Read verification result + sources saved in graph state for this turn
        meta = {}
        try:
            final_state = run_async(chatbot.aget_state(CONFIG))
            meta = {
                "verified": final_state.values.get("verified"),
                "unsupported": final_state.values.get("unsupported", []),
                "sources": final_state.values.get("sources", []),
            }
        except Exception as exc:
            print(f"[frontend] could not read final state: {exc!r}")
        render_meta(meta)

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message, "meta": meta}
    )
