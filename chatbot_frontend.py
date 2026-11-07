
import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

# Initialize chat history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Extract only text from Gemini content blocks
def extract_text(message_chunk):
    content = message_chunk.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

        return "".join(text_parts)

    return ""


user_input = st.chat_input("Type here")

if user_input:
    # Display and save the user's message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant's response
    with st.chat_message("assistant"):
        def generate_response():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                text = extract_text(message_chunk)

                if text:
                    yield text

        ai_message = st.write_stream(generate_response())

    # Save the complete response
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
