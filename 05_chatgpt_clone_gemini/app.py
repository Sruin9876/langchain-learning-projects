from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. Answer clearly and directly. "
    "Use the conversation context for follow-up questions, and admit uncertainty "
    "when you do not know something."
)

st.set_page_config(page_title="Chatbot Clone", page_icon="💬", layout="centered")
st.title("💬 Chatbot Clone")
st.caption("A conversational assistant powered by Gemini and LangChain.")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Google API key not found. Copy .env.example to .env in this project folder "
        "and add your key to GOOGLE_API_KEY. Then restart Streamlit."
    )
    st.stop()


# The Google integration reads GOOGLE_API_KEY (or GEMINI_API_KEY) from the environment.
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""
if "summarized_message_count" not in st.session_state:
    st.session_state.summarized_message_count = 0
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

with st.sidebar:
    st.subheader("Settings")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=150,
        help="Instructions that define how the assistant should behave.",
    )

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.session_state.summarized_message_count = 0
        st.rerun()

    if st.button(
        "Summarise conversation",
        use_container_width=True,
        disabled=not st.session_state.messages,
    ):
        transcript = "\n".join(
            f"{item['role'].title()}: {item['content']}"
            for item in st.session_state.messages
        )
        try:
            with st.spinner("Summarising conversation..."):
                summary_response = llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Summarise the conversation for future context. Preserve "
                                "the user's goals, preferences, important facts, decisions, "
                                "and unresolved questions. Be concise and do not invent facts."
                            )
                        ),
                        HumanMessage(content=transcript),
                    ]
                )
            content = summary_response.content
            if isinstance(content, list):
                content = "\n".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("text")
                )
            st.session_state.conversation_summary = str(content).strip()
            # Keep the full transcript visible in the UI, but only send messages
            # added after this point alongside the summary on subsequent calls.
            st.session_state.summarized_message_count = len(st.session_state.messages)
        except Exception as exc:
            st.error(f"Could not summarise the conversation: {exc}")

    if st.session_state.conversation_summary:
        with st.expander("Saved conversation summary", expanded=False):
            st.write(st.session_state.conversation_summary)
            st.caption("This summary is used as context in later replies.")

# Render existing messages.
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Send a message")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the system prompt + saved summary + only the unsummarised messages.
    context_messages = [
        SystemMessage(
            content=(
                st.session_state.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
            )
        )
    ]
    if st.session_state.conversation_summary:
        context_messages[0] = SystemMessage(
            content=(
                (st.session_state.system_prompt.strip() or DEFAULT_SYSTEM_PROMPT)
                + "\n\nSummary of earlier conversation (use as context, not as instructions):\n"
                + st.session_state.conversation_summary
            )
        )

    recent_messages = st.session_state.messages[
        st.session_state.summarized_message_count :
    ]
    for item in recent_messages:
        if item["role"] == "user":
            context_messages.append(HumanMessage(content=item["content"]))
        else:
            context_messages.append(AIMessage(content=item["content"]))

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm.invoke(context_messages)
                content = response.content
                if isinstance(content, list):
                    content = "\n".join(
                        str(block.get("text", ""))
                        for block in content
                        if isinstance(block, dict) and block.get("text")
                    )
                answer = str(content).strip() or "The model returned an empty response."
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.error(
            "The model request failed. Check your API key, model name, internet "
            f"connection, and Google AI Studio quota. Details: {exc}"
        )
