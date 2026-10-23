import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load this project's .env even when launched from the repository root.
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

st.set_page_config(page_title="Conversational Chatbot", page_icon="💬")
st.title("💬 Conversational Chatbot")
st.caption("A simple chatbot that remembers messages during this browser session.")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error(
        "Gemini API key not found. Create a .env file beside app.py "
        "and set GOOGLE_API_KEY=your_key"
    )
    st.stop()

# Keep the conversation in Streamlit session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept a new user message.
if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert saved messages to LangChain message objects.
    history = [SystemMessage(content=(
        "You are a helpful conversational assistant. Use the conversation "
        "history to answer follow-up questions. If you do not know something, "
        "say so clearly."
    ))]
    for message in st.session_state.messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model=MODEL_NAME,
                    temperature=0.3,
                    google_api_key=api_key,
                )
                response = llm.invoke(history)
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                st.error(f"Model request failed: {exc}")

with st.sidebar:
    st.subheader("Conversation")
    st.write(f"Messages: {len(st.session_state.messages)}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption("History is kept only in the current Streamlit session.")
