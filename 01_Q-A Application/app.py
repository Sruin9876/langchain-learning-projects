from pathlib import Path

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

st.set_page_config(
    page_title="LangChain Demo",
    page_icon="🤖"
)

st.title("LangChain Question Answering")

question = st.text_input("Enter your question")

if st.button("Generate", disabled=not question.strip()):
    with st.spinner("Generating answer..."):
        try:
            llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0
            )

            response = llm.invoke(question)

            st.subheader("Answer")
            st.write(response.text)

        except Exception as e:
            st.error(f"Error: {e}")