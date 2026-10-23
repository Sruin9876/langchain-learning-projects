from pathlib import Path
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

st.set_page_config(
    page_title="Email Generator",
    page_icon="📧",
    layout="centered",
)


def build_chain():
    """Create a modern LangChain prompt -> Gemini -> text-parser chain."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add your Google AI Studio key to this project's .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        timeout=45,
        max_retries=1,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful professional email-writing assistant. Draft only the email; "
                "do not explain your process. Never invent facts, promises, dates, or attachments. "
                "Use placeholders such as [date] when required information is missing. Keep the "
                "message clear and appropriately concise. Include a useful subject line, greeting, "
                "body, and sign-off. Format the result as plain text.",
            ),
            (
                "human",
                "Write an email using the details below.\n\n"
                "Purpose / topic: {topic}\n"
                "Sender name: {sender}\n"
                "Recipient name: {recipient}\n"
                "Writing style: {style}\n"
                "Desired length: {length}\n"
                "Additional details to include: {details}\n\n"
                "Style guidance: Formal means professional and respectful; Appreciative means "
                "warm and thankful; Dissatisfied but constructive means firm without insults or "
                "unprofessional language; Neutral means direct and balanced; Friendly means casual "
                "but considerate. If a field is blank, do not invent missing details.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


@st.cache_resource(show_spinner=False)
def get_chain(model_name: str, api_key_fingerprint: str):
    # The fingerprint invalidates the cached chain if the API key changes; the key
    # itself is not used as the visible cache argument.
    del api_key_fingerprint
    os.environ["GEMINI_MODEL"] = model_name
    return build_chain()


st.title("📧 Email Generator")
st.caption("Draft a clear email from a short brief using Gemini and LangChain.")

with st.form("email_form"):
    topic = st.text_area(
        "What should the email be about?",
        height=140,
        placeholder="Example: Request an extension for the project deadline because the lab was unavailable.",
    )

    col1, col2 = st.columns(2)
    with col1:
        sender = st.text_input("Sender name", placeholder="Your name")
    with col2:
        recipient = st.text_input("Recipient name", placeholder="Professor / Manager / Client")

    col3, col4 = st.columns(2)
    with col3:
        style = st.selectbox(
            "Writing style",
            ["Formal", "Appreciative", "Dissatisfied but constructive", "Neutral", "Friendly"],
        )
    with col4:
        length = st.selectbox("Length", ["Short", "Medium", "Detailed"], index=1)

    details = st.text_area(
        "Extra details (optional)",
        height=90,
        placeholder="Key points to mention, relevant context, action requested, or deadline.",
    )
    submitted = st.form_submit_button("Generate email", type="primary", use_container_width=True)

if submitted:
    if not topic.strip():
        st.warning("Enter the email topic or purpose first.")
    else:
        try:
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if not api_key:
                st.error(
                    "GOOGLE_API_KEY is missing. Create a .env file beside app.py from .env.example, "
                    "add your Google AI Studio API key, and restart Streamlit."
                )
            else:
                model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
                with st.spinner("Drafting your email…"):
                    chain = get_chain(model_name, hashlib.sha256(api_key.encode("utf-8")).hexdigest())
                    email_text = chain.invoke(
                        {
                            "topic": topic.strip(),
                            "sender": sender.strip() or "Not provided",
                            "recipient": recipient.strip() or "Not provided",
                            "style": style,
                            "length": length,
                            "details": details.strip() or "None provided",
                        }
                    )
                st.session_state["generated_email"] = email_text.strip()
                st.session_state["email_output"] = email_text.strip()
        except Exception as exc:
            message = str(exc)
            st.error("Email generation failed.")
            if "API_KEY_INVALID" in message or "API key not valid" in message:
                st.info("Check that GOOGLE_API_KEY in this project's .env is a valid Gemini API key.")
            elif "429" in message or "RESOURCE_EXHAUSTED" in message:
                st.info("The API quota or rate limit may have been reached. Check your Gemini API usage and try again later.")
            else:
                st.code(message[:2500])

if st.session_state.get("generated_email"):
    st.subheader("Generated email")
    st.text_area("Review and edit your draft", value=st.session_state["generated_email"], height=320, key="email_output")
    st.download_button(
        "Download email as .txt",
        data=st.session_state["email_output"],
        file_name="generated_email.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.caption("Review names, dates, facts, and tone before sending. The draft is not sent automatically.")
