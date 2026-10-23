from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

from utils import generate_script, search_topic

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="YouTube Script Writer", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1677d2;
        color: white;
        border: 0;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #105ca5;
        color: white;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 YouTube Script Writing Tool")
st.caption("Research a topic, create a title, and draft a video script with Gemini.")

with st.sidebar:
    st.subheader("Settings")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    st.caption(f"Model: `{model_name}`")
    image_path = BASE_DIR / "assets" / "Youtube.jpg"
    if image_path.exists():
        st.image(str(image_path), width=250)
    st.info("Your API key is read from this project's `.env` file. It is not entered into the page.")

with st.form("script_form"):
    topic = st.text_input("Video topic", placeholder="e.g. How RAG works in AI")
    col1, col2 = st.columns(2)
    with col1:
        video_length = st.slider("Expected video length (minutes)", min_value=1, max_value=30, value=5)
        audience = st.text_input("Target audience", value="beginners")
        language = st.selectbox("Language", ["English", "Hindi", "Hinglish"])
    with col2:
        tone = st.selectbox(
            "Tone",
            ["Clear and educational", "Conversational", "Energetic", "Documentary", "Humorous"],
        )
        creativity = st.slider(
            "Creative style",
            min_value=0,
            max_value=10,
            value=2,
            help="Lower values favour direct, factual structure; higher values encourage more vivid hooks and examples.",
        )
        do_research = st.checkbox("Search the web for supporting information", value=True)
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        submitted = st.form_submit_button("Generate script", type="primary", use_container_width=True)
    with button_col2:
        test_search = st.form_submit_button("Test web search", use_container_width=True)

if test_search:
    if not topic.strip():
        st.error("Enter a video topic first.")
    else:
        with st.spinner("Testing web search only (no Gemini calls)…"):
            test_results, test_error = search_topic(topic.strip())
        if test_results:
            st.success("Web search is working.")
            with st.expander("Search results", expanded=True):
                st.write(test_results)
        else:
            st.error("Web search did not return results.")
            st.code(test_error or "No additional error details were returned.")

if submitted:
    if not topic.strip():
        st.error("Enter a video topic first.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is missing. Create this project's `.env` from `.env.example` and add your Google AI Studio API key."
        )
    else:
        try:
            with st.spinner("Researching the topic and writing your script…"):
                search_result, search_error, title, script = generate_script(
                    topic=topic.strip(),
                    video_length=video_length,
                    creativity=creativity,
                    audience=audience.strip() or "general viewers",
                    tone=tone,
                    language=language,
                    model_name=model_name,
                    do_research=do_research,
                )
            st.success("Your draft is ready.")
            st.subheader("🔥 Suggested title")
            st.write(title)
            st.subheader("📝 Video script")
            st.markdown(script)

            st.download_button(
                "Download script as Markdown",
                data=f"# {title}\n\nTopic: {topic}\n\n{script}\n",
                file_name="youtube_script.md",
                mime="text/markdown",
            )

            with st.expander("🔎 Research notes used"):
                if search_result:
                    st.caption("These are search snippets, not independently verified facts.")
                    st.write(search_result)
                elif search_error:
                    st.warning("Web research did not succeed. The script was generated without search results.")
                    st.code(search_error)
                elif not do_research:
                    st.info("Web research was turned off; the script was generated without external search notes.")
                else:
                    st.warning("The search provider returned no results. The script was generated without external search notes.")
        except Exception as exc:
            st.error("Script generation failed.")
            st.caption("Check your API key, model availability, internet connection, and terminal error details.")
            st.code(f"{type(exc).__name__}: {exc}")
