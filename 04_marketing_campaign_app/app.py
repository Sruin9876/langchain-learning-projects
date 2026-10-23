from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_google_genai import ChatGoogleGenerativeAI


# Load this project's .env even when Streamlit is launched from the workspace root.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

EXAMPLES_BY_TASK = {
    "Write a sales copy": [
        {
            "input": "An insulated reusable bottle that keeps drinks cold all day.",
            "output": "Keep your day refreshingly cool. Our insulated bottle keeps your favourite drinks cold for hours, wherever your plans take you. Refill, reuse, and go."
        },
        {
            "input": "A lightweight backpack for college students with a padded laptop sleeve.",
            "output": "Carry your campus essentials without the extra weight. With a padded laptop sleeve and space for your everyday gear, this backpack is ready for every lecture and late study session."
        },
        {
            "input": "A beginner-friendly online course that teaches Python through projects.",
            "output": "Turn curiosity into code. Learn Python by building practical projects, one manageable step at a time—no advanced experience required."
        },
    ],
    "Create a tweet": [
        {
            "input": "Launching a habit-tracking app for students.",
            "output": "Small habits. Real progress. 🎯 Meet the habit tracker designed to help students stay consistent, one day at a time. What habit are you building?"
        },
        {
            "input": "A new eco-friendly stationery collection.",
            "output": "New notes, lighter footprint 🌱 Meet our eco-friendly stationery collection—made for big ideas and more thoughtful everyday choices."
        },
        {
            "input": "A weekend discount on a local coffee shop's drinks.",
            "output": "Your weekend pick-me-up is calling ☕ Stop by this weekend and enjoy a little treat on us—with special offers on selected drinks."
        },
    ],
    "Write a product description": [
        {
            "input": "A bamboo toothbrush with soft bristles and a recyclable paper box.",
            "output": "Make your daily routine a little more mindful with this bamboo toothbrush. Its soft bristles offer a comfortable clean, while the recyclable paper box helps reduce unnecessary packaging."
        },
        {
            "input": "A compact desk lamp with adjustable brightness for studying.",
            "output": "Create a comfortable study setup with a compact desk lamp featuring adjustable brightness. Direct the light where you need it and choose a level that suits reading, writing, or late-night projects."
        },
        {
            "input": "A cotton tote bag with a wide base for carrying books and groceries.",
            "output": "A practical everyday tote with room for books, groceries, and daily essentials. Made from cotton and designed with a wide base, it is an easy companion for errands, classes, and weekend outings."
        },
    ],
}


def build_prompt(task_type: str, audience: str, word_limit: int, brief: str) -> str:
    examples = EXAMPLES_BY_TASK[task_type]

    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Campaign brief: {input}\nExample result: {output}",
    )

    # This selector includes as many examples as fit within its approximate length budget.
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=900,
    )

    prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=(
            "You are an experienced marketing copywriter. Create the requested content.\n"
            "Task: {task_type}\n"
            "Target audience: {audience}\n"
            "Maximum length: {word_limit} words\n\n"
            "Use the examples as style guidance, not as facts to copy. Keep the writing clear, "
            "specific, engaging, and suitable for the target audience. Do not invent product claims.\n"
            "Examples:"
        ),
        suffix=(
            "\n\nNow create the requested content for this brief:\n"
            "{user_brief}\n\n"
            "Return only the finished marketing content. Keep it within {word_limit} words."
        ),
        input_variables=["task_type", "audience", "word_limit", "user_brief"],
        example_separator="\n\n",
    )

    return prompt_template.format(
        task_type=task_type,
        audience=audience,
        # LengthBasedExampleSelector measures prompt input values by joining them;
        # convert the numeric slider value to text before prompt formatting.
        word_limit=str(word_limit),
        user_brief=brief,
    )


def generate_campaign(brief: str, task_type: str, audience: str, word_limit: int) -> str:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Create a .env file in this project's folder "
            "using .env.example, then add your Google AI Studio API key."
        )

    prompt = build_prompt(task_type, audience, word_limit, brief)
    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
    )
    response = model.invoke(prompt)
    content = response.content

    # Chat models normally return text here; handle content-block responses too.
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        ).strip()
    return str(content).strip()


st.set_page_config(page_title="Marketing Campaign App", page_icon="📣", layout="centered")
st.title("📣 Marketing Campaign Generator")
st.caption("Generate sales copy, social posts, and product descriptions with few-shot prompting.")

with st.form("campaign_form"):
    brief = st.text_area(
        "Describe your product, service, or campaign",
        height=150,
        placeholder="Example: A lightweight backpack for college students with a padded laptop sleeve.",
    )
    task_type = st.selectbox(
        "What should the app create?",
        tuple(EXAMPLES_BY_TASK.keys()),
    )
    audience = st.selectbox(
        "Who is the target audience?",
        ("Children (6–12)", "Teenagers (13–19)", "Adults (20–59)", "Senior citizens (60+)", "General audience"),
    )
    word_limit = st.slider("Maximum word count", min_value=10, max_value=200, value=50, step=5)
    submitted = st.form_submit_button("Generate campaign content", type="primary")

if submitted:
    if not brief.strip():
        st.warning("Enter a product or campaign brief first.")
    else:
        with st.spinner("Generating your marketing content..."):
            try:
                result = generate_campaign(brief.strip(), task_type, audience, word_limit)
                st.subheader("Generated content")
                st.write(result)
                st.caption(f"Model: {MODEL_NAME} · Requested limit: {word_limit} words")
            except Exception as exc:
                st.error(str(exc))
                st.caption("Check your API key, model availability, internet connection, and API quota.")
