from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MCQ(BaseModel):
    """One multiple-choice question."""

    question: str = Field(description="A clear question answerable from the supplied source passages.")
    options: List[str] = Field(description="Exactly four distinct answer choices.")
    correct_answer: str = Field(description="The exact text of one item in options, not just a letter.")
    explanation: str = Field(description="Brief explanation grounded in the supplied source passages.")


class MCQSet(BaseModel):
    """A collection of generated MCQs."""

    questions: List[MCQ] = Field(description="The requested multiple-choice questions.")


def uploaded_files_fingerprint(files) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8", errors="ignore"))
        digest.update(file.getvalue())
    return digest.hexdigest()


def load_uploaded_pdfs(files):
    documents = []
    for uploaded in files:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(uploaded.getvalue())
                temp_path = temp_file.name
            pages = PyPDFLoader(temp_path).load()
            for page in pages:
                page.metadata["source"] = uploaded.name
            documents.extend(pages)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not documents:
        raise ValueError("No pages could be read from the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No readable text was found. This app needs text-based PDFs; scanned PDFs require OCR first."
        )
    return chunks


def build_vector_store(chunks):
    # Local embeddings: no embedding API key or per-request embedding charge.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def format_context(documents) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        # PyPDFLoader usually gives a zero-based page index.
        page_index = doc.metadata.get("page")
        page_label = str(int(page_index) + 1) if isinstance(page_index, int) else "?"
        blocks.append(
            f"[Passage {i} | Source: {source} | PDF page: {page_label}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def generate_mcqs(*, api_key: str, topic: str, count: int, difficulty: str, context: str) -> MCQSet:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=api_key)
    structured_llm = llm.with_structured_output(MCQSet)

    system_prompt = (
        "You are an experienced question setter. Create accurate educational multiple-choice questions "
        "using only the supplied source passages. Do not invent facts or use outside knowledge. "
        "Each question must have exactly four distinct options and exactly one correct answer. "
        "The correct_answer field must exactly match one of the option strings. Distractors should be "
        "plausible but clearly incorrect based on the source. Avoid trick questions, duplicate questions, "
        "and questions whose answer is not supported by the passages. Keep explanations brief and grounded. "
        "If the source does not contain enough information, make fewer questions rather than fabricate."
    )
    user_prompt = (
        f"Create {count} MCQs.\nDifficulty: {difficulty}.\nTopic/focus: {topic or 'the most important concepts in the provided passages'}.\n\n"
        f"Source passages:\n{context}"
    )
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    if not isinstance(result, MCQSet):
        result = MCQSet.model_validate(result)

    # Additional checks: providers can occasionally return malformed option sets.
    cleaned = []
    for item in result.questions:
        options = [str(option).strip() for option in item.options if str(option).strip()]
        if len(options) != 4 or len({option.casefold() for option in options}) != 4:
            continue
        if item.correct_answer.strip() not in options:
            # Try a case-insensitive match, but never silently choose an unrelated option.
            match = next((o for o in options if o.casefold() == item.correct_answer.strip().casefold()), None)
            if match is None:
                continue
            item.correct_answer = match
        item.options = options
        cleaned.append(item)
    if not cleaned:
        raise ValueError("The model did not return any valid four-option questions. Try another topic or fewer questions.")
    return MCQSet(questions=cleaned)


def mcqs_as_records(mcq_set: MCQSet):
    records = []
    for index, item in enumerate(mcq_set.questions, start=1):
        records.append({
            "question_number": index,
            "question": item.question,
            "option_A": item.options[0],
            "option_B": item.options[1],
            "option_C": item.options[2],
            "option_D": item.options[3],
            "correct_answer": item.correct_answer,
            "explanation": item.explanation,
        })
    return records


st.set_page_config(page_title="MCQ Creator", page_icon="📝", layout="wide")
st.title("📝 MCQ Creator")
st.caption("Generate source-grounded multiple-choice questions from PDFs using retrieval + Gemini.")

with st.sidebar:
    st.subheader("Settings")
    api_key_in_env = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if api_key_in_env:
        st.success("Gemini API key found in .env")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    else:
        api_key = st.text_input("Google AI Studio API key", type="password", help="Used for generation only; not saved to disk by this app.")
    st.caption(f"Generation model: `{MODEL_NAME}`")
    st.caption(f"Local embedding model: `{EMBEDDING_MODEL}`")
    st.caption("PDF embeddings run locally; Gemini is used to generate questions.")

st.markdown("### 1. Upload source PDFs")
uploads = st.file_uploader(
    "Choose one or more text-based PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="The included sample_docs folder has example PDFs you can upload to test the app.",
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.processed_fingerprint = None
    st.session_state.chunk_count = 0
    st.session_state.processed_sources = []
    st.session_state.generated_mcqs = None

current_fingerprint = uploaded_files_fingerprint(uploads) if uploads else None
is_processed = bool(uploads) and current_fingerprint == st.session_state.processed_fingerprint and st.session_state.vector_store is not None

process_col, status_col = st.columns([1, 2])
with process_col:
    process_clicked = st.button("Process PDFs", type="primary", disabled=not bool(uploads))
with status_col:
    if is_processed:
        st.success(f"Indexed {len(uploads)} PDF(s) into {st.session_state.chunk_count} text chunks.")
    elif uploads:
        st.info("Process the PDFs before generating MCQs.")

if process_clicked:
    try:
        with st.spinner("Reading PDFs, splitting text, and building the local vector index. First run may download the embedding model..."):
            chunks = load_uploaded_pdfs(uploads)
            store = build_vector_store(chunks)
        st.session_state.vector_store = store
        st.session_state.processed_fingerprint = current_fingerprint
        st.session_state.chunk_count = len(chunks)
        st.session_state.processed_sources = [f.name for f in uploads]
        st.session_state.generated_mcqs = None
        st.success(f"Ready! Processed {len(uploads)} PDF(s) into {len(chunks)} chunks.")
        st.rerun()
    except Exception as exc:
        st.error(f"Could not process the PDFs: {exc}")

st.markdown("### 2. Configure your questions")
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input(
        "Topic or focus",
        placeholder="e.g. key concepts, causes of the event, definitions, Chapter 2",
        help="Used to retrieve relevant passages from the uploaded PDFs.",
    )
with col2:
    count = st.selectbox("Number of MCQs", [3, 5, 8, 10], index=1)

difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"], index=1)
top_k = st.slider("Number of source chunks to retrieve", min_value=2, max_value=10, value=5)

if st.button("Generate MCQs", type="primary", disabled=not is_processed):
    if not api_key:
        st.error("Add your Google AI Studio API key in the project .env file or sidebar.")
    else:
        try:
            search_query = topic.strip() or "important concepts, key definitions, and main ideas"
            with st.spinner("Finding relevant passages and generating MCQs..."):
                retrieved_docs = st.session_state.vector_store.similarity_search(search_query, k=top_k)
                context = format_context(retrieved_docs)
                result = generate_mcqs(
                    api_key=api_key,
                    topic=search_query,
                    count=count,
                    difficulty=difficulty,
                    context=context,
                )
            st.session_state.generated_mcqs = result.model_dump()
            st.session_state.last_retrieved_sources = [
                {
                    "source": doc.metadata.get("source", "Unknown PDF"),
                    "page": int(doc.metadata["page"]) + 1 if isinstance(doc.metadata.get("page"), int) else None,
                    "preview": doc.page_content[:260].strip(),
                }
                for doc in retrieved_docs
            ]
        except Exception as exc:
            st.error(f"MCQ generation failed: {exc}")

saved = st.session_state.generated_mcqs
if saved:
    mcq_set = MCQSet.model_validate(saved)
    st.markdown("---")
    st.markdown(f"## Generated {len(mcq_set.questions)} MCQ(s)")
    records = mcqs_as_records(mcq_set)
    for i, item in enumerate(mcq_set.questions, start=1):
        with st.container(border=True):
            st.markdown(f"**Q{i}. {item.question}**")
            for letter, option in zip(["A", "B", "C", "D"], item.options):
                st.markdown(f"**{letter}.** {option}")
            with st.expander("Show answer and explanation"):
                st.markdown(f"**Correct answer:** {item.correct_answer}")
                st.write(item.explanation)

    st.download_button(
        "Download MCQs as JSON",
        data=json.dumps(saved, ensure_ascii=False, indent=2),
        file_name="generated_mcqs.json",
        mime="application/json",
    )
    csv_data = pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download MCQs as CSV",
        data=csv_data,
        file_name="generated_mcqs.csv",
        mime="text/csv",
    )

    with st.expander("Inspect retrieved source passages"):
        for i, source in enumerate(st.session_state.get("last_retrieved_sources", []), start=1):
            page_info = f", page {source['page']}" if source.get("page") else ""
            st.markdown(f"**Passage {i}: {source['source']}{page_info}**")
            st.write(source["preview"] + ("…" if len(source["preview"]) == 260 else ""))

st.caption("Note: scanned/image-only PDFs may need OCR before this app can extract their text. Always review generated questions and answers before using them in an exam.")
