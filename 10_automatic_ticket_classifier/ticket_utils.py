from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "Documents"
DATASET_PATH = PROJECT_DIR / "data" / "Tickets.csv"
DB_PATH = PROJECT_DIR / "tickets.db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Explicit path means .env is found even when Streamlit is launched from the workspace root.
load_dotenv(PROJECT_DIR / ".env", override=False)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                issue_text TEXT NOT NULL,
                department TEXT NOT NULL,
                ai_answer TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open'
            )"""
        )


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_ticket_classifier() -> dict[str, Any]:
    """Embed the sample tickets, evaluate on a holdout set, and train the final classifier."""
    df = pd.read_csv(DATASET_PATH, header=None, names=["issue", "department"])
    df = df.dropna(subset=["issue", "department"])
    df["issue"] = df["issue"].astype(str).str.strip()
    df["department"] = df["department"].astype(str).str.strip()

    if len(df) < 10 or df["department"].nunique() < 2:
        raise ValueError("The sample ticket dataset must contain multiple departments and enough examples.")

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(df["issue"].tolist())
    labels = df["department"].tolist()
    x_train, x_test, y_train, y_test = train_test_split(
        vectors, labels, test_size=0.25, random_state=42, stratify=labels
    )
    test_model = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    test_model.fit(x_train, y_train)
    predictions = test_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    final_model = LinearSVC(class_weight="balanced", random_state=42, max_iter=10000)
    final_model.fit(vectors, labels)
    return {
        "model": final_model,
        "accuracy": accuracy,
        "report": report,
        "counts": df["department"].value_counts().to_dict(),
        "sample_count": len(df),
        "labels": sorted(df["department"].unique().tolist()),
    }


@st.cache_resource(show_spinner=False)
def get_policy_vectorstore() -> FAISS:
    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF policy manuals were found in {DOCS_DIR}")

    pages = []
    for pdf_path in pdf_paths:
        try:
            loaded_pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            st.warning(f"Could not read {pdf_path.name}: {exc}")
            continue
        for page in loaded_pages:
            page.metadata["source_name"] = pdf_path.name
        pages.extend(loaded_pages)

    if not pages:
        raise ValueError("No readable text was extracted from the policy PDFs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=120)
    chunks = splitter.split_documents(pages)
    return FAISS.from_documents(chunks, get_embeddings())


def classify_ticket(issue: str) -> str:
    bundle = get_ticket_classifier()
    vector = get_embeddings().embed_query(issue)
    return str(bundle["model"].predict([vector])[0])


def answer_from_policies(issue: str, retrieved_docs: list) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Google API key not found. Add GOOGLE_API_KEY=your_key to the project-local .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    context_parts = []
    for doc in retrieved_docs:
        page = int(doc.metadata.get("page", 0)) + 1
        name = doc.metadata.get("source_name", Path(doc.metadata.get("source", "policy PDF")).name)
        context_parts.append(f"SOURCE: {name}, page {page}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a workplace support assistant. Answer using only the supplied policy excerpts.
If the excerpts do not contain enough information, say clearly that the provided policy manuals do not answer the question. Do not invent company policy, deadlines, contacts, or guarantees. Give concise, practical guidance and mention the policy manual/page when useful."""),
        ("human", "Employee issue/question:\n{question}\n\nRetrieved policy excerpts:\n{context}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": issue, "context": context})


def retrieve_policy_docs(issue: str, k: int = 4) -> list:
    return get_policy_vectorstore().similarity_search(issue, k=k)


def create_ticket(issue: str, department: str, answer: str) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO tickets (issue_text, department, ai_answer) VALUES (?, ?, ?)",
            (issue, department, answer),
        )
        return int(cursor.lastrowid)


def get_tickets(department: str = "All", status: str = "All") -> pd.DataFrame:
    init_db()
    query = "SELECT id, created_at, issue_text, department, ai_answer, status FROM tickets WHERE 1=1"
    params: list[str] = []
    if department != "All":
        query += " AND department = ?"
        params.append(department)
    if status != "All":
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id DESC"
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def update_ticket_status(ticket_id: int, status: str) -> None:
    if status not in {"Open", "In progress", "Resolved"}:
        raise ValueError("Unsupported ticket status")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))
