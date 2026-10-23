from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from utils import extract_resume_documents, split_resume_documents, rank_resumes

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env", override=True)

st.set_page_config(page_title="Resume Screening Assistant", page_icon="📄", layout="wide")
st.title("Resume Screening Assistant")
st.caption("Evidence-based resume exploration • Local embeddings and FAISS • Optional Gemini summaries")

with st.sidebar:
    st.header("How this works")
    st.write("Upload resumes and a job description. The app compares their text using local embeddings, then retrieves the most relevant evidence passages.")
    st.warning("This is a decision-support demo, not an automated hiring system. Similarity scores are not qualification scores. Review original resumes and assess every applicant using consistent, job-related criteria.")
    st.write("**Privacy:** PDF parsing and embeddings run locally. If you request AI summaries, selected redacted text excerpts are sent to Google's Gemini API.")

@st.cache_resource(show_spinner="Loading the local embedding model (first run may download it)...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

class CandidateReview(BaseModel):
    resume_id: str = Field(description="The exact candidate ID supplied in the input")
    filename: str = Field(description="The exact resume filename supplied in the input")
    evidence_summary: str = Field(description="Brief factual summary of job-related evidence found in excerpts")
    requirements_supported_by_evidence: list[str] = Field(description="Job requirements explicitly supported by the supplied excerpts")
    requirements_not_evident_in_excerpts: list[str] = Field(description="Requirements for which these excerpts provide no clear evidence; this does not mean the applicant lacks the skill")
    verification_notes: list[str] = Field(description="Specific points a human reviewer should verify in the original resume")

class CandidateReviewBatch(BaseModel):
    reviews: list[CandidateReview]


def redact_personal_contacts(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email redacted]", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)", "[phone redacted]", text)
    return text


def generate_evidence_reviews(job_description: str, candidates: list[dict]) -> CandidateReviewBatch:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Google API key found. Add GOOGLE_API_KEY to this project's .env file.")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You assist a human reviewer by summarizing explicit, job-related evidence from resume excerpts.
Use ONLY the supplied job description and resume excerpts. Do not invent qualifications, employers, dates, degrees, skills, or achievements. Do not infer or evaluate age, gender, race or ethnicity, religion, disability, health, nationality, marital status, family status, or other protected/personal traits. Do not recommend hiring, rejecting, or ranking a person. Absence from supplied excerpts is not evidence that the candidate lacks a qualification; label it as not evident in these excerpts. Keep each statement traceable to the text and be concise."""),
        ("human", "Job description:\n{job_description}\n\nCandidate evidence (JSON):\n{candidate_evidence}\n\nReturn one review for each exact resume_id and filename supplied. Do not add candidates.")
    ])
    chain = prompt | llm.with_structured_output(CandidateReviewBatch)
    # Send only the top matched, redacted excerpts—not full PDF files.
    return chain.invoke({
        "job_description": job_description,
        "candidate_evidence": json.dumps(candidates, ensure_ascii=False),
    })

if "resume_results" not in st.session_state:
    st.session_state.resume_results = []
if "resume_job_description" not in st.session_state:
    st.session_state.resume_job_description = ""
if "resume_reviews" not in st.session_state:
    st.session_state.resume_reviews = {}

job_description = st.text_area(
    "Job description",
    height=180,
    placeholder="Paste the role responsibilities and job-related requirements...",
    help="Use specific role requirements. Avoid adding personal or protected-characteristic preferences.",
)
uploads = st.file_uploader(
    "Upload resumes (PDF only)", type=["pdf"], accept_multiple_files=True,
    help="For the demo, start with 2–5 resumes. Scanned/image-only PDFs may need OCR before they can be read.",
)
number_to_show = st.slider("Number of resumes to display", min_value=1, max_value=20, value=5)

if st.button("Analyze resumes", type="primary", use_container_width=False):
    if not job_description.strip():
        st.error("Paste a job description first.")
    elif not uploads:
        st.error("Upload at least one PDF resume.")
    else:
        try:
            with st.spinner("Extracting resume text..."):
                docs, warnings = extract_resume_documents(uploads)
            for warning in warnings:
                st.warning(warning)
            if not docs:
                st.error("No selectable text was extracted from these PDFs. They may be scanned images; OCR would be needed.")
            else:
                with st.spinner("Loading the embedding model and indexing resume passages locally..."):
                    chunks = split_resume_documents(docs)
                    if not chunks:
                        st.error("No usable text chunks were found in the uploaded resumes.")
                    else:
                        embeddings = get_embeddings()
                        index = FAISS.from_documents(chunks, embeddings)
                        with st.spinner("Comparing the job description with resume passages..."):
                            results = rank_resumes(job_description, index, chunks, number_to_show)
                        st.session_state.resume_results = results
                        st.session_state.resume_job_description = job_description
                        st.session_state.resume_reviews = {}
                        st.success(f"Processed {len({d.metadata['resume_id'] for d in docs})} resumes and indexed {len(chunks)} text passages.")
        except Exception as exc:
            st.error(f"Could not analyze resumes: {exc}")

results = st.session_state.resume_results
if results:
    st.subheader("Resume results")
    st.caption("Ordered by semantic similarity to the job description. Scores describe text similarity only; they are not calibrated percentages of qualification or predictions of job performance.")
    st.write("Review the original resumes and verify every extracted detail. A high similarity result can still be irrelevant, and a lower result can still contain important qualifications.")
    for idx, item in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### {idx}. {item['filename']}")
            st.metric("Semantic similarity (rough)", f"{item['similarity']:.3f}")
            st.caption("Higher means the embedding model found closer text passages for this query; this is not a hiring score.")
            st.markdown("**Retrieved evidence passages**")
            for evidence in item["evidence"]:
                st.markdown(f"- **Page {evidence['page']}** · {evidence['text']}")
            review = st.session_state.resume_reviews.get(item['resume_id'])
            if review:
                st.markdown("**AI evidence summary**")
                st.write(review.evidence_summary)
                if review.requirements_supported_by_evidence:
                    st.markdown("**Requirements supported by the supplied excerpts**")
                    for value in review.requirements_supported_by_evidence:
                        st.markdown(f"- {value}")
                if review.requirements_not_evident_in_excerpts:
                    st.markdown("**Not evident in these excerpts (not proof of absence)**")
                    for value in review.requirements_not_evident_in_excerpts:
                        st.markdown(f"- {value}")
                if review.verification_notes:
                    st.markdown("**Human verification notes**")
                    for value in review.verification_notes:
                        st.markdown(f"- {value}")

    if st.button("Generate evidence-based summaries with Gemini"):
        candidate_payload = []
        for item in results:
            candidate_payload.append({
                "resume_id": item["resume_id"],
                "filename": item["filename"],
                "excerpts": [
                    {"page": e["page"], "text": redact_personal_contacts(e["text"])}
                    for e in item["evidence"]
                ],
            })
        try:
            with st.spinner("Gemini is summarizing the retrieved evidence in one request..."):
                batch = generate_evidence_reviews(st.session_state.resume_job_description, candidate_payload)
            st.session_state.resume_reviews = {review.resume_id: review for review in batch.reviews}
            st.rerun()
        except Exception as exc:
            st.error(f"Could not generate summaries: {exc}")

st.caption("Demo only: no resume is automatically accepted or rejected, and the results should not be used as the sole basis for an employment decision.")
