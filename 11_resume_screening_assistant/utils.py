from __future__ import annotations

from io import BytesIO
from collections import defaultdict
from typing import Iterable

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_resume_documents(uploaded_files) -> tuple[list[Document], list[str]]:
    """Extract selectable text page-by-page and preserve file/page metadata."""
    documents: list[Document] = []
    warnings: list[str] = []
    for file_index, uploaded in enumerate(uploaded_files, start=1):
        filename = uploaded.name
        resume_id = f"resume_{file_index}"
        try:
            reader = PdfReader(BytesIO(uploaded.getvalue()))
            page_texts = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    page_texts.append(text)
                    documents.append(Document(
                        page_content=text,
                        metadata={"filename": filename, "resume_id": resume_id, "page": page_number},
                    ))
            if not page_texts:
                warnings.append(f"{filename}: no selectable text found; a scanned PDF may need OCR.")
        except Exception as exc:
            warnings.append(f"{filename}: could not read this PDF ({exc}).")
    return documents, warnings


def split_resume_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=140,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Ignore empty/near-empty chunks; preserve source page and resume ID metadata.
    return [chunk for chunk in chunks if len(chunk.page_content.strip()) >= 40]


def rank_resumes(job_description: str, index: FAISS, chunks: list[Document], number_to_show: int) -> list[dict]:
    """Compare every indexed passage, then aggregate the three best chunk similarities per resume.

    FAISS's default flat L2 index returns squared Euclidean distances. With normalized
    embedding vectors, cosine similarity equals 1 - squared_distance / 2.
    """
    if not chunks:
        return []
    scored = index.similarity_search_with_score(job_description, k=len(chunks))
    grouped: dict[str, list[tuple[float, Document]]] = defaultdict(list)
    filename_by_id: dict[str, str] = {}
    for doc, distance in scored:
        resume_id = str(doc.metadata.get("resume_id", "unknown"))
        filename = str(doc.metadata.get("filename", "Unknown file"))
        filename_by_id[resume_id] = filename
        cosine = max(-1.0, min(1.0, 1.0 - float(distance) / 2.0))
        grouped[resume_id].append((cosine, doc))

    ranked = []
    for resume_id, items in grouped.items():
        items.sort(key=lambda pair: pair[0], reverse=True)
        top_for_score = items[: min(3, len(items))]
        score = sum(value for value, _ in top_for_score) / len(top_for_score)
        seen = set()
        evidence = []
        for similarity, doc in items:
            text = " ".join(doc.page_content.split())
            key = (doc.metadata.get("page"), text[:160])
            if key in seen:
                continue
            seen.add(key)
            evidence.append({
                "page": int(doc.metadata.get("page", 1)),
                "text": text[:850] + ("…" if len(text) > 850 else ""),
                "chunk_similarity": similarity,
            })
            if len(evidence) >= 3:
                break
        ranked.append({
            "resume_id": resume_id,
            "filename": filename_by_id.get(resume_id, "Unknown file"),
            "similarity": score,
            "evidence": evidence,
        })
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked[: max(1, int(number_to_show))]
