# MCQ Creator — Gemini + LangChain + local FAISS

Upload one or more text-based PDFs, index their text locally, retrieve passages relevant to a topic, and generate structured multiple-choice questions with Gemini. Export the questions as JSON or CSV.

## Architecture

1. **Load:** `PyPDFLoader` extracts PDF pages.
2. **Split:** `RecursiveCharacterTextSplitter` breaks pages into overlapping chunks.
3. **Embed:** `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` converts chunks and queries to vectors locally.
4. **Index/retrieve:** FAISS searches for the chunks most similar to the topic.
5. **Generate:** `ChatGoogleGenerativeAI` with `with_structured_output(MCQSet)` asks Gemini to return validated, structured question data.
6. **Display/export:** Streamlit renders questions and allows JSON/CSV downloads.

Embeddings do not use a cloud API key; MCQ generation uses a Google AI Studio API key. The first indexing run downloads the embedding model and may take a little while.

## Setup on Windows / VS Code

Run commands from your existing `langchain-projects` workspace root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\06_mcq_creator_gemini\requirements.txt
Copy-Item .\06_mcq_creator_gemini\.env.example .\06_mcq_creator_gemini\.env
notepad .\06_mcq_creator_gemini\.env
streamlit run .\06_mcq_creator_gemini\app.py
```

Add your Google AI Studio key to `GOOGLE_API_KEY` in the project-local `.env`. Leave `GEMINI_MODEL=gemini-3.5-flash-lite` or change it to a model available to your API key. The code explicitly loads `.env` from this project folder, even when launched from the workspace root.

You can instead paste a key into the sidebar for one run; it isn't written to disk by the app.

## Try the included samples

The `sample_docs` folder includes the PDFs provided alongside the original notebook. In Streamlit's upload control, select those PDFs from this folder. Then click **Process PDFs**, provide a topic, choose difficulty and question count, and click **Generate MCQs**.

## Notes

- This is a small RAG-style workflow: retrieval selects relevant source chunks, and the LLM writes questions based on those chunks.
- Local FAISS indexes the documents in the current Streamlit session. Reloading the session or restarting the app means you should process the PDFs again.
- Text extraction will not work well on scanned/image-only PDFs without OCR.
- Review the generated questions and answer key before educational or assessment use.
- Keep `.env` secret and do not commit it. The original notebook contained API-key-looking strings; they have been removed from this project.
