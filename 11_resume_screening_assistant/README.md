# Resume Screening Assistant (Learning Project)

A human-in-the-loop resume exploration demo. It compares resume text to a job description using local sentence embeddings and FAISS, shows the passages that drove retrieval, and optionally asks Gemini for concise evidence-based summaries.

## Important use limitation

This is an educational decision-support tool, not an automated hiring system. Similarity is not a calibrated qualification score or a prediction of job performance. Never use the app to automatically accept/reject candidates or as the sole basis for an employment decision. Apply consistent, job-related criteria and review original resumes. Do not use protected/personal characteristics in job criteria.

## Stack

- Streamlit UI
- PyPDF for text extraction
- `RecursiveCharacterTextSplitter` for chunking
- `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) running locally
- FAISS for nearest-vector retrieval
- `ChatGoogleGenerativeAI` + `ChatPromptTemplate` + structured output for optional summaries

## Setup on Windows (PowerShell)

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\11_resume_screening_assistant\requirements.txt
Copy-Item .\11_resume_screening_assistant\.env.example .\11_resume_screening_assistant\.env
notepad .\11_resume_screening_assistant\.env
```

Put your Google AI Studio key after `GOOGLE_API_KEY=` and save. Optional Gemini summaries require this key; local extraction, embeddings and retrieval do not require a Gemini key.

Run:

```powershell
streamlit run .\11_resume_screening_assistant\app.py
```

Upload PDFs and paste a job description, then choose **Analyze resumes**. After results appear, click **Generate evidence-based summaries with Gemini** if you want grounded summaries. Only selected text excerpts, not entire PDF files, are sent to Gemini; emails and phone numbers in those excerpts are redacted before sending.

## Sample resumes

The `sample_resumes` folder contains seven synthetic PDF resumes created for testing this repository. The names and contact details are placeholders, not real candidate data. Upload a few through the browser file picker.

## Notes

- First run downloads the embedding model; this can take a few minutes.
- PDF text extraction works only if text is selectable. Scanned image-only PDFs require OCR.
- Matching is based on embedding similarity. Evidence selection can be imperfect; always inspect the underlying documents.
- The app uses CPU embeddings for broad compatibility. The FAISS index is built per analysis and kept in Streamlit session state, not uploaded to a cloud vector database.
