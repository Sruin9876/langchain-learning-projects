# Automatic Ticket Classification Tool (Gemini + local RAG)

A learning project that combines a small text classifier with retrieval-augmented generation (RAG):

1. A local Sentence Transformers model (`all-MiniLM-L6-v2`) creates embeddings.
2. A Linear SVM predicts whether a ticket belongs to HR, IT, or Transportation.
3. Local FAISS indexes the included policy PDFs and retrieves relevant passages.
4. Gemini uses those passages to draft policy-based guidance.
5. Submitted tickets are saved in a local SQLite database and shown in Ticket Inbox.

No Pinecone or OpenAI account is needed. Gemini is used only for generating the policy answer; the embedding model and classifier run locally. This project sends the submitted issue and retrieved policy snippets to Gemini when generating an answer. Do not use confidential workplace data with a demo API key without approval.

## Included materials

- `Documents/`: sample HR, IT, and transportation policy manuals.
- `data/Tickets.csv`: small labelled sample dataset for training the classifier.
- `pages/1_Ticket_Inbox.py`: view tickets and update status.
- `pages/2_Model_Insights.py`: sample classifier metrics and dataset preview.

The original pre-trained pickle model and API credentials are intentionally excluded. The classifier is trained locally from the included CSV rather than loading an old serialized model.

## Setup (Windows / PowerShell)

Run from your `langchain-projects` workspace root after extracting this folder as `10_automatic_ticket_classifier`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\10_automatic_ticket_classifier\requirements.txt
Copy-Item .\10_automatic_ticket_classifier\.env.example .\10_automatic_ticket_classifier\.env
notepad .\10_automatic_ticket_classifier\.env
```

Add your Google AI Studio API key to `GOOGLE_API_KEY`, save, then run:

```powershell
streamlit run .\10_automatic_ticket_classifier\app.py
```

## First run and performance

The first analysis can take a while because Sentence Transformers downloads `all-MiniLM-L6-v2`, embeds the policy PDFs, and trains the SVM. These resources are cached during the running Streamlit process. If you restart the app, the in-memory policy index is rebuilt. The local model uses CPU by default for compatibility.

## Important limitations

- This is a local demonstration. Submitting a ticket creates/updates a local `tickets.db` file; that runtime database is ignored by Git and is not included in this repository. It does not notify a real support team.
- Classification quality depends on the small sample CSV. Treat the predicted department as a suggestion.
- The policy answer depends on the contents of the included sample manuals and the retrieved chunks. Check the sources shown by the app.
- Only text-based PDFs are supported; scanned PDFs may need OCR.
