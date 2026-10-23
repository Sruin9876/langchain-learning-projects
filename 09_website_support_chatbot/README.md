# Website Support Chatbot — Gemini + LangChain

This is a refreshed version of the original Support Chat Bot project. It takes a website XML sitemap, fetches a bounded number of pages, splits their text, creates local embeddings, indexes them in FAISS, retrieves relevant passages for each user question, and asks Gemini to answer from those passages with source links.

## What changed from the original

- Removed Pinecone: local FAISS is used, so no Pinecone account/API key is needed.
- Removed the unused Hugging Face API key: `sentence-transformers/all-MiniLM-L6-v2` runs locally.
- Added Gemini as the answer-generation model. The original app only displayed retrieved page snippets; this version generates a grounded answer.
- Uses current modular LangChain packages (`langchain_core`, `langchain_text_splitters`, `langchain_community`, `langchain_huggingface`, `langchain_google_genai`) and direct `vector_store.similarity_search()` + `model.invoke()` calls rather than older QA/memory chains.
- Loads this project's `.env` by absolute path, so launching Streamlit from the workspace root works.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- A Google AI Studio API key for generated answers
- Internet access to fetch sitemap pages and to make Gemini requests

## Setup from the `langchain-projects` workspace root (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\09_website_support_chatbot\requirements.txt
Copy-Item .\09_website_support_chatbot\.env.example .\09_website_support_chatbot\.env
notepad .\09_website_support_chatbot\.env
streamlit run .\09_website_support_chatbot\app.py
```

Put your key on the `GOOGLE_API_KEY=` line. Keep `.env` private and do not commit it.

## Use

1. Paste a website XML sitemap URL in the sidebar. The default is the sitemap used by the original project.
2. Choose the maximum number of pages to crawl, then click **Load / re-index website**.
3. After indexing finishes, ask a question in the chat input.
4. Expand **Sources used** below an answer to visit the pages that supplied retrieved passages.

The first run may take a little longer because the local embedding model is downloaded. Indexing duration depends on the number and speed of the website pages. The vector index is kept in Streamlit session state for the current session; re-index after restarting the app.

## Important limitations

- Input must be an XML sitemap or sitemap index, not just a homepage URL.
- The app crawls only up to the configured maximum number of pages. Increase it if the relevant content is not included.
- Websites may block requests, require JavaScript to render content, or publish PDFs/non-HTML content that this simple scraper skips.
- Answers are grounded in retrieved text but should still be checked, especially for policies, pricing, or time-sensitive support information.
- The embedding model runs locally; Gemini is used only for answer generation.
