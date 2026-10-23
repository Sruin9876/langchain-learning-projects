# Similar Words Finder

A Streamlit app that finds semantically related entries from `myData.csv` using Gemini embeddings and a FAISS vector index.

## Run it with your shared environment (Windows PowerShell)

From the `langchain-projects` parent folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\03_similar_words_finder\requirements.txt
Copy-Item .\03_similar_words_finder\.env.example .\03_similar_words_finder\.env
notepad .\03_similar_words_finder\.env
streamlit run .\03_similar_words_finder\app.py
```

In the `.env` file, replace the example value with your Google AI Studio API key:

```env
GOOGLE_API_KEY=your_real_key_here
```

Keep `.env` private; never upload it to GitHub or include it in a ZIP you share.

## How it works

1. Read the words from the CSV.
2. Convert each word into an embedding vector using `gemini-embedding-001`.
3. Store those vectors in a FAISS index.
4. Embed the user's query and retrieve the nearest entries.

This uses an embedding model, not a chat model. There is no system prompt and no generated answer: it retrieves existing items from the CSV. The app reports FAISS vector distances, not probability or a guaranteed similarity percentage.

## Experiment ideas

- Add more words and categories to `myData.csv`.
- Try queries such as `wild animal`, `fruit`, and `sport`.
- Change the number of matches using the slider.
- Replace the word list with book titles, products, courses, or FAQs.
- Compare results from a cloud embedding model with a local embedding model later.
