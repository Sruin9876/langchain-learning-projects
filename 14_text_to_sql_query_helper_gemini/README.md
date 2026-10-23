# Text to SQL Query Helper (Gemini + LangChain)

A small Streamlit application that turns a plain-English request into a SQL draft. This project modernises a notebook that used a Hugging Face-hosted Llama 2 model and older LangChain `PromptTemplate`/`LLMChain` APIs.

## What it does
- Accepts a natural-language request.
- Lets you select SQLite, PostgreSQL, MySQL, SQL Server, or Standard SQL.
- Optionally accepts a table schema for more accurate queries.
- Uses `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser` to generate SQL.
- Lets you download the generated query as a `.sql` file.

## Safety / limitations
- The application only generates SQL text. It does not connect to a database or execute queries.
- Generated SQL can still be incorrect. Verify the dialect, schema, permissions, and semantics before running it.
- The prompt instructs the model to generate read-only queries by default; this is not a formal SQL security validator.
- Gemini calls require a Google AI Studio API key. No Hugging Face token is needed.

## Setup from `langchain-projects` root (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\14_text_to_sql_query_helper_gemini\requirements.txt
Copy-Item .\14_text_to_sql_query_helper_gemini\.env.example .\14_text_to_sql_query_helper_gemini\.env
notepad .\14_text_to_sql_query_helper_gemini\.env
streamlit run .\14_text_to_sql_query_helper_gemini\app.py
```

Set `GOOGLE_API_KEY` in the project's `.env`. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite` and can be changed in `.env`.

The app loads `.env` relative to its own directory, so it works even when launched from the workspace root.

## Learning points
- `ChatPromptTemplate` builds model messages from variables.
- `ChatGoogleGenerativeAI` connects LangChain to Gemini.
- `StrOutputParser` extracts plain text.
- `prompt | llm | parser` is a LangChain Expression Language (LCEL) chain.
- Supplying a schema usually improves SQL quality because the model can refer to actual table and column names.

This task is text-to-SQL generation, not RAG: there are no embeddings or vector store in this app.
