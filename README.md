# LangChain & LLM Learning Projects

A collection of 15 mini-projects built while learning through a [Coursera course (share link)](https://coursera.org/share/7adc4b512f8df1366a0489e6dddcb248). The projects are organised in one repository so the code, experiments, and lessons can be explored together.

Many projects are adapted or reimplemented from course demonstrations and source material, then adjusted for learning with Gemini, current LangChain APIs, local embeddings, or FAISS where appropriate. This is a learning portfolio, not a set of production-ready applications.

## Project index

| # | Project | What it explores | Main concepts |
|---:|---|---|---|
| 01 | [Question Answering App](./01_Q-A%20Application/README.md) | Ask a question and receive an LLM answer | Gemini, LangChain, Streamlit |
| 02 | [Conversational Chatbot](./02_conversational_chatbot/README.md) | Chat with session-level conversation history | Messages, session state, system prompts |
| 03 | [Similar Words Finder](./03_similar_words_finder/README.md) | Find semantically related entries from a CSV | Embeddings, FAISS, similarity search |
| 04 | [Marketing Campaign Generator](./04_marketing_campaign_app/README.md) | Generate marketing copy for different tasks and audiences | Few-shot prompting, example selection |
| 05 | [ChatGPT Clone](./05_chatgpt_clone_gemini/README.md) | Chat interface with editable instructions and conversation summarisation | Chat history, context management |
| 06 | [MCQ Creator](./06_mcq_creator_gemini/README.md) | Generate structured questions from PDFs | Chunking, retrieval, structured output, FAISS |
| 07 | [CSV Data Analyst](./07_csv_data_analysis/README.md) | Ask natural-language questions about CSV data | Pandas, tool calling, LangChain agents |
| 08 | [YouTube Script Writer](./08_youtube_script_writer_gemini/README.md) | Research a topic and draft a video script | Web search, prompt chains, LCEL |
| 09 | [Website Support Chatbot](./09_website_support_chatbot/README.md) | Answer questions using retrieved website content | Web loading, RAG, embeddings, FAISS |
| 10 | [Automatic Ticket Classifier](./10_automatic_ticket_classifier/README.md) | Classify support tickets and retrieve policy passages | SVM, local embeddings, RAG, SQLite |
| 11 | [Resume Screening Assistant](./11_resume_screening_assistant/README.md) | Compare resume evidence with a job description | PDF extraction, vector search, summarisation |
| 12 | [Email Generator](./12_email_generator_gemini/README.md) | Draft emails from a brief and desired tone | Prompt templates, LCEL, text generation |
| 13 | [Invoice Extraction Bot](./13_invoice_extraction_gemini/README.md) | Extract invoice fields into structured rows | PDF text extraction, Pydantic, CSV export |
| 14 | [Text-to-SQL Query Helper](./14_text_to_sql_query_helper_gemini/README.md) | Generate SQL from plain-English requests | Prompting, schema grounding, LCEL |
| 15 | [Customer Care Call Summarizer](./15_customer_care_call_summary/README.md) | Transcribe calls, draft summaries and optional customer emails | Whisper, summarisation, SMTP, structured output |

## Technology overview

- **LLM provider:** Gemini via `langchain-google-genai` for generation tasks.
- **Application UI:** Streamlit.
- **Prompting and chains:** LangChain prompt templates, message objects, and LCEL (`prompt | model | parser`).
- **Retrieval projects:** local Sentence Transformers embeddings and FAISS where used; other projects do not need embeddings.
- **Data tools:** pandas, scikit-learn, SQLite, PyPDF, and Faster-Whisper depending on the project.

The default chat model used by most projects is `gemini-3.5-flash-lite`; confirm that it is available to your Google AI Studio API key, or set `GEMINI_MODEL` in that project's `.env` to another supported model. Similar Words Finder uses the separate embedding model `gemini-embedding-001`.

## Setup (Windows / PowerShell)

### Option A: install the combined requirements

This installs dependencies used across all 15 projects, including local embedding and audio packages. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation on your machine, use the Python executable directly at `.venv\Scripts\python.exe`, or apply an appropriate user-scoped execution policy.

### Option B: install only one project's dependencies

Create/activate a virtual environment, then install the selected project's `requirements.txt`; for example:

```powershell
python -m pip install -r .\06_mcq_creator_gemini\requirements.txt
```

### Configure an individual project

Projects that call Gemini have a `.env.example` template. Copy it to `.env` in the same project folder and put your key in `GOOGLE_API_KEY`:

```powershell
Copy-Item .\06_mcq_creator_gemini\.env.example .\06_mcq_creator_gemini\.env
notepad .\06_mcq_creator_gemini\.env
```

Keep each project's real `.env` out of Git. Never paste API keys into source code or commit history. Some projects use only local processing until you request a Gemini-generated answer; consult that project's README for the exact requirements.

### Run a project

From the repository root, use the path shown in the project table. For example:

```powershell
streamlit run .\06_mcq_creator_gemini\app.py
```

Each project README includes its own setup steps, workflow, and limitations. For the folder `01_Q-A Application`, quote the path because it contains a space:

```powershell
streamlit run ".\01_Q-A Application\app.py"
```

## Course context and attribution

These projects were assembled as hands-on practice while following the [Coursera course/share link provided for this learning journey](https://coursera.org/share/7adc4b512f8df1366a0489e6dddcb248). Some projects are based on course examples or source material and have been modified for practice. 
