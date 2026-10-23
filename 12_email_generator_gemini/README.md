# Email Generator (Gemini + LangChain)

A small Streamlit app that turns an email brief into a draft. It uses LangChain's modern prompt/chain pattern and Google's Gemini API. It does not send emails; you review and copy/download the draft.

## Requirements

- Python 3.10+
- A Google AI Studio API key: https://aistudio.google.com/api-keys
- Your shared `langchain-projects/.venv` environment (recommended)

## Setup from the `langchain-projects` workspace root (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\12_email_generator_gemini\requirements.txt
Copy-Item .\12_email_generator_gemini\.env.example .\12_email_generator_gemini\.env
notepad .\12_email_generator_gemini\.env
```

Put your actual key on the `GOOGLE_API_KEY=` line. Keep it private and never upload `.env` to GitHub.

Run from the workspace root:

```powershell
streamlit run .\12_email_generator_gemini\app.py
```

`app.py` loads the `.env` file next to itself, so the command can be run from the workspace root. The default model is `gemini-3.5-flash-lite`; set `GEMINI_MODEL` in `.env` to another model ID enabled for your API key if needed.

## How it works

1. Streamlit collects the topic, sender, recipient, tone, desired length, and optional details.
2. `ChatPromptTemplate` combines these inputs with instructions.
3. `prompt | ChatGoogleGenerativeAI | StrOutputParser()` sends the prompt to Gemini and returns plain text.
4. Streamlit displays the draft and lets you download it as a text file.

Only one Gemini call is used per email generation. The API key is loaded from `.env`; no local model files are required.
