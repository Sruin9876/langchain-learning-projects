# Question Answering App

A minimal Streamlit app that sends a question to a Gemini chat model through LangChain and displays the answer.

## Setup (Windows PowerShell)

Run these commands from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r ".\01_Q-A Application\requirements.txt"
Copy-Item ".\01_Q-A Application\.env.example" ".\01_Q-A Application\.env"
notepad ".\01_Q-A Application\.env"
streamlit run ".\01_Q-A Application\app.py"
```

Put your Google AI Studio API key in `GOOGLE_API_KEY`. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`; change it to a model available to your API key if needed.

## Learning points

- `ChatGoogleGenerativeAI` connects LangChain to Gemini.
- `.invoke()` sends the question and returns a model response.
- `python-dotenv` loads local configuration from `.env`.

