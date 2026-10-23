# YouTube Script Writing Tool (Gemini + LangChain)

A Streamlit app that researches a topic with DuckDuckGo, generates a suggested video title, and drafts a YouTube script with Gemini.

## Setup (Windows / PowerShell)

Run these commands from your `langchain-projects` workspace root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\08_youtube_script_writer_gemini\requirements.txt
Copy-Item .\08_youtube_script_writer_gemini\.env.example .\08_youtube_script_writer_gemini\.env
notepad .\08_youtube_script_writer_gemini\.env
```

Put your Google AI Studio key after `GOOGLE_API_KEY=`. Keep the key private and do not commit `.env`.

Launch the app:

```powershell
streamlit run .\08_youtube_script_writer_gemini\app.py
```

## How it works

1. Streamlit collects the topic, target audience, length, tone, language, and creative style.
2. `DuckDuckGoSearchRun.invoke()` fetches public search snippets when web research is enabled. Search failure is handled gracefully.
3. `ChatPromptTemplate` builds the prompts.
4. Modern LangChain Expression Language (LCEL) chains use `prompt | llm | StrOutputParser()` to generate a title and then a script.
5. The script is displayed and can be downloaded as Markdown.

The model is configured by `GEMINI_MODEL`, defaulting to `gemini-3.5-flash-lite`. If your account cannot access this model, use a model ID currently available to your Google AI Studio API key.

## Learning notes

- This is a two-step LLM workflow (title, then script), not an autonomous agent.
- DuckDuckGo supplies search snippets; there is no RAG/vector database in this project.
- The creativity slider is translated into prompt guidance instead of sending an explicit temperature parameter, to avoid relying on sampling settings that may be unsupported by some Gemini model families.
- The length is a target, not a guarantee. Spoken scripts often vary from about 120–150 words per minute.

## Useful variations

- Generate 3 alternative titles and let the user choose one.
- Add separate modes for Shorts, tutorials, reviews, and storytelling.
- Add a target language or regional style.
- Add a chapter/timestamp outline before the full script.
- Save title, script, topic, and generation settings to a history file or database.
- Add source URLs as structured search results so viewers can check key claims.
