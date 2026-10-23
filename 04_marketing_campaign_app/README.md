# Marketing Campaign Generator

A Streamlit learning project demonstrating LangChain prompt templates, few-shot prompting, and a length-based example selector. This version uses Gemini rather than OpenAI.

## Setup in your existing workspace

From the `langchain-projects` folder, activate your shared environment and install the project dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\04_marketing_campaign_app\requirements.txt
Copy-Item .\04_marketing_campaign_app\.env.example .\04_marketing_campaign_app\.env
notepad .\04_marketing_campaign_app\.env
```

Put your Google AI Studio API key in `GOOGLE_API_KEY`, save the file, and run:

```powershell
streamlit run .\04_marketing_campaign_app\app.py
```

You can optionally change `GEMINI_MODEL` in `.env` to a model enabled for your API key.

## What to learn

- `PromptTemplate`: describes the format for one example.
- `FewShotPromptTemplate`: combines examples with instructions and the user's request.
- `LengthBasedExampleSelector`: chooses examples that fit an approximate length budget.
- `ChatGoogleGenerativeAI`: sends the final prompt to Gemini.
- Streamlit controls: collect a campaign brief, task, audience, and output-length limit.

The maximum word count is an instruction to the model, not a guaranteed hard limit.

## Security

Never commit `.env` or share it. Use `.env.example` with placeholder values in GitHub. If an API key was included in an uploaded or shared archive, revoke it and create a replacement.
