# Customer Care Call Summary & Email Alert

A learning project based on the supplied video transcript. It accepts multiple customer-care call recordings, transcribes them locally, asks Gemini to prepare a factual call summary plus a customer-facing email draft, and can optionally send the reviewed email over SMTP.

## What is included

- `app.py`: Streamlit interface for audio uploads, pasted transcripts, review, summaries, draft edits, download, and optional email sending.
- `utils.py`: local Whisper transcription, a modern LangChain prompt-to-structured-output chain, and SMTP email delivery.
- `sample_call_transcript.txt`: fictional transcript for testing the summary workflow without audio transcription.
- `.env.example`: configuration template; it contains no real credentials.

## How the pipeline works

1. Upload an MP3/WAV/M4A/etc. recording or paste a transcript.
2. `faster-whisper` transcribes uploaded audio locally. The first run downloads the selected model (default: `base`). It uses CPU/int8 for compatibility.
3. `ChatPromptTemplate | ChatGoogleGenerativeAI.with_structured_output(...)` returns a structured summary, key actions, sentiment, follow-up indicator, email subject, and email body.
4. Review and edit the generated email. Sending is never automatic.
5. If SMTP is configured, the app sends the email only when the user clicks **Send email**. Otherwise, the draft can still be downloaded.

## Setup in the shared workspace

From the root of your `langchain-projects` workspace in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\15_customer_care_call_summary\requirements.txt
Copy-Item .\15_customer_care_call_summary\.env.example .\15_customer_care_call_summary\.env
notepad .\15_customer_care_call_summary\.env
```

Set `GOOGLE_API_KEY` to a key from Google AI Studio. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`; change it if that model is not available to your API key.

Then run:

```powershell
streamlit run .\15_customer_care_call_summary\app.py
```

The app loads `.env` from its own project folder even when launched from the workspace root.

## Optional email delivery

Without SMTP configuration, you can still generate and download drafts. To send actual email, fill these in `.env`:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_gmail_address@gmail.com
SMTP_PASSWORD=your_google_app_password
SENDER_EMAIL=your_gmail_address@gmail.com
```

For Gmail, use a Google App Password if your account supports it; do not put your normal Google password in the file. Keep `.env` private and out of Git.

The supplied video used Zapier's email action. This updated version uses standard authenticated SMTP for the optional send step to avoid relying on the older Zapier/LangChain agent integration. The app does not infer a recipient from an audio transcript; enter and confirm the customer's email before sending. A real deployment should retrieve recipient addresses from an authorised customer database.

## Important limitations

- Transcription may be slow on long recordings, especially on CPU. `tiny` is faster; `base` is the default balance.
- The first Whisper run downloads a model from the internet.
- Gemini receives the transcript when you request the summary/email; don't upload calls unless you are authorised to process them.
- AI may mishear or mis-summarise details. Review the transcript and email before sending.
- Sentiment is an approximate signal, not a ground-truth label or a reason by itself to change how a customer is treated.
- SMTP details, network access, and live Gemini credentials are not provided in the ZIP. The app must be tested with your own setup.
