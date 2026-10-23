# Invoice Extraction Bot (Gemini + LangChain)

A Streamlit app that extracts structured invoice fields from one or more selectable-text PDF invoices and exports them to CSV.

## Features
- Upload PDFs or use the three included synthetic demo invoices. They are marked as fictional and use placeholder contact details.
- Extract invoice number/date, line items, quantities, unit prices, line amounts, total, currency symbol, customer contact details and source filename.
- Supports multiple line items by producing one CSV row per line item and repeating invoice-level fields.
- Uses Gemini with a Pydantic schema for structured output; avoids parsing model text with `eval`.
- Uses PyPDF for local text extraction; no vector database or embedding model is required for this invoice extraction task.
- Keeps per-file errors visible and allows other files to finish processing.

## Setup (Windows PowerShell)
Run from your `langchain-projects` workspace root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\13_invoice_extraction_gemini\requirements.txt
Copy-Item .\13_invoice_extraction_gemini\.env.example .\13_invoice_extraction_gemini\.env
notepad .\13_invoice_extraction_gemini\.env
streamlit run .\13_invoice_extraction_gemini\app.py
```

Set `GOOGLE_API_KEY` to your Google AI Studio API key. `GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`; you can change it in `.env` or in the app sidebar.

## Notes and limitations
- Invoice PDFs must contain selectable text. Scanned image-only PDFs need OCR, which is not included.
- Extraction can make mistakes; verify exported rows against the original invoices before using them for bookkeeping.
- Don't commit `.env` or upload it to GitHub.
- Extracted invoice text is sent to Google Gemini for processing. Avoid uploading sensitive invoices unless you are authorised to do so.
