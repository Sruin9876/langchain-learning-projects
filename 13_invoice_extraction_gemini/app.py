from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils import build_extraction_chain, extract_invoice, rows_to_dataframe

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

st.set_page_config(page_title="Invoice Extraction Bot", page_icon="🧾", layout="wide")
st.title("🧾 Invoice Extraction Bot")
st.caption("Extract structured invoice data from selectable-text PDFs and export it to CSV.")

with st.sidebar:
    st.header("Configuration")
    default_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    model_name = st.text_input("Gemini model", value=default_model, help="Defaults to GEMINI_MODEL in your .env file.")
    st.caption("PDFs are parsed locally. Extracted text is sent to Gemini for structured extraction.")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Add your Google AI Studio API key to this project's .env file before extracting data.")

source = st.radio("Choose invoices", ["Upload PDF files", "Use included sample invoices"], horizontal=True)
files: list[tuple[str, bytes]] = []
if source == "Upload PDF files":
    uploads = st.file_uploader("Upload one or more invoice PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads:
        files = [(f.name, f.getvalue()) for f in uploads]
else:
    sample_dir = APP_DIR / "sample_invoices"
    sample_paths = sorted(sample_dir.glob("*.pdf"))
    selected_names = st.multiselect("Select sample invoices", [p.name for p in sample_paths], default=[p.name for p in sample_paths])
    for path in sample_paths:
        if path.name in selected_names:
            files.append((path.name, path.read_bytes()))
    if not sample_paths:
        st.error("No sample invoices are included in this folder.")

st.info("Tip: Start with one sample invoice. Scanned/image-only PDFs need OCR, which this version does not include.")

if st.button("Extract invoice data", type="primary", disabled=not files or not os.getenv("GOOGLE_API_KEY")):
    all_rows = []
    errors = []
    progress = st.progress(0)
    status = st.empty()
    try:
        chain = build_extraction_chain(model_name.strip())
    except Exception as e:
        st.error(f"Could not initialise the Gemini model: {e}")
        st.stop()

    for i, (name, data) in enumerate(files, start=1):
        status.write(f"Processing {i}/{len(files)}: {name}")
        try:
            rows, page_count, _text = extract_invoice(chain, data, name)
            all_rows.extend(rows)
            st.caption(f"✓ {name}: processed {page_count} page(s).")
        except Exception as e:
            errors.append({"file": name, "error": str(e)})
        progress.progress(i / len(files))

    status.empty()
    if all_rows:
        df = pd.DataFrame(all_rows)
        st.subheader("Extracted data")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
            file_name="invoice_extraction.csv", mime="text/csv", key="download_invoice_csv"
        )
        st.success(f"Extracted {len(all_rows)} line-item row(s) from {len(files) - len(errors)} invoice file(s).")
    if errors:
        st.subheader("Files needing attention")
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
    if not all_rows and not errors:
        st.warning("No invoice data was extracted.")

with st.expander("How extraction works"):
    st.markdown(
        "1. **PyPDF** extracts selectable text locally.\n"
        "2. **ChatPromptTemplate** supplies extraction instructions and the invoice text.\n"
        "3. **Gemini structured output + Pydantic** returns predictable fields instead of a string parsed with `eval`.\n"
        "4. **Pandas** assembles the rows and creates the CSV download."
    )
    st.warning("Verify totals, dates, contact details, and line items against the source PDF before using exported data for accounting. This is an extraction aid, not accounting software.")
