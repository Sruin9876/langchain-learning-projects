from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils import email_configured, generate_call_email, send_email, transcribe_audio_bytes


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env", override=False)

st.set_page_config(page_title="Customer Care Call Summary", page_icon="📞", layout="wide")
st.title("📞 Customer Care Call Summary & Email Alert")
st.write(
    "Transcribe customer-care recordings, prepare an evidence-based call summary, "
    "and draft a follow-up email for review."
)
st.warning(
    "Privacy reminder: call recordings can contain personal or payment information. "
    "Use recordings you are authorised to process. Transcription runs locally; the transcript is sent to Gemini "
    "when you generate the summary/email. Emails are sent only after you explicitly click Send email."
)

if "call_records" not in st.session_state:
    st.session_state.call_records = {}

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small"],
        index=1,
        help="tiny is fastest; base is a reasonable balance; small may be more accurate but slower.",
    )
    st.caption(f"Gemini model: {os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')}")
    if email_configured():
        st.success("SMTP email sending is configured.")
    else:
        st.info("Email sending is not configured yet. You can still create and download drafts.")
    if st.button("Clear processed calls", use_container_width=True):
        st.session_state.call_records = {}
        st.rerun()


def add_record(record_id: str, record: dict) -> None:
    st.session_state.call_records[record_id] = record


tab_audio, tab_transcript = st.tabs(["Upload call recordings", "Paste a transcript"])

with tab_audio:
    uploaded_files = st.file_uploader(
        "Upload one or more call recordings",
        type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "ogg", "webm", "flac"],
        accept_multiple_files=True,
        help="Transcription is performed locally using faster-whisper. The first use downloads the selected model.",
    )
    if st.button("Transcribe uploaded recordings", type="primary", disabled=not uploaded_files):
        progress = st.progress(0, text="Preparing transcription...")
        completed = 0
        failures: list[str] = []
        for index, uploaded in enumerate(uploaded_files):
            data = uploaded.getvalue()
            record_id = hashlib.sha256(data).hexdigest()[:16]
            if record_id in st.session_state.call_records:
                completed += 1
                progress.progress(completed / len(uploaded_files), text=f"Already processed: {uploaded.name}")
                continue
            try:
                with st.spinner(f"Transcribing {uploaded.name}..."):
                    transcript, language = transcribe_audio_bytes(
                        data, Path(uploaded.name).suffix.lower(), model_size
                    )
                if not transcript:
                    raise ValueError("No speech was detected in this recording.")
                add_record(
                    record_id,
                    {
                        "filename": uploaded.name,
                        "transcript": transcript,
                        "language": language or "unknown",
                        "source": "audio",
                        "draft": None,
                        "sent": False,
                    },
                )
            except Exception as exc:
                failures.append(f"{uploaded.name}: {exc}")
            completed += 1
            progress.progress(completed / len(uploaded_files), text=f"Processed {completed} of {len(uploaded_files)}")
        progress.empty()
        if failures:
            for failure in failures:
                st.error(failure)
        if completed and not failures:
            st.success("Transcription step finished.")

with tab_transcript:
    st.caption("Use this option to test the summary/email workflow without waiting for speech-to-text.")
    pasted_name = st.text_input("Call label", value="Pasted call transcript", key="pasted_call_name")
    pasted_transcript = st.text_area("Paste call transcript", height=180, key="pasted_transcript")
    if st.button("Use pasted transcript", disabled=not pasted_transcript.strip()):
        record_id = hashlib.sha256(pasted_transcript.strip().encode("utf-8")).hexdigest()[:16]
        add_record(
            record_id,
            {
                "filename": pasted_name.strip() or "Pasted call transcript",
                "transcript": pasted_transcript.strip(),
                "language": "not detected (text input)",
                "source": "text",
                "draft": None,
                "sent": False,
            },
        )
        st.success("Transcript added below.")

if st.session_state.call_records:
    st.divider()
    st.header("Processed calls")
    st.caption("Review the transcript, provide/confirm the recipient, and generate the customer email. No email is sent automatically.")

    for record_id, record in list(st.session_state.call_records.items()):
        with st.container(border=True):
            st.subheader(record["filename"])
            st.caption(f"Source: {record['source']} · Detected language: {record.get('language', 'unknown')}")
            with st.expander("Review / correct transcript", expanded=False):
                transcript_key = f"transcript_{record_id}"
                if transcript_key not in st.session_state:
                    st.session_state[transcript_key] = record["transcript"]
                st.text_area("Transcript text", key=transcript_key, height=180)

            name_key = f"customer_name_{record_id}"
            recipient_key = f"recipient_{record_id}"
            col_name, col_email = st.columns(2)
            with col_name:
                customer_name = st.text_input("Customer name (optional)", key=name_key)
            with col_email:
                recipient = st.text_input("Customer email (confirm before sending)", key=recipient_key)

            if st.button("Generate summary and email draft", key=f"generate_{record_id}", type="primary"):
                try:
                    with st.spinner("Gemini is preparing the summary and email draft..."):
                        draft = generate_call_email(
                            st.session_state.get(f"transcript_{record_id}", record["transcript"]),
                            customer_name,
                        )
                    record["draft"] = draft.model_dump()
                    record["sent"] = False
                    st.success("Draft generated. Review and edit it before sending.")
                except Exception as exc:
                    st.error(f"Could not generate the draft: {exc}")

            if record.get("draft"):
                draft = record["draft"]
                st.markdown("**Call summary**")
                st.write(draft.get("call_summary", ""))
                sentiment = draft.get("sentiment", "unclear").capitalize()
                needs_followup = bool(draft.get("follow_up_needed", False))
                metric1, metric2 = st.columns(2)
                metric1.metric("Call sentiment", sentiment)
                metric2.metric("Follow-up indicated", "Yes" if needs_followup else "No")
                if draft.get("follow_up_reason"):
                    st.caption(f"Follow-up reason: {draft['follow_up_reason']}")
                actions = draft.get("key_actions", [])
                if actions:
                    with st.expander("Key actions from the call"):
                        for action in actions:
                            st.write(f"- {action}")

                subject_key = f"subject_{record_id}"
                body_key = f"body_{record_id}"
                if subject_key not in st.session_state:
                    st.session_state[subject_key] = draft.get("email_subject", "Follow-up on your customer-care call")
                if body_key not in st.session_state:
                    st.session_state[body_key] = draft.get("email_body", "")
                st.text_input("Email subject (editable)", key=subject_key)
                st.text_area("Email body (editable)", key=body_key, height=220)

                col_send, col_download = st.columns(2)
                with col_send:
                    send_clicked = st.button(
                        "Send email",
                        key=f"send_{record_id}",
                        type="primary",
                        disabled=not email_configured() or not recipient.strip(),
                    )
                with col_download:
                    email_text = f"To: {recipient}\nSubject: {st.session_state[subject_key]}\n\n{st.session_state[body_key]}"
                    st.download_button(
                        "Download email draft (.txt)",
                        data=email_text,
                        file_name=f"{Path(record['filename']).stem}_email_draft.txt",
                        mime="text/plain",
                        key=f"download_{record_id}",
                    )
                if not email_configured():
                    st.caption("To enable sending, configure SMTP settings in this project's .env. Downloading drafts works without SMTP.")
                if send_clicked:
                    try:
                        send_email(
                            recipient,
                            st.session_state[subject_key],
                            st.session_state[body_key],
                        )
                        record["sent"] = True
                        st.success(f"Email sent to {recipient}.")
                    except Exception as exc:
                        st.error(f"Email could not be sent: {exc}")
                if record.get("sent"):
                    st.caption("This call's email was sent during this Streamlit session.")
else:
    st.info("Upload and transcribe a recording, or paste a transcript, to get started.")

st.divider()
st.caption("Prototype only: review transcripts and drafts before sending. Do not use sentiment as the sole basis for customer treatment.")
