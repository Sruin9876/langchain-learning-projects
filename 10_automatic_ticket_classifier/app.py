from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ticket_utils import (
    PROJECT_DIR,
    answer_from_policies,
    classify_ticket,
    create_ticket,
    get_policy_vectorstore,
    init_db,
    retrieve_policy_docs,
)

load_dotenv(PROJECT_DIR / ".env", override=False)
st.set_page_config(page_title="Automatic Ticket Classification", page_icon="🎫", layout="wide")
init_db()

st.title("🎫 Automatic Ticket Classification")
st.caption("Ask a workplace question, get guidance from the policy manuals, and route the issue to the right department.")

with st.sidebar:
    st.subheader("How this demo works")
    st.write("1. Search the included HR, IT and transportation policy PDFs.\n2. Classify the issue with a local SVM model trained on the included ticket examples.\n3. Save submitted tickets in a local SQLite database.")
    st.divider()
    st.caption("First run: the local embedding model downloads and the policy index is built. Later queries reuse the cached resources while Streamlit is running.")

issue = st.text_area(
    "Describe your issue or question",
    placeholder="Example: My laptop cannot connect to the office Wi-Fi and I need help.",
    height=120,
)

if st.button("Analyze issue", type="primary", use_container_width=True):
    if not issue.strip():
        st.warning("Enter a question or issue first.")
    elif len(issue.strip()) < 8:
        st.warning("Please add a little more detail so the issue can be classified accurately.")
    else:
        st.session_state["last_issue"] = issue.strip()
        with st.spinner("Loading the local model and searching policy manuals on the first run can take a little while..."):
            try:
                st.session_state["last_department"] = classify_ticket(issue.strip())
                st.session_state["last_docs"] = retrieve_policy_docs(issue.strip(), k=4)
                try:
                    st.session_state["last_answer"] = answer_from_policies(
                        issue.strip(), st.session_state["last_docs"]
                    )
                    st.session_state["answer_error"] = None
                except Exception as exc:
                    st.session_state["last_answer"] = ""
                    st.session_state["answer_error"] = str(exc)
                st.session_state["analysis_error"] = None
            except Exception as exc:
                st.session_state["analysis_error"] = str(exc)

if st.session_state.get("analysis_error"):
    st.error(f"Could not analyze the issue: {st.session_state['analysis_error']}")

if st.session_state.get("last_issue") and st.session_state.get("last_department"):
    st.divider()
    st.subheader("Analysis")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Suggested department", st.session_state["last_department"])
        st.caption("This is a prediction from a sample ML model. Check the department before submitting.")
    with c2:
        st.markdown("**Policy-based guidance**")
        if st.session_state.get("answer_error"):
            st.warning("The ticket was classified, but a policy answer could not be generated.")
            st.caption(st.session_state["answer_error"])
        else:
            st.write(st.session_state.get("last_answer", ""))

    sources = st.session_state.get("last_docs", [])
    if sources:
        with st.expander("View retrieved policy sources"):
            for doc in sources:
                name = doc.metadata.get("source_name", Path(doc.metadata.get("source", "Policy PDF")).name)
                page = int(doc.metadata.get("page", 0)) + 1
                st.markdown(f"**{name} — page {page}**")
                st.write(doc.page_content[:700] + ("…" if len(doc.page_content) > 700 else ""))

    st.info("Submitting stores the ticket locally in this demo. It does not email or notify a real department.")
    if st.button("Submit ticket", type="primary"):
        new_id = create_ticket(
            st.session_state["last_issue"],
            st.session_state["last_department"],
            st.session_state.get("last_answer", "Policy answer unavailable"),
        )
        st.success(f"Ticket #{new_id} saved to the local ticket inbox.")
