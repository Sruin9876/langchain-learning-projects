import pandas as pd
import streamlit as st
from ticket_utils import DATASET_PATH, get_ticket_classifier

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
st.title("📊 Classifier Insights")
st.write("This page shows how the sample SVM classifier performs on a held-out split of the included labelled ticket examples.")

with st.spinner("Loading local embeddings and training the sample classifier..."):
    try:
        bundle = get_ticket_classifier()
    except Exception as exc:
        st.error(f"Could not train the classifier: {exc}")
        st.stop()

a, b = st.columns(2)
a.metric("Sample ticket examples", bundle["sample_count"])
b.metric("Holdout accuracy", f"{bundle['accuracy']:.1%}")
st.caption("Accuracy is measured on one train/test split of a small sample dataset; it is not a guarantee of real-world performance.")

st.subheader("Examples by department")
counts = pd.DataFrame({"Department": list(bundle["counts"].keys()), "Examples": list(bundle["counts"].values())})
st.bar_chart(counts.set_index("Department"))
st.subheader("Dataset preview")
df = pd.read_csv(DATASET_PATH, header=None, names=["Ticket text", "Department"])
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Per-class metrics")
report_rows = []
for label in bundle["labels"]:
    metrics = bundle["report"].get(label, {})
    report_rows.append({
        "Department": label,
        "Precision": metrics.get("precision", 0.0),
        "Recall": metrics.get("recall", 0.0),
        "F1 score": metrics.get("f1-score", 0.0),
        "Test samples": int(metrics.get("support", 0)),
    })
st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
