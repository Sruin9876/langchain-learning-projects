from pathlib import Path
import csv
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "myData.csv"
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="Similar Words Finder", page_icon="🔎")
st.title("🔎 Similar Words Finder")
st.write("Enter a word or short phrase to find semantically related items in the CSV dataset.")

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    st.error("API key not found. Create a .env file in this project folder using .env.example, then add your Google AI Studio API key.")
    st.stop()


def load_words(csv_path: Path) -> list[str]:
    """Read the Words column from the CSV, including files saved with a UTF-8 BOM."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return []
        word_column = next((name for name in reader.fieldnames if name.strip().lower() == "words"), reader.fieldnames[0])
        return [row.get(word_column, "").strip() for row in reader if row.get(word_column, "").strip()]


@st.cache_resource
def build_vector_store(csv_path: str):
    words = load_words(Path(csv_path))
    if not words:
        raise ValueError("The CSV contains no words. Add entries under the 'Words' column.")

    documents = [Document(page_content=word, metadata={"word": word}) for word in words]
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store, len(words)


if not CSV_PATH.exists():
    st.error(f"Dataset not found: {CSV_PATH.name}. Keep myData.csv in this project folder.")
    st.stop()

try:
    vector_store, number_of_words = build_vector_store(str(CSV_PATH))
except Exception as exc:
    st.error("Could not create the vector index. Check your API key, internet connection, installed packages, and CSV file.")
    st.exception(exc)
    st.stop()

st.caption(f"Dataset loaded: {number_of_words} words")
query = st.text_input("What are you looking for?", placeholder="e.g. wild animal, fruit, sport")
top_k = st.slider("Number of matches", min_value=1, max_value=min(5, number_of_words), value=min(3, number_of_words))

if st.button("Find similar items", type="primary"):
    if not query.strip():
        st.warning("Enter a word or phrase first.")
    else:
        try:
            results = vector_store.similarity_search_with_score(query.strip(), k=min(top_k, number_of_words))
            st.subheader("Closest matches")
            for rank, (document, distance) in enumerate(results, start=1):
                st.markdown(f"**{rank}. {document.page_content}**")
                st.caption(f"Vector distance: {distance:.4f} · Lower usually means closer in this index.")
            st.info("These are semantic matches from your CSV, not guaranteed dictionary synonyms. Results depend on the dataset and embedding model.")
        except Exception as exc:
            st.error("Search failed. Check your API key, quota, and internet connection.")
            st.exception(exc)

with st.expander("What happens behind the scenes?"):
    st.markdown(
        "1. The app reads words from `myData.csv`.\n"
        "2. Gemini Embeddings converts each word into a vector (a list of numbers).\n"
        "3. FAISS indexes those vectors for fast nearest-neighbour search.\n"
        "4. Your query is embedded, and the app returns the closest stored entries."
    )
