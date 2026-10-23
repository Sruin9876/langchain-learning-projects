from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

st.set_page_config(page_title="Text to SQL Helper", page_icon="🧮", layout="wide")
st.title("Text to SQL Query Helper")
st.caption("Describe what you want from your data and generate a SQL draft with Gemini + LangChain.")

with st.sidebar:
    st.subheader("SQL settings")
    dialect = st.selectbox(
        "SQL dialect",
        ["SQLite", "PostgreSQL", "MySQL", "Microsoft SQL Server", "Standard SQL"],
        index=0,
        help="SQL syntax differs slightly between database systems.",
    )
    model_name = st.text_input("Gemini model", value=DEFAULT_MODEL)
    st.caption("The model setting is read from GEMINI_MODEL in .env by default.")

st.warning(
    "This app only generates SQL text. It does not connect to or execute queries against a database. "
    "Review the SQL and verify table/column names before running it."
)

col1, col2 = st.columns([1, 1])
with col1:
    request = st.text_area(
        "Describe the query in plain English",
        value='Extract all the unique values from column "age".',
        height=140,
        placeholder="Example: Find the average salary for each department, highest first.",
    )
with col2:
    schema = st.text_area(
        "Table schema (recommended)",
        value="",
        height=140,
        placeholder="Example:\nemployees(id INTEGER, name TEXT, age INTEGER, department TEXT, salary REAL)",
        help="Provide table and column names/types so the generated SQL matches your database. You can leave this blank, but results may be less accurate.",
    )

extra_instructions = st.text_input(
    "Optional constraints",
    placeholder="Example: Return at most 20 rows; use aliases; include NULL values.",
)

SYSTEM_PROMPT = """You are a careful text-to-SQL assistant.
Convert the user's plain-language request into one SQL query for the specified SQL dialect.
Return only the SQL query, without Markdown code fences or explanatory prose.
Use the provided schema when present. Do not invent tables or columns that contradict the schema.
If the request is ambiguous or cannot be answered from the available schema, return SQL comments explaining the missing information instead of fabricating a query.
By default, generate a read-only query (SELECT or WITH ... SELECT). Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, or other data-changing statements.
Do not execute SQL or claim that it has been executed.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "SQL dialect: {dialect}\n\nTable schema:\n{schema}\n\nUser request:\n{request}\n\nAdditional constraints:\n{extra_instructions}",
        ),
    ]
)

if st.button("Generate SQL", type="primary", disabled=not request.strip(), use_container_width=False):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Google API key not found. Add GOOGLE_API_KEY=your_key to the project's .env file, "
            "then restart Streamlit."
        )
    elif not model_name.strip():
        st.error("Enter a Gemini model name.")
    else:
        try:
            with st.spinner("Generating SQL..."):
                llm = ChatGoogleGenerativeAI(model=model_name.strip(), google_api_key=api_key)
                chain = prompt | llm | StrOutputParser()
                sql = chain.invoke(
                    {
                        "dialect": dialect,
                        "schema": schema.strip() or "No schema provided. Use only table and column names explicitly stated in the user's request; if necessary information is missing, return SQL comments describing what is needed.",
                        "request": request.strip(),
                        "extra_instructions": extra_instructions.strip() or "None",
                    }
                ).strip()
            # Avoid leaving enclosing code fences on display if a provider returns them anyway.
            if sql.startswith("```\n") or sql.startswith("```sql") or sql.startswith("```SQL"):
                lines = sql.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                sql = "\n".join(lines).strip()
            st.session_state["generated_sql"] = sql
            st.session_state["generated_dialect"] = dialect
        except Exception as exc:
            st.error(f"SQL generation failed: {exc}")

if st.session_state.get("generated_sql"):
    st.subheader("Generated SQL")
    st.caption(f"Dialect: {st.session_state.get('generated_dialect', dialect)}")
    st.code(st.session_state["generated_sql"], language="sql")
    st.download_button(
        "Download SQL",
        data=st.session_state["generated_sql"],
        file_name="generated_query.sql",
        mime="text/sql",
    )
    st.info("Check table names, column names, permissions, and query behaviour before running this SQL against a database.")

with st.expander("How this app works"):
    st.markdown(
        "1. `ChatPromptTemplate` combines the SQL dialect, schema and request.\n"
        "2. `ChatGoogleGenerativeAI` sends the prompt to Gemini.\n"
        "3. `StrOutputParser` returns the model response as plain text.\n\n"
        "The chain uses LangChain Expression Language: `prompt | llm | StrOutputParser()`. "
        "This app does not use embeddings, a vector database, or a live SQL connection."
    )
