# Conversational Chatbot (LangChain + Gemini + Streamlit)

A small learning project that demonstrates:
- Calling a Gemini chat model through LangChain
- Passing conversation history using `HumanMessage`, `AIMessage`, and `SystemMessage`
- Keeping chat messages in Streamlit's `st.session_state`
- Rendering a chat-style interface

## Setup

This project is designed to use the shared virtual environment in the parent `langchain-projects` folder.

1. Copy this project folder into your `langchain-projects` workspace.
2. In this folder, copy `.env.example` to `.env`.
3. Put your Gemini API key in `.env`:

   `GOOGLE_API_KEY=your_actual_key`

4. Open a terminal at the `langchain-projects` workspace root and activate the shared environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

5. Install any missing dependencies:

   ```powershell
   python -m pip install -r .\02_conversational_chatbot\requirements.txt
   ```

6. Run the app from the workspace root:

   ```powershell
   streamlit run .\02_conversational_chatbot\app.py
   ```

## Notes

- The app reads `GEMINI_MODEL` from this project's `.env`; the default is `gemini-3.5-flash-lite`. Change it to a model currently available to your Google AI Studio API key if needed.
- Conversation history is in session memory only. It is not saved to a database and may disappear when the session resets.
- Never commit `.env` or share your API key.
