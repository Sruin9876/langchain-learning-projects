# Chatbot Clone — Gemini + current LangChain message API

A small Streamlit chatbot based on the uploaded ChatGPT Clone project, updated to use Gemini instead of OpenAI.

## Features

- Streamlit chat UI (`st.chat_message` and `st.chat_input`)
- Gemini through `ChatGoogleGenerativeAI`
- Conversation history held in `st.session_state`
- Editable system prompt
- Optional conversation summarisation; the summary is used as context for later messages, reducing the amount of old message history sent to the model
- Project-local `.env` loading so launching the app from the workspace root works

## Setup in the shared workspace

1. Extract `05_chatgpt_clone_gemini` inside the `langchain-projects` folder.
2. In PowerShell, move to the `langchain-projects` root and activate the existing environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r .\05_chatgpt_clone_gemini\requirements.txt
   Copy-Item .\05_chatgpt_clone_gemini\.env.example .\05_chatgpt_clone_gemini\.env
   notepad .\05_chatgpt_clone_gemini\.env
   ```

3. Replace the placeholder `GOOGLE_API_KEY` with your Google AI Studio API key. Leave the `GEMINI_MODEL` line as `gemini-3.5-flash-lite`, or change it to another model ID available to your API key.
4. Start the app from the `langchain-projects` root:

   ```powershell
   streamlit run .\05_chatgpt_clone_gemini\app.py
   ```

## What to learn

- `HumanMessage`, `AIMessage`, and `SystemMessage` represent user input, assistant replies, and system instructions.
- `llm.invoke(messages)` sends the prepared conversation to the model.
- `st.session_state` survives Streamlit reruns for the active browser session.
- The summary button compresses the conversation into a summary. Later requests send the summary plus messages added since the summary was made.

This version intentionally avoids the older `ConversationChain` and `ConversationSummaryMemory` classes in the original source and implements the chat flow directly with message objects.

## Security

Keep `.env` private. Never commit or upload your real API key. `.env.example` is only a template and should contain placeholders, not real credentials.
