# RAG Chatbot (Portfolio Project 1)

Upload your own documents and ask questions about them. Answers are grounded in the
documents, cite file and page, and abstain when the answer is not there.

## Quick start
1. `python -m venv .venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. `docker compose up -d`
5. `python app/check_setup.py`  (all three lines should say OK)

## Project layout
- `app/`   pipeline code (parse, chunk, embed, retrieve, generate)
- `ui/`    Streamlit interface
- `eval/`  test questions and evaluation scripts
- `tests/` unit and integration tests
- `data/`  sample documents and uploads
