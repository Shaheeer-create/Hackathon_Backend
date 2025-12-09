This project provides a retrieval service for the "AI-Native Technical Textbook".

Local development

- Create a virtualenv and install dependencies:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Create a `.env` file with the following variables (or set them in your environment):

```
COHERE_API_KEY=your_cohere_key
QDRANT_URL=https://....qdrant.cloud
QDRANT_API_KEY=your_qdrant_key
GEMINI_API_KEY=your_generative_api_key   # if you use main.py CLI
SITEMAP_URL=https://ai-native-technical-textbook.vercel.app/sitemap.xml
```

- Run the FastAPI app locally:

```pwsh
uvicorn api.index:app --reload --port 8000
```

Then open `http://localhost:8000` for a health check and POST to `http://localhost:8000/retrieve` with JSON `{ "query": "your question" }`.

Deploy to Vercel

- Make sure `vercel` CLI is installed and you are logged in.
- In the Vercel dashboard or via the CLI set the environment variables `COHERE_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` (and `GEMINI_API_KEY` if needed).
- From the project root run:

```pwsh
vercel deploy --prod
```

Notes

- Secrets were removed from source and must be provided via environment variables.
- `chunks.py` contains ingestion helpers; running it will create/update the configured Qdrant collection. It will only run when executed directly (`python chunks.py`).
- `main.py` is preserved as a CLI interactive agent tool; it will not auto-run on import.
