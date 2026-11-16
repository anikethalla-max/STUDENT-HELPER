# Learn a Topic frontend

Start the FastAPI backend from the repository root.

```powershell
uv run --locked --env-file .env python -m backend.local_web
```

In a second terminal, start Next.js. Use Node.js 20.9 or newer.

```powershell
cd frontend
npm ci
npm run dev
```

Open [the local study workspace](http://127.0.0.1:3000). Select a study activity, enter a topic, and optionally attach up to four TXT, Markdown, or text-based PDF files. Each file can be up to 8 MB. The Python process reads the existing provider settings from the root `.env`. Deep Research uses rate-limited Tavily MCP search without a key and uses your account limits if you set `TAVILY_API_KEY`. With a cloud model, passages the Tutor retrieves are included in the model request.

The local API uses in-memory checkpoints by default. Follow [PostgreSQL setup](../backend/POSTGRES.md) to retain workflow checkpoints across backend restarts. Recent results and session links are held in the browser tab's session storage. The API listens on loopback and is intended for local use. Its OpenAPI documentation is available at [localhost:8787/docs](http://127.0.0.1:8787/docs).
