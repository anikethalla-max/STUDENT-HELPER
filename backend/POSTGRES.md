# Connect local PostgreSQL

The Python backend connects directly to PostgreSQL. Use pgAdmin to create and inspect the application database.

1. Connect to the local PostgreSQL server in pgAdmin (host 127.0.0.1, port 5432).
2. Under **Login/Group Roles**, create `learn_topic_app`, give it a password, and enable **Can login?**.
3. Under **Databases**, create `learn_topic` with `learn_topic_app` as its owner.
4. Set `DATABASE_URL` in the project-root `.env`. Percent-encode reserved URI characters in the password, such as `@` as `%40` and `#` as `%23`.

```dotenv
DATABASE_URL=postgresql://learn_topic_app:YOUR_PASSWORD@127.0.0.1:5432/learn_topic
```

In PowerShell, from the project root, add PostgreSQL to the PATH and run the database check:

```powershell
$env:Path = 'D:\POSTGRESQL\bin;' + $env:Path
uv run --locked --env-file .env python -m backend.local_web --check-db
```

The check initializes the checkpoint tables without binding a port or calling a model.
Stop any older backend process before restarting it in the same PowerShell session:

```powershell
uv run --locked --env-file .env python -m backend.local_web
```

The backend's /health endpoint reports "checkpoints": "postgresql". To verify resume,
start a plan, leave the approval prompt open, restart the backend, then approve the
plan in the same tab. In pgAdmin, SELECT count(*) FROM public.checkpoints; shows the
checkpoint count. Recent chat links live in browser tab storage.

## Enable local Chroma for chat history

Keep `DATABASE_URL` in `.env` so LangGraph saves your chat threads in PostgreSQL.
Download the [standalone Windows Chroma CLI](https://github.com/chroma-core/chroma/releases/tag/cli-1.4.4)
to `.local-runtime/chroma.exe` in the project on D:. Use the standalone binary
so the backend does not need Chroma's Python package. From the project folder run:

```powershell
.\backend\start-local-memory.ps1
$env:TEMP = Join-Path $PWD '.local-runtime\tmp'
$env:TMP = $env:TEMP
$env:OLLAMA_HOST = '127.0.0.1:11435'
ollama pull nomic-embed-text
```

The pull is only needed once. The launcher starts local Chroma and an Ollama
server for embeddings. The model, server logs, and temporary files go under
`.local-runtime`; the Chroma database goes under `.chroma`. Set these values in `.env`:

```dotenv
CHAT_EMBEDDING_MODEL=nomic-embed-text
CHAT_EMBEDDING_URL=http://127.0.0.1:11435
CHROMA_PORT=8000
```

With TEMP and TMP set as above, launch FastAPI using `uv run --locked --cache-dir .uv-cache --env-file .env python -m backend.local_web`.
Open http://127.0.0.1:8787/health. It should report `"chat_history_index": "chroma"`.
Groq still generates replies. No Chroma account or API key is required. Existing
chats are indexed when you use them again. If a vector search fails, the agents
can search the checkpoint text by keyword.
