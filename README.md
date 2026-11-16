# Learn a Topic

Learn a Topic is a local study assistant. It can propose a plan, explain a concept,
quiz you, or research a question and cite web sources. You can also upload notes
and ask it to use only the attached material.

The Next.js frontend talks to a FastAPI backend. LangGraph routes work to the
Planner, Tutor, Assessment, and Deep Research agents. PostgreSQL can save
conversation checkpoints, and local Chroma can index earlier chat turns for
semantic search. Deep Research uses Tavily MCP.

## Run locally

1. Copy `.env.example` to `.env` and configure a model provider. The example
   is set up for Groq. Start without PostgreSQL or Chroma by leaving
   `DATABASE_URL` and `CHAT_EMBEDDING_MODEL` blank.
2. In the repository root, install the Python dependencies and start the backend:

   ```sh
   uv sync --locked
   uv run --locked --env-file .env python -m backend.local_web
   ```

3. In a second terminal, start the frontend:

   ```sh
   cd frontend
   npm ci
   npm run dev
   ```

4. Open [http://127.0.0.1:3000](http://127.0.0.1:3000).

See the [PostgreSQL and Chroma setup guide](backend/POSTGRES.md) for persistent
memory and [architecture notes](ARCHITECTURE.md) for the design. The app is
intended to run locally.
