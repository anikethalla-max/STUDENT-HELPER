"""Local FastAPI service for the learning workflow."""

import argparse
import asyncio
import base64
import logging
import os
import tempfile
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal
from uuid import UUID, uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, StringConstraints
from starlette.middleware.trustedhost import TrustedHostMiddleware

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from langsmith import tracing_context

from agents.graph import build_learning_graph, legacy_chat_messages
from backend.chat_vectors import ChromaChatIndex
from learn_topic.retrieval import load_study_file
from mcp_server.tavily import TavilySearchLimit, TavilySearchUnavailable, load_search_tool

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES = 4 * (4 * ((MAX_FILE_BYTES + 2) // 3)) + 32 * 1024
logger = logging.getLogger(__name__)
StudyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class StudyFile(BaseModel):
    name: str
    data: str


class StartStudy(BaseModel):
    request: StudyText
    level: Literal["beginner", "intermediate", "advanced"]
    notes_only: StrictBool = False
    files: list[StudyFile] = Field(default_factory=list, max_length=4)


class ResumeStudy(BaseModel):
    id: UUID
    interrupt_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    reply: StrictBool | StudyText


class FollowUpStudy(BaseModel):
    id: UUID
    request: StudyText


class EndStudy(BaseModel):
    id: UUID


def make_graph(checkpointer: BaseCheckpointSaver | None = None, chat_index: ChromaChatIndex | None = None):
    provider = os.getenv("LEARN_TOPIC_PROVIDER", "openai").strip().lower()
    model_name = os.getenv("LEARN_TOPIC_MODEL")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "groq":
        model = ChatGroq(
            model=model_name or "openai/gpt-oss-20b",
            api_key=os.environ["GROQ_API_KEY"],
            timeout=60,
        )
    elif provider == "ollama":
        model = ChatOllama(model=model_name or "qwen2.5:7b", base_url=ollama_url)
    elif provider == "openai":
        model = ChatOpenAI(model=model_name or "gpt-4.1-mini", timeout=60)
    else:
        raise ValueError("Set LEARN_TOPIC_PROVIDER to groq, ollama, or openai.")

    async def tavily():
        return await load_search_tool(os.getenv("TAVILY_API_KEY"))

    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "").strip()
    return build_learning_graph(
        model,
        load_search_tool=tavily,
        embeddings=OllamaEmbeddings(model=embedding_model, base_url=ollama_url) if embedding_model else None,
        checkpointer=checkpointer,
        chat_index=chat_index,
    )


async def open_postgres_checkpointer(database_url: str, resources: AsyncExitStack) -> BaseCheckpointSaver:
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as error:
        raise RuntimeError("The PostgreSQL driver could not load. Add PostgreSQL's bin directory to PATH.") from error
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from psycopg import Error
    from psycopg.conninfo import make_conninfo

    serializer = JsonPlusSerializer(allowed_msgpack_modules=[
        ("agents.assessment", "AssessmentFeedback"),
        ("agents.assessment", "PracticeQuestion"),
        ("agents.planner", "PlannerDecision"),
        ("agents.research", "Finding"),
        ("agents.research", "ResearchResult"),
        ("agents.tutor", "Lesson"),
        ("agents.tutor", "LessonDraft"),
        ("learn_topic.retrieval", "SourceReference"),
    ])
    try:
        saver = await resources.enter_async_context(AsyncPostgresSaver.from_conn_string(
            make_conninfo(database_url, connect_timeout=5), serde=serializer,
        ))
        await saver.setup()
    except Error as error:
        raise RuntimeError(
            "PostgreSQL setup failed. Check DATABASE_URL, the app role's password, and its permission "
            "to create tables in the application database."
        ) from error
    return saver


async def open_chat_index(database_url: str) -> ChromaChatIndex | None:
    model = os.getenv("CHAT_EMBEDDING_MODEL", "").strip()
    if not model:
        return None
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to persist the chat threads before enabling Chroma.")
    embeddings = OllamaEmbeddings(
        model=model, base_url=os.getenv("CHAT_EMBEDDING_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        client_kwargs={"timeout": 15.0},
    )
    index = ChromaChatIndex(int(os.getenv("CHROMA_PORT") or "8000"), model, embeddings)
    await asyncio.to_thread(index.setup)
    return index


def study_documents(files: list[StudyFile]):
    documents = []
    with tempfile.TemporaryDirectory() as directory:
        for index, item in enumerate(files):
            filename = PureWindowsPath(item.name).name
            suffix = Path(filename).suffix.lower()
            if filename != item.name or ":" in filename or suffix not in {".txt", ".md", ".pdf"}:
                raise ValueError("Use a .txt, .md, or text-based PDF file.")
            try:
                payload = base64.b64decode(item.data, validate=True)
            except ValueError as error:
                raise ValueError(f"{filename} has invalid file data.") from error
            if len(payload) > MAX_FILE_BYTES:
                raise ValueError("Each study file must be 8 MB or smaller.")
            path = Path(directory, f"source-{index}{suffix}")
            path.write_bytes(payload)
            try:
                documents.extend({**document, "source": filename} for document in load_study_file(path))
            except ValueError as error:
                raise ValueError(str(error).replace(path.name, filename, 1)) from error
    return documents

def public_result(thread_id, state):
    closed = state.get("closed") is True
    pending = () if closed else state.get("__interrupt__", ())
    result = {
        "id": thread_id,
        "closed": closed,
        "pending": {**pending[0].value, "interrupt_id": pending[0].id} if pending else None,
    }
    for name in ("lesson", "feedback", "research"):
        if value := state.get(name):
            result[name] = value.model_dump(mode="json")
    if "decision" in state and state["decision"].action == "plan":
        result["plan"] = state["decision"].model_dump(mode="json")
    result["declined"] = state.get("approved") is False
    if state.get("feedback") and state.get("question") and state.get("answer"):
        result["attempt"] = {"question": state["question"].question, "answer": state["answer"]}
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as resources:
        database_url = os.getenv("DATABASE_URL", "").strip()
        app.state.checkpointer = await open_postgres_checkpointer(database_url, resources) if database_url else None
        app.state.chat_index = await open_chat_index(database_url)
        app.state.graph = None
        app.state.workflow_lock = asyncio.Lock()
        yield


app = FastAPI(title="Learn a Topic", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])


@app.middleware("http")
async def local_requests(request: Request, call_next):
    if request.method == "POST":
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"error": "Use the local study workspace."}, status_code=403)
        if request.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/json":
            return JSONResponse({"error": "Send a JSON study request."}, status_code=415)
        length = request.headers.get("content-length", "")
        if not length.isascii() or not length.isdecimal() or len(length) > 12:
            return JSONResponse({"error": "Provide a valid Content-Length."}, status_code=411)
        if int(length) > MAX_BODY_BYTES:
            return JSONResponse({"error": "Attach up to four study files of 8 MB each."}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def api_error(_request: Request, error: HTTPException):
    return JSONResponse({"error": error.detail}, status_code=error.status_code, headers=error.headers)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, error: RequestValidationError):
    fields = ", ".join(".".join(map(str, item["loc"][1:])) or "body" for item in error.errors())
    return JSONResponse({"error": f"Invalid study request fields: {fields}."}, status_code=422)


@app.get("/health")
async def health(request: Request):
    return {
        "ready": True,
        "checkpoints": "postgresql" if request.app.state.checkpointer is not None else "memory",
        "chat_history_index": "chroma" if request.app.state.chat_index is not None else "keyword",
    }


def _graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 24}


async def can_retry_approval(graph, config, checkpoint, operation: ResumeStudy) -> bool:
    """Match a repeated approval to the last pause before a failed lesson or quiz."""
    if (
        operation.reply is not True
        or checkpoint.values.get("approved") is not True
        or not any(task.error and task.name in {"tutor", "prepare_question"} for task in checkpoint.tasks)
    ):
        return False
    history = [state async for state in graph.aget_state_history(config, limit=8)]
    for previous in history:
        if previous.interrupts:
            last = previous.interrupts[0]
            return last.value["kind"] == "approval" and last.id == operation.interrupt_id
    return False


async def run_workflow(
    request: Request,
    thread_id: str,
    operation: dict | ResumeStudy | FollowUpStudy | EndStudy,
    background: BackgroundTasks | None = None,
):
    service = request.app.state
    try:
        async with asyncio.timeout(180), service.workflow_lock:
            if service.graph is None:
                service.graph = make_graph(service.checkpointer, service.chat_index)
            graph = service.graph
            config = _graph_config(thread_id)
            with tracing_context(enabled=False):
                graph_input = operation
                if not isinstance(operation, dict):
                    checkpoint = await graph.aget_state(config)
                    if not checkpoint.values:
                        raise HTTPException(404, "This chat is unavailable. Start a new session.")
                    if isinstance(operation, EndStudy):
                        if not checkpoint.values.get("closed"):
                            await graph.aupdate_state(config, {"closed": True}, as_node="close_chat")
                        return public_result(thread_id, {**checkpoint.values, "closed": True})
                    if checkpoint.values.get("closed"):
                        raise HTTPException(409, "This chat has ended. Start a new session.")
                    active = checkpoint.interrupts[0] if checkpoint.interrupts else None
                    if isinstance(operation, ResumeStudy):
                        if active is None:
                            if not await can_retry_approval(graph, config, checkpoint, operation):
                                raise HTTPException(409, "This study step is no longer pending.")
                            graph_input = None  # The approval was already recorded; retry the failed node.
                        else:
                            if active.id != operation.interrupt_id:
                                raise HTTPException(409, "This study step is no longer pending.")
                            if (active.value["kind"] == "approval") != isinstance(operation.reply, bool):
                                raise HTTPException(400, "The response does not match the pending study step.")
                            graph_input = Command(resume=operation.reply)
                    else:
                        if active is not None:
                            raise HTTPException(409, "Respond to the current study step before asking a follow-up.")
                        graph_input = {"request": operation.request}
                        if not checkpoint.values.get("chat_messages"):
                            graph_input["chat_messages"] = legacy_chat_messages(checkpoint.values)
                state = await graph.ainvoke(graph_input, config=config, durability="sync")
            if background is not None and service.chat_index and (messages := state.get("chat_messages")):
                background.add_task(service.chat_index.index_history, thread_id, messages)
            return public_result(thread_id, state)
    except HTTPException:
        raise
    except TavilySearchLimit as error:
        message = (
            "Tavily's search limit was reached. Check the credits for your Tavily API key."
            if os.getenv("TAVILY_API_KEY", "").strip()
            else "Tavily's free search limit was reached. Set TAVILY_API_KEY in .env and restart to keep researching."
        )
        raise HTTPException(429, message) from error
    except TavilySearchUnavailable as error:
        message = (
            "Tavily search is unavailable. Check the Tavily API key and remaining credits."
            if os.getenv("TAVILY_API_KEY", "").strip()
            else "Tavily keyless search is unavailable. Try again later or set TAVILY_API_KEY in .env and restart."
        )
        raise HTTPException(503, message) from error
    except TimeoutError as error:
        raise HTTPException(504, "The study workflow timed out.") from error
    except Exception as error:
        logger.warning("Study workflow failed for thread %s (%s)", thread_id, type(error).__name__)
        if getattr(error, "status_code", None) == 401:
            provider = os.getenv("LEARN_TOPIC_PROVIDER", "openai").strip().lower()
            if provider == "groq":
                raise HTTPException(
                    502, "Groq rejected the API key. Update GROQ_API_KEY in .env and restart the backend."
                ) from error
        raise HTTPException(502, "The study workflow failed. Check your provider, model, and API settings.") from error


@app.post("/start")
async def start_study(body: StartStudy, request: Request, background: BackgroundTasks):
    try:
        documents = await asyncio.to_thread(study_documents, body.files) if body.files else []
    except (OSError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    if body.notes_only and not documents:
        raise HTTPException(400, "Attach a source to use sources only.")
    return await run_workflow(request, str(uuid4()), {
        "request": body.request,
        "level": body.level,
        "documents": documents,
        "notes_only": body.notes_only,
        "material_summaries": tuple(dict.fromkeys(document["source"] for document in documents)),
    }, background)


@app.post("/resume")
async def resume_study(body: ResumeStudy, request: Request, background: BackgroundTasks):
    return await run_workflow(request, str(body.id), body, background)


@app.post("/follow-up")
async def follow_up_study(body: FollowUpStudy, request: Request, background: BackgroundTasks):
    return await run_workflow(request, str(body.id), body, background)


@app.post("/end")
async def end_study(body: EndStudy, request: Request):
    return await run_workflow(request, str(body.id), body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Learn a Topic workflow server.")
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Connect to PostgreSQL and initialize the checkpoint tables, then exit.",
    )
    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if args.check_db and not database_url:
        parser.exit(2, "Set DATABASE_URL in .env and load it with uv run --env-file .env.\n")

    # Psycopg async connections on Windows require a selector event loop.
    loop_factory = asyncio.SelectorEventLoop if database_url and os.name == "nt" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        if args.check_db:
            async def check() -> bool:
                async with AsyncExitStack() as resources:
                    await open_postgres_checkpointer(database_url, resources)
                    return await open_chat_index(database_url) is not None
            try:
                vectors_ready = runner.run(check())
            except RuntimeError as error:
                raise SystemExit(str(error)) from None
            print("PostgreSQL connected. LangGraph checkpoint tables are ready.")
            if vectors_ready:
                print("Local Chroma chat history index is ready.")
            return

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8787))
        runner.run(server.serve())
        if not server.started:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
