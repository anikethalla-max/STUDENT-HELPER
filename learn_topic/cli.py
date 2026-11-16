"""Run an individual learning agent from a terminal."""

import argparse
import asyncio
import os
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langsmith import tracing_context

from agents.assessment import PracticeQuestion, create_question, grade_answer
from agents.planner import plan_learning
from agents.tutor import teach
from learn_topic.retrieval import StudyDocument, load_study_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Learn a Topic agent.")
    parser.add_argument("topic", nargs="?", help="What you want to learn or practise")
    parser.add_argument(
        "--agent", choices=("tutor", "planner", "assessment", "research", "workflow"), default="tutor"
    )
    parser.add_argument("--level", default="beginner", help="Your current level")
    parser.add_argument("--notes", type=Path, action="append", help="Study file (.txt, .md, or .pdf), repeat to add more")
    parser.add_argument("--notes-only", action="store_true", help="Require evidence from notes")
    parser.add_argument("--progress", help="Existing learning progress summary for the Planner")
    parser.add_argument("--question-file", type=Path, help="Saved Assessment question JSON to grade")
    parser.add_argument("--answer", help="Learner's answer to the saved question")
    parser.add_argument("--trace", action="store_true", help="Send this run to LangSmith")
    args = parser.parse_args()

    if args.agent not in {"tutor", "workflow"} and (args.notes is not None or args.notes_only):
        parser.error("--notes and --notes-only require --agent tutor or --agent workflow.")
    if args.agent not in {"planner", "workflow"} and args.progress is not None:
        parser.error("--progress requires --agent planner or --agent workflow.")
    grading = args.question_file is not None or args.answer is not None
    if grading:
        if args.agent != "assessment" or args.question_file is None or args.answer is None:
            parser.error("Grading requires --agent assessment, --question-file, and --answer.")
        if args.topic is not None:
            parser.error("Grading reads the question from --question-file. Omit the topic.")
    elif args.topic is None:
        parser.error("Provide a topic to learn or practise.")

    provider = os.getenv("LEARN_TOPIC_PROVIDER", "openai").strip().lower()
    if provider not in {"groq", "ollama", "openai"}:
        parser.error("LEARN_TOPIC_PROVIDER must be groq, ollama, or openai.")
    groq_key = os.getenv("GROQ_API_KEY")
    if provider == "groq" and not groq_key:
        parser.exit(2, "Set GROQ_API_KEY in .env and load it with uv run --env-file .env.\n")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        parser.exit(2, "Set OPENAI_API_KEY in your environment or in .env with uv run --env-file .env.\n")
    if args.trace and not os.environ.get("LANGSMITH_API_KEY"):
        parser.exit(2, "Set LANGSMITH_API_KEY in your environment or in .env before using --trace.\n")

    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "").strip()
    try:
        documents = [document for path in (args.notes or []) for document in load_study_file(path)]
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url) if documents and embedding_model else None
        with tracing_context(enabled=args.trace):
            model_name = os.getenv("LEARN_TOPIC_MODEL")
            if provider == "ollama":
                model = ChatOllama(
                    model=model_name or "qwen2.5:7b",
                    base_url=ollama_url,
                    temperature=0,
                )
            elif provider == "groq":
                model = ChatGroq(
                    model=model_name or "openai/gpt-oss-20b",
                    api_key=groq_key,
                )
            else:
                model = ChatOpenAI(model=model_name or "gpt-4.1-mini")
            if args.agent == "planner":
                result = plan_learning(
                    args.topic,
                    model=model,
                    level=args.level,
                    learning_progress=args.progress or "",
                )
            elif args.agent == "assessment":
                if args.question_file is not None:
                    question = PracticeQuestion.model_validate_json(
                        args.question_file.read_text(encoding="utf-8-sig")
                    )
                    result = grade_answer(question, args.answer, model=model)
                else:
                    result = create_question(args.topic, model=model, level=args.level)
            elif args.agent == "research":
                result = asyncio.run(run_research(args.topic, model=model))
            elif args.agent == "workflow":
                asyncio.run(run_workflow(args, model=model, documents=documents, embeddings=embeddings))
                return
            else:
                result = teach(
                    args.topic,
                    model=model,
                    level=args.level,
                    notes_only=args.notes_only,
                    documents=documents,
                    embeddings=embeddings,
                )
    except ConnectionError:
        if provider == "ollama" or embedding_model:
            parser.exit(1, "Couldn't connect to Ollama. Start Ollama and check OLLAMA_BASE_URL.\n")
        parser.exit(1, f"Couldn't connect to {provider}. Check the network connection.\n")
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"{args.agent.title()}: {error}\n")

    print(result.model_dump_json(indent=2))


async def run_research(question, *, model):
    from agents.research import research_topic
    from mcp_server.tavily import load_search_tool

    search_tool = await load_search_tool(os.getenv("TAVILY_API_KEY"))
    return await research_topic(question, model=model, search_tool=search_tool)


async def run_workflow(
    args, *, model, documents: list[StudyDocument], embeddings: Embeddings | None = None
) -> None:
    from uuid import uuid4

    from langgraph.types import Command

    from agents.graph import build_learning_graph
    from mcp_server.tavily import load_search_tool

    async def load_search():
        return await load_search_tool(os.getenv("TAVILY_API_KEY"))

    graph = build_learning_graph(
        model,
        load_search_tool=load_search,
        embeddings=embeddings,
    )
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 24}
    state = await graph.ainvoke({
        "request": args.topic,
        "level": args.level,
        "documents": documents,
        "material_summaries": tuple(dict.fromkeys(document["source"] for document in documents)),
        "notes_only": args.notes_only,
        "learning_progress": args.progress or "",
    }, config=config)

    lesson_shown = False
    while pending := state.get("__interrupt__"):
        prompt = pending[0].value
        if prompt["kind"] == "approval":
            print(f"Study plan for {prompt['topic']}")
            for index, step in enumerate(prompt["steps"], start=1):
                print(f"{index}. {step}")
            reply = input("Approve plan? [y/N] ").strip().casefold() in {"y", "yes"}
        else:
            if prompt["kind"] == "answer" and "lesson" in prompt:
                print(f"\n{prompt['lesson']}\n\nExample: {prompt['example']}")
                lesson_shown = True
            reply = input(f"\n{prompt['question']}\n> ").strip()
            while not reply:
                reply = input("Enter a response: ").strip()
        state = await graph.ainvoke(Command(resume=reply), config=config)

    if state.get("approved") is False:
        print("Plan declined.")
    for key in ("lesson", "feedback", "research"):
        if state.get(key) is not None and (key != "lesson" or not lesson_shown):
            print(state[key].model_dump_json(indent=2))
    if lesson_shown:
        for source in state["lesson"].sources:
            page = f", page {source.page}" if source.page is not None else ""
            print(f"Source [{source.source_id}] {source.source}{page}: {source.excerpt}")


if __name__ == "__main__":
    main()
