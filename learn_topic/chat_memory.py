"""Recent context and searchable history for one learning thread."""

import logging
from typing import Protocol, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool

from learn_topic.retrieval import StudyDocument, StudyRetriever

RECENT_MESSAGES = 8
RECENT_CHARACTERS = 8_000
logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    speaker: str
    text: str


class ChatHistoryIndex(Protocol):
    def search(
        self, thread_id: str, messages: list[ChatMessage], query: str
    ) -> list[dict[str, str]]: ...


def recent_chat_context(messages: list[ChatMessage]) -> str:
    """Fit recent messages into the prompt, keeping the newest message intact when possible."""
    recent = []
    remaining = RECENT_CHARACTERS
    for message in reversed(messages[-RECENT_MESSAGES:]):
        if remaining <= 0:
            break
        line = f"{message['speaker']}: {message['text']}"
        recent.append(line if len(line) <= remaining else line[:remaining - 1] + "…")
        remaining -= len(recent[-1]) + 2
    return "\n\n".join(reversed(recent))


def chat_history_tool(
    messages: list[ChatMessage],
    embeddings: Embeddings | None = None,
    *,
    index: ChatHistoryIndex | None = None,
    thread_id: str | None = None,
) -> BaseTool:
    """Create a bounded search tool scoped to the current graph thread."""
    retriever: StudyRetriever | None = None
    searches = 0

    @tool
    def search_chat_history(query: str) -> list[dict[str, str]]:
        """Search earlier turns in this chat. Use 'first messages' to read the chat's opening."""
        nonlocal retriever, searches
        if searches >= 3:
            return [{"message": "History search limit reached. Use the context already available."}]
        searches += 1
        history = messages[:-1] if messages and messages[-1]["speaker"] == "Learner" else messages
        if query.strip().casefold() == "first messages":
            return [
                {"message": f"Message {index} · {message['speaker']}", "text": message["text"][:800]}
                for index, message in enumerate(history[:5], 1)
            ]
        if index is not None and thread_id is not None and query.strip() and history:
            try:
                if hits := index.search(thread_id, history, query):
                    return hits
            except Exception as error:
                logger.warning("Chat vector search failed (%s); using keyword search.", type(error).__name__)
        if retriever is None:
            documents = [
                StudyDocument(source=f"Message {index} · {message['speaker']}", page=None, text=message["text"])
                for index, message in enumerate(history, 1)
            ]
            retriever = StudyRetriever(documents, embeddings if index is None else None)
        return [{"message": item.source, "text": item.excerpt} for item in retriever.search(query)]

    return search_chat_history
