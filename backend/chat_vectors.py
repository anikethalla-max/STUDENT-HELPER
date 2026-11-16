"""Local Chroma index for chat messages saved in LangGraph checkpoints."""

import json
import logging
from hashlib import sha256
from http.client import HTTPConnection
from itertools import batched
from threading import Lock
from uuid import UUID

from langchain_core.embeddings import Embeddings

from learn_topic.chat_memory import ChatMessage
from learn_topic.retrieval import passage_chunks

logger = logging.getLogger(__name__)
COLLECTIONS = "/api/v2/tenants/default_tenant/databases/default_database/collections"


class ChromaChatIndex:
    def __init__(self, port: int, model: str, embeddings: Embeddings):
        self.port = port
        self.embeddings = embeddings
        model_key = sha256(model.encode()).hexdigest()[:16]
        self.collection_name = f"learn_topic_chat_{model_key}"
        self.lock = Lock()

    def _post(self, route: str, **body):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", route, json.dumps(body, allow_nan=False), {"Content-Type": "application/json"})
            reply = connection.getresponse()
            if not 200 <= reply.status < 300:
                raise RuntimeError(f"Local Chroma returned HTTP {reply.status}.")
            return json.loads(reply.read() or b"null")
        finally:
            connection.close()

    def setup(self) -> None:
        try:
            collection = self._post(
                COLLECTIONS, name=self.collection_name, get_or_create=True,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except OSError as error:
            raise RuntimeError(f"Start the local Chroma server on 127.0.0.1:{self.port}.") from error
        self.collection = f"{COLLECTIONS}/{UUID(collection['id'])}"

    def index_history(self, thread_id: str, messages: list[ChatMessage]) -> None:
        """Warm the index after a turn so the reply does not wait for embeddings."""
        try:
            self._sync(thread_id, messages)
        except Exception as error:
            logger.warning("Chat vector indexing deferred (%s).", type(error).__name__)

    def _sync(self, thread_id: str, messages: list[ChatMessage]) -> None:
        # Stable IDs make replays idempotent and allow a partial batch to be retried.
        passages = (
            (f"{thread_id}:{position}:{number}", position, message["speaker"], passage)
            for position, message in enumerate(messages, 1)
            for number, passage in enumerate(passage_chunks(message["text"]), 1)
        )
        with self.lock:
            for batch in batched(passages, 32):
                existing = set(self._post(f"{self.collection}/get", ids=[entry[0] for entry in batch], include=[])["ids"])
                missing = [entry for entry in batch if entry[0] not in existing]
                if not missing:
                    continue
                texts = [entry[3] for entry in missing]
                self._post(f"{self.collection}/upsert",
                    ids=[entry[0] for entry in missing],
                    documents=texts,
                    embeddings=self.embeddings.embed_documents(texts),
                    metadatas=[
                        {"thread_id": thread_id, "message_no": position, "speaker": speaker}
                        for _, position, speaker, _ in missing
                    ],
                )

    def search(self, thread_id: str, messages: list[ChatMessage], query: str) -> list[dict[str, str]]:
        """Backfill missing passages, then search only the current thread's earlier turns."""
        self._sync(thread_id, messages)
        result = self._post(f"{self.collection}/query",
            query_embeddings=[self.embeddings.embed_query(query[:2000])],
            where={"$and": [{"thread_id": thread_id}, {"message_no": {"$lte": len(messages)}}]},
            n_results=12,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        seen = set()
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            position = meta["message_no"]
            if position in seen or distance >= 1:
                continue
            seen.add(position)
            hits.append({"message": f"Message {position} ({meta['speaker']})", "text": text})
            if len(hits) == 3:
                break
        return hits
