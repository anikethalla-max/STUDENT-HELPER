"""Load study material and rank passages for the Tutor's search tool."""

import re
from collections import Counter
from collections.abc import Iterator
from itertools import chain
from math import log1p, sqrt, sumprod
from pathlib import Path
from typing import TypedDict

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class StudyDocument(TypedDict):
    source: str
    page: int | None
    text: str


class SourceReference(BaseModel):
    source_id: str
    source: str
    page: int | None
    excerpt: str


def load_study_file(path: Path) -> list[StudyDocument]:
    """Keep PDF page boundaries so citations point back to the source."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        documents = [StudyDocument(source=path.name, page=None, text=path.read_text(encoding="utf-8-sig"))]
    elif suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            with path.open("rb") as file:
                reader = PdfReader(file)
                if reader.is_encrypted:
                    raise ValueError(f"{path.name} is encrypted. Supply an unlocked PDF.")
                documents = [
                    StudyDocument(source=path.name, page=index, text=page.extract_text())
                    for index, page in enumerate(reader.pages, 1)
                ]
        except PdfReadError as error:
            raise ValueError(f"Unable to extract text from {path.name}: {error}") from error
    else:
        raise ValueError(f"{path.name}: supply a .txt, .md, or text-based .pdf file.")
    readable = [document for document in documents if document["text"].strip()]
    if not readable:
        raise ValueError(f"{path.name} has no readable text. Scanned PDFs need OCR.")
    return readable


def passage_chunks(text: str) -> Iterator[str]:
    """Split near word boundaries and retain overlap for context."""
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), 100):
        end = min(start + 120, len(words))
        yield text[words[start].start():words[end - 1].end()]
        if end == len(words):
            break


def _terms(text: str) -> Counter[str]:
    return Counter(re.findall(r"[^\W_]+", text.casefold()))


class StudyRetriever:
    """Index one request's documents and combine keyword and optional vector rankings."""

    def __init__(self, documents: list[StudyDocument], embeddings: Embeddings | None = None):
        self.embeddings = embeddings
        self.passages = [
            SourceReference(source_id=f"notes-{index}", source=source, page=page, excerpt=excerpt)
            for index, (source, page, excerpt) in enumerate(
                (
                    (document["source"], document["page"], chunk)
                    for document in documents
                    for chunk in passage_chunks(document["text"])
                ),
                1,
            )
        ]
        self.term_counts = [_terms(passage.excerpt) for passage in self.passages]
        self.document_frequency = Counter(chain.from_iterable(counts.keys() for counts in self.term_counts))
        self.vectors: list[list[float]] | None = None

    def search(self, query: str, *, limit: int = 3) -> list[SourceReference]:
        terms = _terms(query)
        if not terms or not self.passages:
            return []
        weights = {
            term: log1p(len(self.passages) / frequency)
            for term in terms
            if (frequency := self.document_frequency[term])
        }
        scores = [
            sum(weight * log1p(counts[term]) for term, weight in weights.items())
            / sqrt(max(1, counts.total()))
            for counts in self.term_counts
        ]
        rankings = [self._rank(scores)]

        if self.embeddings is not None:
            if self.vectors is None:
                self.vectors = self.embeddings.embed_documents([passage.excerpt for passage in self.passages])
            query_vector = self.embeddings.embed_query(query)
            query_norm = sqrt(sumprod(query_vector, query_vector))
            similarities = []
            for vector in self.vectors:
                denominator = query_norm * sqrt(sumprod(vector, vector))
                similarities.append(sumprod(query_vector, vector) / denominator if denominator else 0.0)
            rankings.append(self._rank(similarities))

        # Reciprocal rank fusion keeps keyword and cosine scores on separate scales.
        combined: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking, 1):
                combined[index] = combined.get(index, 0.0) + 1 / (60 + rank)
        ranked = sorted(combined, key=combined.__getitem__, reverse=True)
        return [self.passages[index] for index in ranked[:limit]]

    @staticmethod
    def _rank(scores: list[float]) -> list[int]:
        return sorted((index for index, score in enumerate(scores) if score > 0), key=scores.__getitem__, reverse=True)
