import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from zhipuai import ZhipuAI

from utils.config import get_required_setting, get_setting


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWLEDGE_DIR = DATA_DIR / "travel_knowledge"
VECTOR_STORE_DIR = DATA_DIR / "vector_store" / "chroma"
SIGNATURE_FILE = VECTOR_STORE_DIR / "travel_knowledge.signature.json"
COLLECTION_NAME = "travel_knowledge"
DEFAULT_EMBEDDING_MODEL = "embedding-3"


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    content: str

    @property
    def document(self) -> str:
        return f"{self.title}\n{self.content}"

    @property
    def id(self) -> str:
        raw = f"{self.source}|{self.title}|{self.content}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _split_markdown_sections(path: Path) -> List[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+", text)
    chunks: List[KnowledgeChunk] = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        lines = section.splitlines()
        title = lines[0].strip()
        content = "\n".join(line.strip() for line in lines[1:] if line.strip()).strip()
        if content:
            chunks.append(KnowledgeChunk(source=path.name, title=title, content=content))
    return chunks


@lru_cache(maxsize=1)
def _load_knowledge_chunks() -> List[KnowledgeChunk]:
    if not KNOWLEDGE_DIR.exists():
        return []

    chunks: List[KnowledgeChunk] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        chunks.extend(_split_markdown_sections(path))
    return chunks


def _build_query(parsed_request: Dict[str, Any], user_message: str = "") -> str:
    preferences = " ".join(str(item) for item in parsed_request.get("preferences", []) or [])
    return " ".join(
        item
        for item in [
            str(parsed_request.get("city", "")),
            str(parsed_request.get("travel_group", "")),
            str(parsed_request.get("transport_preference", "")),
            str(parsed_request.get("special_requirements", "")),
            preferences,
            user_message,
        ]
        if item
    )


def _knowledge_signature(chunks: Sequence[KnowledgeChunk]) -> str:
    payload = [
        {"id": chunk.id, "source": chunk.source, "title": chunk.title}
        for chunk in chunks
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _read_signature() -> str:
    if not SIGNATURE_FILE.exists():
        return ""
    try:
        return json.loads(SIGNATURE_FILE.read_text(encoding="utf-8")).get("signature", "")
    except (OSError, json.JSONDecodeError):
        return ""


def _write_signature(signature: str) -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    SIGNATURE_FILE.write_text(
        json.dumps({"signature": signature}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_chroma_client():
    import chromadb

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


def _extract_embedding(item: Any) -> List[float]:
    embedding = getattr(item, "embedding", None)
    if embedding is None and isinstance(item, dict):
        embedding = item.get("embedding")
    return [float(value) for value in embedding or []]


def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    model = get_setting("ZHIPU_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
    client = ZhipuAI(api_key=get_required_setting("ZHIPU_API_KEY"))
    response = client.embeddings.create(model=model, input=list(texts))
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data", [])
    embeddings = [_extract_embedding(item) for item in data or []]
    if len(embeddings) != len(texts) or any(not item for item in embeddings):
        raise RuntimeError("Embedding response size does not match input size.")
    return embeddings


def _rebuild_vector_store(chunks: Sequence[KnowledgeChunk], signature: str) -> Any:
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    documents = [chunk.document for chunk in chunks]
    embeddings = _embed_texts(documents)
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk.source,
                "title": chunk.title,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    )
    _write_signature(signature)
    return collection


def _get_vector_collection(chunks: Sequence[KnowledgeChunk]) -> Any:
    client = _get_chroma_client()
    signature = _knowledge_signature(chunks)
    if _read_signature() != signature:
        return _rebuild_vector_store(chunks, signature)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    try:
        if collection.count() != len(chunks):
            return _rebuild_vector_store(chunks, signature)
    except Exception:
        return _rebuild_vector_store(chunks, signature)
    return collection


def _retrieve_with_chroma(
    chunks: Sequence[KnowledgeChunk],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    collection = _get_vector_collection(chunks)
    query_embedding = _embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, 1),
        include=["documents", "metadatas", "distances"],
    )

    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    selected = []
    for metadata, distance in zip(metadatas, distances):
        selected.append(
            {
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "content": metadata.get("content", ""),
                "score": round(1 / (1 + float(distance or 0)), 4),
            }
        )
    return selected


def _retrieve_with_tfidf(
    chunks: Sequence[KnowledgeChunk],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    documents = [chunk.document for chunk in chunks]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    ranked_indexes = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)
    selected = []
    for index in ranked_indexes[: max(top_k, 1)]:
        if scores[index] <= 0 and selected:
            continue
        chunk = chunks[index]
        selected.append(
            {
                "source": chunk.source,
                "title": chunk.title,
                "content": chunk.content,
                "score": round(float(scores[index]), 4),
            }
        )
    return selected


def _format_summary(items: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        f"- [{item['source']}] {item['title']}: {item['content']}"
        for item in items
    )


def retrieve_travel_knowledge(
    parsed_request: Dict[str, Any],
    user_message: str = "",
    top_k: int = 4,
) -> Dict[str, Any]:
    chunks = _load_knowledge_chunks()
    if not chunks:
        return {"query": "", "chunks": [], "summary": "", "retriever": "none"}

    query = _build_query(parsed_request, user_message)
    retriever = "chroma_embedding"
    try:
        selected = _retrieve_with_chroma(chunks, query, top_k)
    except Exception as exc:
        retriever = "tfidf_fallback"
        selected = _retrieve_with_tfidf(chunks, query, top_k)
        fallback_error = str(exc)
    else:
        fallback_error = ""

    return {
        "query": query,
        "chunks": selected,
        "summary": _format_summary(selected),
        "retriever": retriever,
        "fallback_error": fallback_error,
    }
