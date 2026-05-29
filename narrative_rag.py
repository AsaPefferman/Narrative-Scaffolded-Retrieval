"""
narrative_rag.py
================
Narrative-aware Retrieval Augmented Generation for long-form creative writing
and roleplay. Combines semantic vector search with a structural scaffold index
to preserve chronological ordering and narrative coherence.

Architecture:
  - VectorStore:   semantic search (cosine; swap for ChromaDB/Qdrant in prod)
  - ScaffoldIndex: SQLite-backed structural/temporal index
  - EmbeddingModel: persistent, incremental TF-IDF fallback embedder
  - NarrativeRAG:  orchestrates the above; ordered retrieval + delta ingestion

Usage:
  from narrative_rag import NarrativeRAG

  rag = NarrativeRAG(path="./my_story")
  rag.ingest_text(raw_text, llm_fn=my_ollama_fn)
  context = rag.retrieve_context("the confrontation at the tower")
  print(context.render())

CHANGELOG (review fixes)
  #1/#2 EmbeddingModel is now persistent and incremental. Because TF-IDF
        weights shift as the corpus grows, ingest_text() re-embeds ALL chunks
        on every ingest so stored vectors stay commensurable across sessions.
        The conflicting double-fit in __init__/ingest_text is gone.
  #3    Module-level chunker renamed split_into_chunks() to remove the
        chunk_text name collision; loop variable renamed accordingly.
  #4/#5 upsert_chunk() is now idempotent: chunk-scoped rows are deleted before
        re-insert, and character_facts carries a chunk_id for clean removal.
  #6    Character introduced/last-seen updates use explicit MIN/MAX and are
        correct under out-of-order ingestion.
  #7    include_recent selects the actual last-N sequences from the sorted set,
        robust to gaps.
  #8    VectorStore batches writes; ingest saves once instead of per-chunk.
"""

import json
import sqlite3
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from collections import Counter
from datetime import datetime
import math


# Default similarity threshold for merging two plot-thread descriptions.
# Tune per embedding model; see THRESHOLD GUIDANCE in the docs. Real embedding
# models separate concepts more sharply than the TF-IDF fallback and tolerate a
# higher value safely.
THREAD_MERGE_THRESHOLD = 0.82


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    chunk_id: str
    sequence: int
    scene_id: str
    arc: str
    characters_present: list[str]
    location: str
    emotional_tone: str
    plot_threads_active: list[str]
    plot_threads_resolved: list[str]
    new_facts: list[str]
    summary: str
    text: str
    embedding: list[float] = field(default_factory=list)
    # {thread_id: human description} — used for embedding-based thread
    # resolution and the canonical registry (thread-drift fixes 1-4).
    plot_thread_descriptions: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks: list[ChunkMetadata]
    characters: dict
    active_threads: list[dict]
    recent_scenes: list[dict]

    def render(self, include_full_text: bool = True) -> str:
        """Render a context window suitable for injection into an LLM prompt."""
        parts = []

        if self.characters:
            parts.append("=== CHARACTER STATE ===")
            for name, info in self.characters.items():
                facts = "; ".join(info.get("known_facts", []))
                parts.append(f"  {name}: {facts}")
            parts.append("")

        if self.active_threads:
            parts.append("=== ACTIVE PLOT THREADS ===")
            for t in self.active_threads:
                parts.append(
                    f"  [{t['thread_id']}] opened at chunk "
                    f"{t['opened_at_sequence']}: {t['description']}")
            parts.append("")

        parts.append("=== RELEVANT STORY SEGMENTS (chronological) ===")
        for chunk in self.chunks:
            parts.append(
                f"[Scene: {chunk.scene_id} | Seq: {chunk.sequence} | {chunk.location}]")
            parts.append(chunk.text if include_full_text else f"  Summary: {chunk.summary}")
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scaffold Index (SQLite)
# ---------------------------------------------------------------------------

class ScaffoldIndex:
    """
    Structural index over the narrative. Fast lookups for characters and their
    known facts, plot-thread status, scene ordering, and sequence->chunk mapping.

    upsert_chunk() is idempotent: calling it twice with the same chunk_id leaves
    the index in the same state (fixes #4, #5).
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                sequence    INTEGER UNIQUE,
                scene_id    TEXT,
                arc         TEXT,
                location    TEXT,
                emotional_tone TEXT,
                summary     TEXT,
                text        TEXT
            );

            CREATE TABLE IF NOT EXISTS characters (
                name        TEXT PRIMARY KEY,
                introduced_at_sequence INTEGER,
                last_seen_sequence     INTEGER,
                arc         TEXT
            );

            -- chunk_id added so facts can be cleaned up on re-ingestion (#4)
            CREATE TABLE IF NOT EXISTS character_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id    TEXT,
                character   TEXT,
                fact        TEXT,
                added_at_sequence INTEGER
            );

            CREATE TABLE IF NOT EXISTS chunk_characters (
                chunk_id    TEXT,
                character   TEXT,
                UNIQUE(chunk_id, character)
            );

            CREATE TABLE IF NOT EXISTS plot_threads (
                thread_id   TEXT PRIMARY KEY,
                description TEXT,
                status      TEXT DEFAULT 'active',
                opened_at_sequence  INTEGER,
                closed_at_sequence  INTEGER
            );

            CREATE TABLE IF NOT EXISTS chunk_threads (
                chunk_id    TEXT,
                thread_id   TEXT,
                role        TEXT,  -- 'active' | 'resolved' | 'mentioned'
                UNIQUE(chunk_id, thread_id, role)
            );

            CREATE TABLE IF NOT EXISTS scenes (
                scene_id    TEXT PRIMARY KEY,
                first_sequence INTEGER,
                last_sequence  INTEGER,
                summary     TEXT
            );
        """)
        self.conn.commit()

    def upsert_chunk(self, meta: ChunkMetadata):
        # --- Idempotency: clear any chunk-scoped rows first (#4, #5) ---
        self.conn.execute("DELETE FROM chunk_characters WHERE chunk_id=?", (meta.chunk_id,))
        self.conn.execute("DELETE FROM chunk_threads   WHERE chunk_id=?", (meta.chunk_id,))
        self.conn.execute("DELETE FROM character_facts WHERE chunk_id=?", (meta.chunk_id,))

        # --- Chunk row ---
        self.conn.execute("""
            INSERT OR REPLACE INTO chunks
            (chunk_id, sequence, scene_id, arc, location, emotional_tone, summary, text)
            VALUES (?,?,?,?,?,?,?,?)
        """, (meta.chunk_id, meta.sequence, meta.scene_id, meta.arc,
              meta.location, meta.emotional_tone, meta.summary, meta.text))

        # --- Characters (correct under out-of-order ingestion; #6) ---
        for char in meta.characters_present:
            self.conn.execute("""
                INSERT OR IGNORE INTO characters
                (name, introduced_at_sequence, last_seen_sequence)
                VALUES (?,?,?)
            """, (char, meta.sequence, meta.sequence))
            # introduced = earliest sequence ever seen; last_seen = latest
            self.conn.execute(
                "UPDATE characters SET introduced_at_sequence = MIN(introduced_at_sequence, ?) "
                "WHERE name = ?", (meta.sequence, char))
            self.conn.execute(
                "UPDATE characters SET last_seen_sequence = MAX(last_seen_sequence, ?) "
                "WHERE name = ?", (meta.sequence, char))
            self.conn.execute(
                "INSERT OR IGNORE INTO chunk_characters (chunk_id, character) VALUES (?,?)",
                (meta.chunk_id, char))

        # --- New facts (now carry chunk_id for clean re-ingestion) ---
        for fact in meta.new_facts:
            owner = next(
                (c for c in meta.characters_present if c.lower() in fact.lower()),
                meta.characters_present[0] if meta.characters_present else "WORLD"
            )
            self.conn.execute("""
                INSERT INTO character_facts (chunk_id, character, fact, added_at_sequence)
                VALUES (?,?,?,?)
            """, (meta.chunk_id, owner, fact, meta.sequence))

        # --- Plot threads ---
        for thread_id in meta.plot_threads_active:
            desc = meta.plot_thread_descriptions.get(thread_id, thread_id)
            self.conn.execute("""
                INSERT OR IGNORE INTO plot_threads
                (thread_id, description, status, opened_at_sequence)
                VALUES (?,?,?,?)
            """, (thread_id, desc, 'active', meta.sequence))
            # keep the earliest opening sequence if seen out of order
            self.conn.execute(
                "UPDATE plot_threads SET opened_at_sequence = MIN(opened_at_sequence, ?) "
                "WHERE thread_id = ?", (meta.sequence, thread_id))
            self.conn.execute(
                "INSERT OR IGNORE INTO chunk_threads (chunk_id, thread_id, role) VALUES (?,?,?)",
                (meta.chunk_id, thread_id, 'active'))

        for thread_id in meta.plot_threads_resolved:
            self.conn.execute("""
                UPDATE plot_threads SET status='resolved',
                    closed_at_sequence = CASE
                        WHEN closed_at_sequence IS NULL THEN ?
                        ELSE MIN(closed_at_sequence, ?) END
                WHERE thread_id=?
            """, (meta.sequence, meta.sequence, thread_id))
            self.conn.execute(
                "INSERT OR IGNORE INTO chunk_threads (chunk_id, thread_id, role) VALUES (?,?,?)",
                (meta.chunk_id, thread_id, 'resolved'))

        # --- Scenes ---
        self.conn.execute("""
            INSERT OR IGNORE INTO scenes (scene_id, first_sequence, last_sequence)
            VALUES (?,?,?)
        """, (meta.scene_id, meta.sequence, meta.sequence))
        self.conn.execute(
            "UPDATE scenes SET first_sequence = MIN(first_sequence, ?) WHERE scene_id = ?",
            (meta.sequence, meta.scene_id))
        self.conn.execute(
            "UPDATE scenes SET last_sequence = MAX(last_sequence, ?) WHERE scene_id = ?",
            (meta.sequence, meta.scene_id))

        self.conn.commit()

    def get_character_state(self) -> dict:
        characters = {}
        for row in self.conn.execute("SELECT * FROM characters").fetchall():
            name = row["name"]
            facts = self.conn.execute(
                "SELECT fact FROM character_facts WHERE character=? ORDER BY added_at_sequence, id",
                (name,)
            ).fetchall()
            characters[name] = {
                "introduced": row["introduced_at_sequence"],
                "last_seen": row["last_seen_sequence"],
                "arc": row["arc"],
                "known_facts": [f["fact"] for f in facts],
            }
        return characters

    def get_active_threads(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM plot_threads WHERE status='active' ORDER BY opened_at_sequence"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_threads(self) -> list[dict]:
        """All threads, active and resolved (used by retroactive repair, Fix 4)."""
        rows = self.conn.execute(
            "SELECT * FROM plot_threads ORDER BY opened_at_sequence"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_thread_embeddings(self, embedder) -> dict[str, list[float]]:
        """Return {thread_id: embedding(description)} for all active threads."""
        return {
            t["thread_id"]: embedder.embed(t["description"])
            for t in self.get_active_threads()
        }

    def resolve_thread_id(self, candidate_id: str, candidate_description: str,
                          embedder, threshold: float = THREAD_MERGE_THRESHOLD,
                          return_score: bool = False,
                          thread_embeddings: Optional[dict] = None):
        """
        Return the canonical thread_id for a candidate (Fix 2).

        If an existing active thread's description is sufficiently similar to the
        candidate's, return that thread's id; otherwise return candidate_id
        unchanged. This catches paraphrase-level drift that prompt injection
        (Fix 1) misses. With return_score=True, returns (thread_id, score).

        thread_embeddings: optional precomputed {thread_id: vector} map. Pass it
        when resolving several candidates against the same thread set to avoid
        re-embedding every active thread description per candidate.
        """
        if thread_embeddings is None:
            thread_embeddings = self.get_thread_embeddings(embedder)
        if not thread_embeddings:
            return (candidate_id, 0.0) if return_score else candidate_id

        candidate_vec = embedder.embed(candidate_description)
        best_id, best_score = None, 0.0
        for existing_id, existing_vec in thread_embeddings.items():
            score = cosine(candidate_vec, existing_vec)
            if score > best_score:
                best_id, best_score = existing_id, score

        winner = best_id if best_score >= threshold else candidate_id
        return (winner, best_score) if return_score else winner

    def apply_thread_merge(self, merge_plan: dict[str, str]):
        """
        Rewrite chunk_threads and plot_threads to replace duplicate ids with
        their canonical ids throughout (Fix 4).

        UPDATE OR IGNORE handles the case where a chunk already carries the
        canonical id in the same role (the UNIQUE constraint would otherwise
        fire); the trailing DELETE clears any rows that couldn't be migrated.
        """
        for duplicate_id, canonical_id in merge_plan.items():
            if duplicate_id == canonical_id:
                continue
            self.conn.execute(
                "UPDATE OR IGNORE chunk_threads SET thread_id=? WHERE thread_id=?",
                (canonical_id, duplicate_id))
            self.conn.execute(
                "DELETE FROM chunk_threads WHERE thread_id=?", (duplicate_id,))
            # Preserve the earliest opening sequence on the survivor
            self.conn.execute("""
                UPDATE plot_threads SET opened_at_sequence = MIN(
                    opened_at_sequence,
                    COALESCE((SELECT opened_at_sequence FROM plot_threads WHERE thread_id=?), opened_at_sequence)
                ) WHERE thread_id=?
            """, (duplicate_id, canonical_id))
            self.conn.execute(
                "DELETE FROM plot_threads WHERE thread_id=?", (duplicate_id,))
        self.conn.commit()

    def get_chunks_by_sequence(self, sequences: list[int]) -> list[ChunkMetadata]:
        if not sequences:
            return []
        placeholders = ",".join("?" * len(sequences))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE sequence IN ({placeholders}) ORDER BY sequence",
            sequences
        ).fetchall()
        if not rows:
            return []

        # Fetch all related rows in three grouped queries rather than 3 per
        # chunk (avoids the N+1 round-trips that dominated retrieval cost).
        chunk_ids = [row["chunk_id"] for row in rows]
        cph = ",".join("?" * len(chunk_ids))

        chars_by_chunk: dict[str, list[str]] = {}
        for r in self.conn.execute(
            f"SELECT chunk_id, character FROM chunk_characters WHERE chunk_id IN ({cph})",
            chunk_ids,
        ):
            chars_by_chunk.setdefault(r["chunk_id"], []).append(r["character"])

        active_by_chunk: dict[str, list[str]] = {}
        resolved_by_chunk: dict[str, list[str]] = {}
        for r in self.conn.execute(
            f"SELECT chunk_id, thread_id, role FROM chunk_threads WHERE chunk_id IN ({cph})",
            chunk_ids,
        ):
            target = active_by_chunk if r["role"] == "active" else resolved_by_chunk
            target.setdefault(r["chunk_id"], []).append(r["thread_id"])

        return [
            ChunkMetadata(
                chunk_id=row["chunk_id"],
                sequence=row["sequence"],
                scene_id=row["scene_id"],
                arc=row["arc"],
                characters_present=chars_by_chunk.get(row["chunk_id"], []),
                location=row["location"],
                emotional_tone=row["emotional_tone"],
                plot_threads_active=active_by_chunk.get(row["chunk_id"], []),
                plot_threads_resolved=resolved_by_chunk.get(row["chunk_id"], []),
                new_facts=[],
                summary=row["summary"],
                text=row["text"],
            )
            for row in rows
        ]

    def get_recent_scenes(self, n: int = 3) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM scenes ORDER BY last_sequence DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_max_sequence(self) -> int:
        row = self.conn.execute("SELECT MAX(sequence) AS m FROM chunks").fetchone()
        return row["m"] or 0

    def get_all_sequences(self) -> list[int]:
        rows = self.conn.execute("SELECT sequence FROM chunks ORDER BY sequence").fetchall()
        return [r["sequence"] for r in rows]

    def iter_chunk_texts(self):
        """Yield (chunk_id, sequence, text) for every chunk, ordered.

        Used by re-embedding, which needs only the text and avoids the
        character/thread joins that get_chunks_by_sequence performs.
        """
        for r in self.conn.execute(
            "SELECT chunk_id, sequence, text FROM chunks ORDER BY sequence"
        ):
            yield r["chunk_id"], r["sequence"], r["text"]


# ---------------------------------------------------------------------------
# Vector store (batched writes; #8)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Lightweight in-process vector store with JSON persistence.

    Writes are batched: upsert(save=False) defers disk I/O so a bulk ingest can
    flush once with save(), avoiding the O(n^2) rewrite cost of saving per chunk.
    Swap this class for a ChromaDB/Qdrant backend in production.
    """

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self._dirty = False
        if self.store_path.exists():
            with open(self.store_path) as f:
                self._data = json.load(f)

    def save(self):
        if self._dirty:
            with open(self.store_path, "w") as f:
                json.dump(self._data, f)
            self._dirty = False

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return cosine(a, b)

    def upsert(self, chunk_id: str, embedding: list[float], sequence: int, save: bool = True):
        self._data[chunk_id] = {"embedding": embedding, "sequence": sequence}
        self._dirty = True
        if save:
            self.save()

    def clear(self, save: bool = False):
        """Drop all vectors (used before a full re-embed)."""
        self._data = {}
        self._dirty = True
        if save:
            self.save()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, int, float]]:
        scored = [
            (cid, e["sequence"], self._cosine(query_embedding, e["embedding"]))
            for cid, e in self._data.items()
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Embedding model: persistent + incremental (#1, #2)
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Minimal TF-IDF embedder for offline/no-dependency use.

    State (vocabulary, document frequencies, IDF, doc count) is persisted to
    disk and accumulated incrementally via add_documents(). Because IDF shifts
    whenever the corpus grows, every stored vector becomes stale after new
    documents are added; the orchestrator handles this by re-embedding all
    chunks on ingest. Replace embed() with a real model (sentence-transformers
    or an Ollama embedding endpoint) for production-quality retrieval.
    """

    def __init__(self, dim: int = 512, path: Optional[str] = None):
        self._vocab: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._dim = dim
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self._load()

    def _load(self):
        with open(self.path) as f:
            data = json.load(f)
        self._vocab = data["vocab"]
        self._df = data["df"]
        self._idf = data["idf"]
        self._doc_count = data["doc_count"]
        self._dim = data["dim"]

    def _save(self):
        if not self.path:
            return
        with open(self.path, "w") as f:
            json.dump({
                "vocab": self._vocab, "df": self._df, "idf": self._idf,
                "doc_count": self._doc_count, "dim": self._dim,
            }, f)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def add_documents(self, texts: list[str]):
        """
        Incrementally fold new documents into the corpus statistics.
        Extends the vocabulary, updates document frequencies, recomputes IDF,
        and persists. Does NOT reset existing state (fixes #1, #2).
        """
        for text in texts:
            tokens = set(self._tokenize(text))
            self._doc_count += 1
            for t in tokens:
                if t not in self._vocab and len(self._vocab) < self._dim:
                    self._vocab[t] = len(self._vocab)
                self._df[t] = self._df.get(t, 0) + 1
        self._idf = {
            t: math.log((self._doc_count + 1) / (self._df.get(t, 0) + 1)) + 1
            for t in self._vocab
        }
        self._save()

    def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = [0.0] * self._dim
        for token, count in tf.items():
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] = (count / total) * self._idf.get(token, 1.0)
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


# ---------------------------------------------------------------------------
# LLM metadata extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a narrative analyst. Extract structured metadata from this story/roleplay chunk.

EXISTING PLOT THREADS — use these exact IDs if the chunk touches them:
{existing_threads}

If a thread in the chunk matches one above, reuse that exact thread_id. Only
create a new snake_case thread_id if the chunk introduces a genuinely new thread
not represented in the list above.

Respond ONLY with valid JSON, no preamble, no markdown fences.

Required fields:
{{
  "scene_id": "short_snake_case_scene_name",
  "arc": "act_1 | act_2 | act_3 | or custom arc name",
  "characters_present": ["Character1", "Character2"],
  "location": "location name",
  "emotional_tone": "one word: tense | warm | ominous | playful | sad | neutral | etc",
  "plot_threads_active": ["existing_or_new_thread_id", ...],
  "plot_threads_resolved": ["thread_id_if_any_closed", ...],
  "plot_thread_descriptions": {{"thread_id": "one-line description of the thread", ...}},
  "new_facts": ["brief fact established in this chunk", ...],
  "summary": "one sentence summary of this chunk"
}}

Story chunk:
{chunk}
"""


def extract_metadata_llm(chunk: str, llm_fn: Callable[[str], str],
                         existing_threads: Optional[list[dict]] = None) -> dict:
    """
    Call an LLM to extract narrative metadata. llm_fn: prompt -> response.

    existing_threads (Fix 1): the current active-thread registry, injected into
    the prompt so the model reuses canonical ids instead of inventing new ones
    for threads it has already seen.
    """
    if existing_threads:
        thread_context = "\n".join(
            f"  - {t['thread_id']}: {t.get('description', t['thread_id'])}"
            for t in existing_threads)
    else:
        thread_context = "  (none yet — this is the first chunk)"

    prompt = EXTRACTION_PROMPT.format(
        existing_threads=thread_context, chunk=chunk[:3000])
    response = llm_fn(prompt)
    clean = re.sub(r"```(?:json)?", "", response).strip().strip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "scene_id": f"scene_{uuid.uuid4().hex[:6]}",
            "arc": "unknown",
            "characters_present": [],
            "location": "unknown",
            "emotional_tone": "neutral",
            "plot_threads_active": [],
            "plot_threads_resolved": [],
            "plot_thread_descriptions": {},
            "new_facts": [],
            "summary": chunk[:100] + "...",
        }


# ---------------------------------------------------------------------------
# Text chunking  (renamed from chunk_text -> split_into_chunks; fixes #3)
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph boundaries.

    Renamed from chunk_text() to remove the collision with the per-chunk loop
    variable in NarrativeRAG.ingest_text() (review item #3).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            overlap_paras: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) <= overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current = overlap_paras
            current_len = overlap_len
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# Backwards-compatible alias (deprecated; prefer split_into_chunks)
chunk_text = split_into_chunks


# ---------------------------------------------------------------------------
# Canonical thread registry (Fix 3)
# ---------------------------------------------------------------------------

class ThreadRegistry:
    """
    Authoritative, auditable record of canonical plot-thread ids and their
    aliases, persisted to thread_registry.json alongside the scaffold.

    Every merge decision is logged (never silently discarded), which makes
    merges reversible (undo_merge) and lets a reviewer audit drift patterns.
    The registry is a separate file, so it survives scaffold corruption.
    """

    def __init__(self, registry_path: str):
        self.path = Path(registry_path)
        self._data: dict = {}
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        # Reverse index {alias -> canonical_id} so resolve() is O(1) instead of
        # scanning every canonical entry on each lookup.
        self._alias_index: dict[str, str] = {}
        for canonical_id, entry in self._data.items():
            for alias in entry.get("aliases", []):
                self._alias_index[alias] = canonical_id

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def description_of(self, thread_id: str) -> Optional[str]:
        """Return the stored description for a canonical id, or None."""
        entry = self._data.get(thread_id)
        return entry["description"] if entry else None

    def register(self, thread_id: str, description: str, opened_at: int):
        """Register a new canonical thread (no-op if already known)."""
        if thread_id not in self._data:
            self._data[thread_id] = {
                "canonical_id": thread_id,
                "description": description,
                "status": "active",
                "opened_at": opened_at,
                "aliases": [],
                "merge_log": [],
                "evidence_chunks": [opened_at],
            }
            self._save()

    def record_alias(self, canonical_id: str, absorbed_id: str,
                     absorbed_at_sequence: int,
                     similarity_score: Optional[float] = None,
                     method: str = "embedding"):
        """Record that absorbed_id was merged into canonical_id."""
        entry = self._data.get(canonical_id)
        if entry is None:
            return
        if absorbed_id not in entry["aliases"]:
            entry["aliases"].append(absorbed_id)
            self._alias_index[absorbed_id] = canonical_id
            entry["merge_log"].append({
                "absorbed_id": absorbed_id,
                "similarity_score": similarity_score,
                "absorbed_at_sequence": absorbed_at_sequence,
                "method": method,
                "timestamp": datetime.utcnow().isoformat(),
            })
            self._save()

    def resolve(self, thread_id: str) -> str:
        """Return the canonical id for any known id or alias (O(1) lookup)."""
        if thread_id in self._data:
            return thread_id
        return self._alias_index.get(thread_id, thread_id)

    def add_evidence(self, canonical_id: str, sequence: int):
        entry = self._data.get(canonical_id)
        if entry is not None and sequence not in entry["evidence_chunks"]:
            entry["evidence_chunks"].append(sequence)
            self._save()

    def undo_merge(self, absorbed_id: str, canonical_id: str,
                   description: Optional[str] = None):
        """
        Manually split a bad merge: drop the alias from the canonical entry and
        re-register absorbed_id as its own thread. Follow with
        ScaffoldIndex.apply_thread_merge / a re-ingest to fix chunk_threads.
        """
        entry = self._data.get(canonical_id)
        if entry is not None:
            entry["aliases"] = [a for a in entry["aliases"] if a != absorbed_id]
            entry["merge_log"] = [m for m in entry["merge_log"]
                                  if m["absorbed_id"] != absorbed_id]
            self._alias_index.pop(absorbed_id, None)
            self._save()
        self.register(absorbed_id,
                      description=description or f"[restored from merge with {canonical_id}]",
                      opened_at=0)

    def get_all_canonical(self) -> list[dict]:
        return list(self._data.values())


# ---------------------------------------------------------------------------
# Retroactive repair + health check (Fix 4)
# ---------------------------------------------------------------------------

def repair_thread_duplicates(scaffold: "ScaffoldIndex",
                             registry: "ThreadRegistry",
                             embedder,
                             threshold: float = THREAD_MERGE_THRESHOLD,
                             dry_run: bool = True) -> dict:
    """
    Detect and merge duplicate plot threads across the entire scaffold (Fix 4).

    Always run with dry_run=True first and review the proposed plan. Back up
    scaffold.db before applying. Returns {duplicate_id: canonical_id}.
    """
    all_threads = scaffold.get_all_threads()
    embeddings = {t["thread_id"]: embedder.embed(t["description"]) for t in all_threads}
    merge_plan: dict[str, str] = {}

    for i, t1 in enumerate(all_threads):
        if t1["thread_id"] in merge_plan:
            continue
        for t2 in all_threads[i + 1:]:
            if t2["thread_id"] in merge_plan:
                continue
            sim = cosine(embeddings[t1["thread_id"]], embeddings[t2["thread_id"]])
            if sim >= threshold:
                # Canonical = whichever opened earlier
                canonical = min(t1, t2, key=lambda t: t["opened_at_sequence"] or 0)
                duplicate = t1 if canonical is t2 else t2
                merge_plan[duplicate["thread_id"]] = canonical["thread_id"]

    if dry_run:
        print(f"Dry run — {len(merge_plan)} merge(s) proposed:")
        for dup, canon in merge_plan.items():
            print(f"  {dup} -> {canon}")
        return merge_plan

    scaffold.apply_thread_merge(merge_plan)
    # Build a description lookup so canonical entries can be registered if the
    # registry has never seen them (e.g. a scaffold built before Fix 3 existed).
    desc_by_id = {t["thread_id"]: t.get("description", t["thread_id"])
                  for t in all_threads}
    opened_by_id = {t["thread_id"]: (t.get("opened_at_sequence") or 0)
                    for t in all_threads}
    for dup_id, canon_id in merge_plan.items():
        registry.register(canon_id, desc_by_id.get(canon_id, canon_id),
                          opened_by_id.get(canon_id, 0))
        registry.record_alias(canonical_id=canon_id, absorbed_id=dup_id,
                              absorbed_at_sequence=0, method="retroactive_repair")
    print(f"Repair complete. {len(merge_plan)} thread(s) merged.")
    return merge_plan


def check_thread_health(scaffold: "ScaffoldIndex", warn_threshold: int = 15) -> int:
    """
    Warn when the active-thread count exceeds a plausibility threshold — a
    common signature of drift/fragmentation rather than real complexity.
    Returns the active-thread count.
    """
    active = scaffold.get_active_threads()
    if len(active) > warn_threshold:
        print(f"Warning: {len(active)} active threads detected. Consider running "
              f"repair_thread_duplicates(..., dry_run=True) to check for drift.")
    return len(active)


# ---------------------------------------------------------------------------
# Main NarrativeRAG class
# ---------------------------------------------------------------------------

class NarrativeRAG:
    """
    Narrative-aware RAG system.

    Parameters
    ----------
    path : str
        Directory for the scaffold DB, vector store, and embedder state.
    neighbor_window : int
        Sequence neighbors to expand around each semantic hit.
    chunk_size : int
        Approximate character size of each ingested chunk.
    recent_count : int
        How many trailing chunks to always include in retrieval.
    """

    def __init__(self, path: str = "./narrative_index",
                 neighbor_window: int = 1,
                 chunk_size: int = 800,
                 recent_count: int = 2,
                 thread_merge_threshold: float = THREAD_MERGE_THRESHOLD):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.neighbor_window = neighbor_window
        self.chunk_size = chunk_size
        self.recent_count = recent_count
        self.thread_merge_threshold = thread_merge_threshold

        self.scaffold = ScaffoldIndex(str(self.path / "scaffold.db"))
        self.vectors = VectorStore(str(self.path / "vectors.json"))
        # Persistent embedder: loads accumulated vocab/IDF if present (#2).
        self.embedder = EmbeddingModel(path=str(self.path / "embedder.json"))
        # Canonical thread registry: alias tracking + auditable merges (Fix 3).
        self.registry = ThreadRegistry(str(self.path / "thread_registry.json"))

    def ingest_text(self, raw_text: str,
                    llm_fn: Optional[Callable[[str], str]] = None,
                    verbose: bool = True):
        """
        Ingest a raw text block (conversation, chapter, etc.). Additive.

        Steps:
          1. Chunk the new text.
          2. Fold the new chunk texts into the embedder corpus FIRST, so the
             TF-IDF vocabulary is populated before thread-resolution embeds any
             descriptions (otherwise early descriptions embed to zero vectors).
          3. For each chunk: extract metadata with the current active-thread
             registry injected into the prompt (Fix 1); resolve every candidate
             thread id through the registry + embedding pipeline (Fixes 2-3);
             then write the chunk to the scaffold with canonical ids.
          4. Re-embed ALL chunks so vectors stay commensurable, flush once.
        """
        segments = split_into_chunks(raw_text, chunk_size=self.chunk_size)
        base_sequence = self.scaffold.get_max_sequence()

        # Populate corpus statistics before any description embedding (Fix 2).
        self.embedder.add_documents(segments)

        for i, segment in enumerate(segments):
            sequence = base_sequence + i + 1
            if verbose:
                print(f"  Ingesting chunk {sequence}/{base_sequence + len(segments)}...")

            if llm_fn:
                # Fix 1: inject the live active-thread registry into the prompt.
                existing = self.scaffold.get_active_threads()
                meta_dict = extract_metadata_llm(segment, llm_fn, existing_threads=existing)
            else:
                meta_dict = self._heuristic_metadata(segment, sequence)

            descriptions = meta_dict.get("plot_thread_descriptions", {}) or {}

            # Fixes 2-3: resolve candidate ids to canonical ones before writing.
            resolved_active, resolved_desc = self._resolve_threads(
                meta_dict.get("plot_threads_active", []), descriptions, sequence)
            resolved_done, resolved_desc2 = self._resolve_threads(
                meta_dict.get("plot_threads_resolved", []), descriptions, sequence)
            resolved_desc.update(resolved_desc2)

            meta = ChunkMetadata(
                chunk_id=str(uuid.uuid4()),
                sequence=sequence,
                scene_id=meta_dict.get("scene_id", f"scene_{sequence:04d}"),
                arc=meta_dict.get("arc", "unknown"),
                characters_present=meta_dict.get("characters_present", []),
                location=meta_dict.get("location", "unknown"),
                emotional_tone=meta_dict.get("emotional_tone", "neutral"),
                plot_threads_active=resolved_active,
                plot_threads_resolved=resolved_done,
                new_facts=meta_dict.get("new_facts", []),
                summary=meta_dict.get("summary", segment[:80]),
                text=segment,
                plot_thread_descriptions=resolved_desc,
            )
            self.scaffold.upsert_chunk(meta)
            for tid in set(resolved_active + resolved_done):
                self.registry.add_evidence(tid, sequence)

        # Re-embed ALL chunks against the updated IDF, flush once (#1, #8).
        self._reembed_all(verbose=verbose)

        if verbose:
            print(f"  Ingested {len(segments)} chunks. Total: {self.scaffold.get_max_sequence()}")

    def _resolve_threads(self, candidate_ids: list[str], descriptions: dict,
                         sequence: int) -> tuple[list[str], dict]:
        """
        Map a list of candidate thread ids to canonical ids (Fixes 1-3 merge
        point). Returns (canonical_ids, {canonical_id: description}).

        Resolution order:
          Phase 1 — registry alias lookup (free, no embedding).
          Phase 2 — embedding similarity against active scaffold threads; on a
                    hit, record the alias so it's free next time.
          Phase 3 — genuinely new: register it in the canonical registry.
        """
        canonical_ids: list[str] = []
        out_desc: dict = {}
        # Embed active-thread descriptions once for this batch of candidates
        # rather than once per candidate (the scaffold's active set is stable
        # until upsert_chunk runs after resolution).
        thread_embeddings = self.scaffold.get_thread_embeddings(self.embedder)
        for cid in candidate_ids:
            description = descriptions.get(cid, cid.replace("_", " "))

            # Phase 1: known alias?
            canonical = self.registry.resolve(cid)
            if canonical != cid:
                canonical_ids.append(canonical)
                out_desc[canonical] = self.registry.description_of(canonical) or description
                continue

            # Phase 2: embedding near-match against existing active threads?
            resolved, score = self.scaffold.resolve_thread_id(
                cid, description, self.embedder, self.thread_merge_threshold,
                return_score=True, thread_embeddings=thread_embeddings)
            if resolved != cid:
                self.registry.record_alias(
                    canonical_id=resolved, absorbed_id=cid,
                    absorbed_at_sequence=sequence,
                    similarity_score=round(score, 4), method="embedding")
                canonical_ids.append(resolved)
                out_desc[resolved] = self.registry.description_of(resolved) or description
                continue

            # Phase 3: genuinely new thread.
            self.registry.register(cid, description, sequence)
            canonical_ids.append(cid)
            out_desc[cid] = description

        return canonical_ids, out_desc

    def repair_threads(self, threshold: Optional[float] = None,
                       dry_run: bool = True) -> dict:
        """Convenience wrapper around repair_thread_duplicates (Fix 4)."""
        return repair_thread_duplicates(
            self.scaffold, self.registry, self.embedder,
            threshold=threshold or self.thread_merge_threshold, dry_run=dry_run)

    def thread_health(self, warn_threshold: int = 15) -> int:
        """Convenience wrapper around check_thread_health (Fix 4)."""
        return check_thread_health(self.scaffold, warn_threshold=warn_threshold)

    def _reembed_all(self, verbose: bool = False):
        """
        Recompute embeddings for every stored chunk using the current embedder
        state, then flush the vector store a single time. Required because the
        TF-IDF weights move whenever the corpus grows (#1).
        """
        self.vectors.clear(save=False)
        count = 0
        for chunk_id, sequence, text in self.scaffold.iter_chunk_texts():
            self.vectors.upsert(chunk_id, self.embedder.embed(text), sequence, save=False)
            count += 1
        self.vectors.save()
        if verbose:
            print(f"  Re-embedded {count} chunks against updated vocabulary.")

    def retrieve_context(self, query: str,
                         top_k: int = 5,
                         include_recent: bool = True) -> RetrievalResult:
        """
        Retrieve an ordered, state-complete narrative context window.

        Phase 1: semantic search.
        Phase 2: neighbor expansion (restore local context).
        Phase 3: chronological sort.
        Phase 4: state merge (characters + active threads).
        """
        query_embedding = self.embedder.embed(query)
        hits = self.vectors.search(query_embedding, top_k=top_k * 2)

        all_seqs = self.scaffold.get_all_sequences()
        all_seq_set = set(all_seqs)
        expanded_seqs: set[int] = set()

        for _, sequence, _ in hits[:top_k]:
            for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                candidate = sequence + offset
                if candidate in all_seq_set:
                    expanded_seqs.add(candidate)

        # Always include the actual trailing chunks, robust to gaps (#7)
        if include_recent and all_seqs:
            for s in all_seqs[-self.recent_count:]:
                expanded_seqs.add(s)

        ordered_chunks = self.scaffold.get_chunks_by_sequence(sorted(expanded_seqs))

        return RetrievalResult(
            chunks=ordered_chunks,
            characters=self.scaffold.get_character_state(),
            active_threads=self.scaffold.get_active_threads(),
            recent_scenes=self.scaffold.get_recent_scenes(n=3),
        )

    def get_story_state(self) -> dict:
        return {
            "total_chunks": self.scaffold.get_max_sequence(),
            "characters": self.scaffold.get_character_state(),
            "active_threads": self.scaffold.get_active_threads(),
            "recent_scenes": self.scaffold.get_recent_scenes(n=5),
        }

    def _heuristic_metadata(self, text: str, sequence: int) -> dict:
        """Fallback metadata extraction without an LLM (naive; demo only)."""
        stop = {"the", "a", "an", "and", "but", "or", "so", "yet", "he", "she",
                "they", "it", "we", "i", "you", "his", "her", "their", "its",
                "not", "missing"}
        cleaned = (re.sub(r"['\u2019]s\b", "", text))  # drop possessive 's
        caps = list({
            re.sub(r"[^A-Za-z]", "", w) for w in cleaned.split()
            if w and w[0].isupper() and len(re.sub(r"[^A-Za-z]", "", w)) > 2
            and w.lower().strip(".,!?\"'") not in stop
        })[:4]
        caps = [c for c in caps if c]
        return {
            "scene_id": f"scene_{sequence:04d}",
            "arc": "unknown",
            "characters_present": caps,
            "location": "unknown",
            "emotional_tone": "neutral",
            "plot_threads_active": [],
            "plot_threads_resolved": [],
            "new_facts": [],
            "summary": text[:120].replace("\n", " ") + "...",
        }


# ---------------------------------------------------------------------------
# Ollama integration helper
# ---------------------------------------------------------------------------

def make_ollama_fn(model: str = "mistral", host: str = "http://localhost:11434"):
    """Return an llm_fn that calls a local Ollama instance for extraction."""
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests to use Ollama integration")

    def call_ollama(prompt: str) -> str:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]

    return call_ollama


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("NarrativeRAG — demo mode (no LLM, heuristic metadata)\n")

    sample = """
    Mira stepped from the shadows of the collapsed tower, her cloak still damp
    from the river crossing. The Warden stood at the far end of the hall,
    back turned, studying a map pinned to the crumbling wall.

    "You knew about the vault," she said. Not a question.

    He turned slowly. There was no surprise in his face. "I've known since
    before you were sent here. The question is what you intend to do with
    the information."

    Mira's hand drifted toward the hilt at her hip. The oath she had sworn
    three years ago in the capital suddenly felt very far away.

    "The key," she said. "Where is it?"

    The Warden smiled for the first time — a thin, tired thing. "That's
    exactly what Renn asked me before he disappeared."

    The name hit her like cold water. Renn. Her brother. Missing for seven months.

    "You have ten seconds to start talking," she said, and drew the blade.
    """

    import tempfile
    demo_dir = tempfile.mkdtemp(prefix="narrative_rag_demo_")
    rag = NarrativeRAG(path=demo_dir)
    rag.ingest_text(sample, verbose=True)

    print("\n--- Story State ---")
    state = rag.get_story_state()
    print(f"Characters: {list(state['characters'].keys())}")
    print(f"Active threads: {[t['thread_id'] for t in state['active_threads']]}")

    print("\n--- Retrieval: 'confrontation about the vault' ---")
    result = rag.retrieve_context("confrontation about the vault")
    print(result.render())
