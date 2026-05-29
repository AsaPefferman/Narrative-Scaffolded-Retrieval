Narrative-Scaffolded Retrieval (NSR)
A reference implementation and white paper for chronologically-coherent RAG
for long-form generative fiction and roleplay.
The problem
Conventional RAG is built for factual Q&A: it retrieves semantically similar
chunks with no regard for when they happened in a story. For narrative, order
is meaning. NSR pairs a semantic vector store with a structural scaffold
index (sequence, scenes, characters, plot-thread lifecycle), and reassembles
retrieved material into a coherent, chronologically-ordered, state-complete
context window.
Contents
File	Description
`docs/NSR_white_paper.docx`	The white paper (RFC v0.1) for review
`src/narrative_rag.py`	Single-module reference implementation
`build_whitepaper.js`	Script that generates the white paper
Quick start
```bash
# Run the offline demo (no LLM, heuristic metadata)
python3 src/narrative_rag.py
```
```python
from narrative_rag import NarrativeRAG, make_ollama_fn

rag = NarrativeRAG(path="./my_story")
llm = make_ollama_fn(model="mistral")   # local Ollama

rag.ingest_text(open("chapter_01.txt").read(), llm_fn=llm)

ctx = rag.retrieve_context("the confrontation about the vault")
print(ctx.render())   # paste into your next generation prompt
```
Architecture in one diagram
```
        ingest                         retrieve
        ──────                         ────────
   text → chunks                  query → embed
        │                              │
   ┌────┴─────┐                   semantic search (vector store)
   │          │                        │
 extract    embed                 neighbor expansion (±window)
 metadata     │                        │
   │          │                  chronological sort
   ▼          ▼                        │
 SCAFFOLD   VECTOR                state merge (scaffold)
 (SQLite)   STORE                       │
   └────┬─────┘                         ▼
        │                    ordered, state-complete window
   bidirectional
   coupling
```
Status
This is a request for comment. The white paper enumerates open questions
(extraction fidelity, contradiction handling, merge weighting, evaluation).
Critique welcome.
Production swap-outs
Replace `VectorStore` with ChromaDB or Qdrant
Replace `EmbeddingModel` (TF-IDF fallback) with a real embedding model
(e.g. `nomic-embed-text` via Ollama, or sentence-transformers)
Replace `make_ollama_fn` target model with your preferred local backend
