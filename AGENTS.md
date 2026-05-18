# AGENTS.md

## Purpose
This repository demonstrates a local, production-style RAG agent for portfolio and interview use.

## Engineering Notes
- Keep the system deterministic and grounded.
- Prefer maintainable Python modules over framework-heavy abstractions.
- Every factual answer should cite retrieved chunks.
- Refuse with the exact project refusal message when context is insufficient.
- Preserve local-first deployment assumptions: Ollama on the host machine, Qdrant in Docker, Streamlit for demo UX.
