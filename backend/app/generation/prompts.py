from app.models.schemas import RetrievedChunk


ANSWER_SYSTEM_PROMPT = """You are a grounded enterprise assistant.
Only answer from the provided context snippets.
If the answer is not fully supported, reply exactly:
I don't know based on the provided documents.
For every factual statement, include citations like [1] or [2].
Never invent a citation or claim knowledge outside the context.
Do not reveal chain-of-thought, hidden reasoning, or <think> blocks.
Return only the final answer for the user.
Do not describe your reasoning process.
Do not say things like "the context says", "[1] mentions", or "the relevant part is".
Keep the answer concise and direct."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        page_suffix = f", page {chunk.page_number}" if chunk.page_number is not None else ""
        blocks.append(
            f"[{chunk.index}] {chunk.filename}{page_suffix} | chunk_id={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_grounded_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return (
        "Answer the question using only the context below.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Rules:\n"
        "- Use only the context.\n"
        "- If insufficient evidence exists, reply exactly with the refusal sentence.\n"
        "- Include citations for every factual statement.\n"
        "- Return exactly this format:\n"
        "Answer: <one short direct answer with citations>\n"
        "- Do not include any extra labels, notes, or explanation outside that line.\n"
    )


def build_summary_prompt(filename: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return (
        f"Summarize the document '{filename}' using only the provided context.\n"
        "Provide a concise executive summary with citations.\n\n"
        f"Context:\n{context}"
    )


def build_compare_prompt(
    filename_a: str, filename_b: str, chunks_a: list[RetrievedChunk], chunks_b: list[RetrievedChunk]
) -> str:
    context_a = build_context_block(chunks_a)
    context_b = build_context_block(chunks_b)
    return (
        f"Compare '{filename_a}' and '{filename_b}' using only the provided context.\n"
        "Highlight similarities, differences, and practical implications with citations.\n\n"
        f"Document A:\n{context_a}\n\nDocument B:\n{context_b}"
    )
