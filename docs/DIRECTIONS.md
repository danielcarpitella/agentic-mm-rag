# DIRECTIONS.md — Possible future directions for the project

---



## Direction A — Agentic *composed* retrieval (recommended)

**Idea.** Today the agent searches only with text: `SEARCH("...")`. Extend the protocol  
with a second tool: `REFINE(image_k, "text modification")`. The model inspects the  
results, critiques them ("right period, but the interior is needed, not the facade"), and launches  
a **composed image+text** query, implemented training-free with CLIP embedding arithmetic / a combiner (as in notebook 02). Retrieval becomes multi-turn and self-correcting.

**Feasibility.** High: the retriever already exists; the CLIP encoding of the starting
image is needed for the composed query, along with a branch in the trigger parser.

**Possible extension (not necessarily to be implemented):** refinement feedback also provided by the **user**, not only by the model.

## Direction B — LMM as verifier / reranker (self-verifying RAG)

**Idea.** CLIP retrieves broadly (k=15–20); the LMM inspects the candidates and discards
irrelevant ones before answering; for each claim, the final answer cites which image
supports what.

**Why.** It addresses CLIP's real weakness (similarity ≠ relevance) and would be easy to measure.

- composable with A or a fallback if A proceeds quickly.



## Direction C — Systematic evaluation (multiplier, not an alternative)

**Idea.** Mini-benchmark of 50–100 visual-knowledge questions in the chosen domain, comparing three
conditions:

1. no-RAG (LMM only)
2. RAG single-pass (retrieve-once)
3. agentic (loop)

**Metrics:** accuracy (LLM judge), average number of searches.......

- if possible, to be combined with A (or B), not an alternative.



## Direction D — Scale + demo with visible trace (packaging)

**Idea.** Real dataset of 2,000–5,000 items (Wikipedia / WikiWeb2M subset, as required) + Gradio interface (web interface to show everything more clearly, including retrieved images) that shows the agent's reasoning **live**:
issued queries, retrieved images, critiques, illustrated final answer.

---

