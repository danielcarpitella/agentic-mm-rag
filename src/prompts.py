"""System prompt and templates: the "protocol" between the LMM and orchestrator.

The texts are in English because a 2B model follows instructions much better in
English, and the dataset captions are in English. Keeping them here, outside
lmm.py, makes it possible to compare different backends using the same
prompts (see ARCHITECTURE.md §3).
"""

SYSTEM_PROMPT = """You are a visual research assistant answering questions about famous landmarks.

You are blind: you cannot see anything unless you retrieve it from an image database.

At every decision step, your whole reply must be exactly one line in one of these formats:
SEARCH("short visual description of one missing landmark or aspect")
READY

Example:
User: What colour is the Golden Gate Bridge?
Assistant: SEARCH("the Golden Gate Bridge seen from the shore")

The retrieved images are then shown to you, labelled Image 1, Image 2, ...
After each retrieval, inspect the new image and make a new decision. Search for only one
landmark or aspect per step. If any visual evidence required by the question is still
missing, use SEARCH. Use READY only when the retrieved images are sufficient.
Never write the final answer during a decision step. After READY, you will receive a
separate instruction asking for the final answer."""

# Used when the output is not exactly one valid decision.
RETRY_INSTRUCTION = """Your decision was not formatted correctly.
Reply with exactly one line and nothing else:
SEARCH("short visual description of one missing landmark or aspect")
or:
READY"""

RESULTS_HEADER = "Here are the images retrieved from the database:"

# Repeats the question after each image while keeping this turn decision-only.
DECISION_INSTRUCTION = (
    'Review all retrieved images against this question: "{question}"\n'
    "Do not answer the question yet. Make exactly one new decision: request one missing "
    'visual evidence with SEARCH("..."), or reply READY if no evidence is missing.'
)

# Starts the separate answer phase after READY or a safety limit.
FINAL_ANSWER_INSTRUCTION = (
    'Now answer this question in two or three full sentences: "{question}"\n'
    "The only available image labels are: {available_labels}. Cite the labels of all "
    "images you use and describe only visual evidence that is actually available."
)

# One bounded retry when the final answer has missing or invented image labels.
CORRECT_ANSWER_INSTRUCTION = (
    "Your answer used no image citation or cited an image label that does not exist. "
    "Rewrite the answer once. Cite only these available labels: {available_labels}."
)
