"""Demo UI: python app.py [--config CONFIG] [--no-thread]

Left column: the same frozen model answering WITHOUT retrieval.
Right column: the agentic loop as a chronological timeline, streamed one event at
a time from the orchestrator's on_event hook (see ARCHITECTURE.md §2.3).

The UI never touches the loop: it only renders the typed events it receives. A
new event type (for example a future "thinking" step) shows up as a generic card
until it gets its own rendering branch in render_event().
"""

import argparse
import base64
import html
import io
import json
import os
import queue
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

# Same OpenMP workaround as main.py (mlx + torch/faiss in one process).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gradio as gr
from PIL import Image

from src.config import load_config
from src.lmm import LMM, Message
from src.orchestrator import PARENTHETICAL_IMAGE_LABEL_PATTERN, Orchestrator
from src.prompts import BASELINE_ANSWER_INSTRUCTION, BASELINE_SYSTEM_PROMPT
from src.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

# Palette shared with the design mockup.
INK = "#1c1b18"
MUTED = "#6b685f"
BORDER = "#dedbd2"
TEAL = "#1f7a86"
TEAL_BG = "#e3f1f3"
TEAL_INK = "#155a63"
AMBER = "#b0741c"
AMBER_BG = "#fbf5e9"
AMBER_BORDER = "#e6d3ad"
GREEN = "#3d7a3a"
GREEN_BG = "#ecf4ec"
RED = "#9b3b2e"
RED_BG = "#fbf0ee"
MONO = "font-family: 'IBM Plex Mono', Menlo, 'Courier New', monospace;"


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------


@lru_cache(maxsize=256)
def thumbnail_data_uri(image_path: str) -> str:
    """Small base64 JPEG so gr.HTML can show local images without file serving."""
    with Image.open(PROJECT_ROOT / image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((240, 240))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


# ----------------------------------------------------------------------------
# Rendering (pure functions: events in, HTML out)
# ----------------------------------------------------------------------------


def _tag(text: str, color: str) -> str:
    return (
        f'<div style="font-size:10.5px;font-weight:600;color:{color};'
        f'width:100px;flex-shrink:0;letter-spacing:.04em">{text}</div>'
    )


def _card(inner: str, accent: str | None = None, bg: str = "#fff", border: str = BORDER) -> str:
    left = f"border-left:3px solid {accent};" if accent else ""
    return (
        f'<div style="display:flex;gap:12px;align-items:center;padding:7px 12px;'
        f'background:{bg};border:1px solid {border};{left}border-radius:5px;'
        f'flex-grow:1;min-width:0;flex-wrap:wrap">{inner}</div>'
    )


def _row(step_label: str, card: str) -> str:
    return (
        f'<div style="display:flex;gap:12px;align-items:center">'
        f'<div style="{MONO}width:46px;flex-shrink:0;font-size:11px;color:{MUTED}">'
        f"{step_label}</div>{card}</div>"
    )


def _labels(labels: list[int]) -> str:
    return ", ".join(f"Image {label}" for label in labels)


def highlight_citations(text: str) -> str:
    """Escape the answer and turn every (Image N) into a highlighted chip."""
    escaped = html.escape(text)
    return PARENTHETICAL_IMAGE_LABEL_PATTERN.sub(
        lambda m: (
            f'<span style="background:{TEAL_BG};color:{TEAL_INK};padding:1px 5px;'
            f'border-radius:3px;font-weight:500">Image {m.group(1)}</span>'
        ),
        escaped,
    )


def render_event(event: dict, hits_by_label: dict[int, dict]) -> str:
    """One timeline row per event. Unknown types fall through to a generic card."""
    kind = event["type"]
    step = event.get("step")
    step_label = f"step {step}" if step is not None else ""

    if kind == "decision":
        line = html.escape(first_line(event["raw"]))
        if line.upper().rstrip(".!") == "READY":
            inner = (
                _tag("MODEL", GREEN)
                + f'<div style="{MONO}font-size:12.5px">READY</div>'
                + f'<div style="font-size:12px;color:{MUTED}">evidence sufficient · generating final answer</div>'
            )
            return _row(step_label, _card(inner, accent=GREEN))
        inner = _tag("MODEL", TEAL) + f'<div style="{MONO}font-size:12.5px">{line}</div>'
        return _row(step_label, _card(inner, accent=TEAL))

    if kind == "retrieval":
        rows = []
        for hit in event["hits"]:
            name = html.escape(hit["id"].replace("_", " "))
            inner = (
                _tag("RETRIEVER", MUTED)
                + f'<img src="{thumbnail_data_uri(hit["image_path"])}" '
                f'style="width:60px;height:40px;object-fit:cover;border-radius:3px">'
                + f'<div style="font-size:12.5px;font-weight:500">Image {hit["label"]} · {name}</div>'
                + f'<div style="{MONO}font-size:11px;color:{MUTED}">CLIP {hit["score"]:.3f} · added to context</div>'
            )
            rows.append(_row("", _card(inner)))
        return "".join(rows)

    if kind == "duplicate":
        what = "Duplicate of" if not event["new_evidence"] else "Partly duplicate of"
        inner = _tag("ORCHESTRATOR", AMBER) + (
            f'<div style="font-size:12.5px">{what} <span style="{MONO}">{_labels(event["labels"])}</span>'
            " · not re-inserted · model asked to decide again</div>"
        )
        return _row("", _card(inner, accent=AMBER, bg=AMBER_BG, border=AMBER_BORDER))

    if kind == "invalid_decision":
        inner = _tag("ORCHESTRATOR", AMBER) + (
            '<div style="font-size:12.5px">Decision not in protocol format · asking once more</div>'
        )
        return _row("", _card(inner, accent=AMBER, bg=AMBER_BG, border=AMBER_BORDER))

    if kind == "decision_failed":
        inner = _tag("ORCHESTRATOR", RED) + (
            '<div style="font-size:12.5px">Second invalid decision · stopping the loop</div>'
        )
        return _row("", _card(inner, accent=RED, bg=RED_BG))

    if kind == "limit":
        reason = "Image limit" if event["reason"] == "image_limit" else "Step limit"
        inner = _tag("ORCHESTRATOR", AMBER) + (
            f'<div style="font-size:12.5px">{reason} reached · forcing final answer</div>'
        )
        return _row("", _card(inner, accent=AMBER, bg=AMBER_BG, border=AMBER_BORDER))

    if kind == "final_prompt":
        inner = _tag("ORCHESTRATOR", MUTED) + (
            f'<div style="font-size:12px;color:{MUTED}">Fresh answer context · images only · '
            f"labels {_labels(event['labels'])}</div>"
        )
        return _row("", _card(inner))

    if kind == "invalid_answer":
        inner = _tag("VALIDATOR", RED) + (
            '<div style="font-size:12px;color:' + RED + '">citations missing or invalid · '
            "one clean regeneration</div>"
        )
        return _row("", _card(inner, accent=RED, bg=RED_BG))

    if kind == "answer":
        thumbs = "".join(
            f'<img src="{thumbnail_data_uri(hit["image_path"])}" '
            f'style="width:84px;height:56px;object-fit:cover;border-radius:4px">'
            for _, hit in sorted(hits_by_label.items())
        )
        body = (
            f'<div style="display:flex;gap:14px;padding:12px 14px;background:#f3f2ee;'
            f'border-radius:6px;flex-grow:1;min-width:0">'
            f'<div style="flex-grow:1;font-size:14px;line-height:1.55">{highlight_citations(event["text"])}</div>'
            f'<div style="display:flex;gap:6px;flex-shrink:0">{thumbs}</div></div>'
        )
        if event.get("no_evidence"):
            badge_color, badge_bg, badge_text = MUTED, "#f3f2ee", "no image was retrieved · no grounded answer possible"
        elif event["valid"]:
            badge_text = "citations valid · every claim carries an image label"
            if event.get("corrected"):
                badge_text += " · after one regeneration"
            badge_color, badge_bg = GREEN, GREEN_BG
        else:
            badge_color, badge_bg = RED, RED_BG
            badge_text = "validation failed · shown exactly as generated, no citations added"
        badge = (
            f'<div style="display:flex;gap:8px;align-items:center;padding:6px 10px;'
            f'background:{badge_bg};border-radius:6px;font-size:12px;color:{badge_color}">{badge_text}</div>'
        )
        return (
            f'<div style="display:flex;gap:12px;align-items:flex-start;margin-top:6px">'
            f'<div style="{MONO}width:46px;flex-shrink:0;font-size:11px;color:{MUTED};padding-top:12px">answer</div>'
            f"{body}</div>{badge}"
        )

    # Generic fallback: a future event type (e.g. "thinking") is visible at once.
    payload = {k: v for k, v in event.items() if k not in ("type", "step")}
    inner = _tag(html.escape(kind.upper()), MUTED) + (
        f'<div style="font-size:12px;color:{MUTED};white-space:pre-wrap">'
        f"{html.escape(json.dumps(payload, ensure_ascii=False, default=str))}</div>"
    )
    return _row(step_label, _card(inner, bg="#f7f6f2"))


def render_timeline(events: list[dict], elapsed: float, running: bool) -> str:
    """Whole right column: header with run stats + one row per event."""
    steps = sum(1 for e in events if e["type"] == "decision")
    searches = sum(1 for e in events if e["type"] == "search")
    duplicates = sum(1 for e in events if e["type"] == "duplicate")
    hits_by_label = {
        hit["label"]: hit for e in events if e["type"] == "retrieval" for hit in e["hits"]
    }
    stats = (
        f"{steps} steps · {searches} searches · {duplicates} duplicate blocked · "
        f"{len(hits_by_label)} images · {elapsed:.0f} s"
    )
    if running:
        stats += " · running…"
    rendered = [
        render_event(e, hits_by_label)
        for e in events
        if e["type"] not in ("question", "search", "ready")
    ]
    if not rendered:
        rendered.append(
            f'<div style="display:flex;align-items:center;justify-content:center;height:120px;'
            f'border:1px dashed {BORDER};border-radius:6px;color:{MUTED};font-size:12px">'
            + ("waiting for the first decision…" if running else "run a question to see the loop")
            + "</div>"
        )
    return (
        f'<div class="panel" style="border-color:{INK}">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<div style="font-size:14px;font-weight:600">Agentic loop</div>'
        f'<div style="{MONO}font-size:11px;color:{MUTED}">{stats}</div></div>'
        f'<div style="height:1px;background:{BORDER}"></div>'
        + "".join(rendered)
        + "</div>"
    )


def render_baseline(answer: str | None, elapsed: float | None, running: bool) -> str:
    """Left column: the same model, no retrieval, no images."""
    meta = "no retrieval · 0 images"
    if elapsed is not None:
        meta += f" · {elapsed:.0f} s"
    if running:
        meta += " · running…"
    if answer is None:
        body = (
            f'<div style="font-size:13px;color:{MUTED}">'
            + ("generating…" if running else "run a question to compare")
            + "</div>"
        )
        badge = ""
    else:
        body = f'<div style="font-size:14px;line-height:1.6">{html.escape(answer)}</div>'
        badge = (
            f'<div style="display:flex;gap:8px;align-items:center;padding:8px 10px;'
            f'background:{RED_BG};border-radius:6px;font-size:12px;color:{RED}">'
            "no image evidence · claims cannot be verified · no citations</div>"
        )
    return (
        '<div class="panel">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'<div style="font-size:14px;font-weight:600">Model alone</div>'
        f'<div style="{MONO}font-size:11px;color:{MUTED}">{meta}</div></div>'
        f'<div style="height:1px;background:{BORDER}"></div>'
        f'<div style="display:flex;align-items:center;justify-content:center;height:120px;'
        f'border:1px dashed {BORDER};border-radius:6px;color:{MUTED};font-size:12px">'
        "no visual evidence in context</div>"
        f"{body}{badge}</div>"
    )


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------


class Demo:
    """Holds the models loaded once and runs the two conditions."""

    def __init__(self, config_path: Path, threaded: bool):
        self.cfg = load_config(config_path)
        self.lmm = LMM(self.cfg.model)
        self.retriever = Retriever(self.cfg.retriever)
        self.threaded = threaded
        self.model_name = self.cfg.model.name.split("/")[-1]
        self.encoder_name = self.cfg.retriever.encoder.split("/")[-1]
        self.index_size = len(self.retriever.items)

    def run_baseline(self, question: str) -> str:
        messages = [
            Message(role="system", text=BASELINE_SYSTEM_PROMPT),
            Message(role="user", text=BASELINE_ANSWER_INSTRUCTION.format(question=question)),
        ]
        return self.lmm.generate(messages).strip()

    def stream_events(self, question: str, sink):
        """Yield orchestrator events as they happen (thread + queue), or all at
        once after a synchronous run when --no-thread is set."""
        if not self.threaded:
            events: list[dict] = []
            self._orchestrator(lambda e: (events.append(e), sink(e))).run(question)
            yield from events
            return

        pending: queue.Queue = queue.Queue()
        done = object()

        def worker() -> None:
            try:
                self._orchestrator(pending.put).run(question)
            except Exception as error:  # surfaced in the UI instead of a silent hang
                pending.put({"type": "error", "message": repr(error)})
            finally:
                pending.put(done)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            try:
                event = pending.get(timeout=0.5)
            except queue.Empty:
                yield None  # heartbeat: lets the UI refresh the elapsed time
                continue
            if event is done:
                return
            sink(event)
            yield event

    def _orchestrator(self, on_event) -> Orchestrator:
        return Orchestrator(
            lmm=self.lmm,
            retriever=self.retriever,
            cfg=self.cfg.orchestrator,
            top_k=self.cfg.retriever.top_k,
            log_dir=LOG_DIR,
            on_event=on_event,
        )


def make_run_both(demo: Demo):
    def run_both(question: str):
        question = (question or "").strip()
        if not question:
            yield render_baseline(None, None, False), render_timeline([], 0, False)
            return

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        events_path = LOG_DIR / f"events_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        events_file = events_path.open("a")

        def record(event: dict) -> None:
            events_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            events_file.flush()

        record({"type": "run_start", "question": question})

        # 1) Model alone (fast): fills the left column first.
        yield render_baseline(None, None, True), render_timeline([], 0, False)
        t0 = time.time()
        baseline = demo.run_baseline(question)
        baseline_elapsed = time.time() - t0
        record({"type": "baseline_answer", "text": baseline, "seconds": baseline_elapsed})
        left = render_baseline(baseline, baseline_elapsed, False)
        yield left, render_timeline([], 0, True)

        # 2) Agentic loop: streamed event by event.
        events: list[dict] = []
        t0 = time.time()
        for event in demo.stream_events(question, record):
            if event is not None:
                events.append(event)
            yield left, render_timeline(events, time.time() - t0, True)
        record({"type": "run_end", "seconds": time.time() - t0})
        events_file.close()
        yield left, render_timeline(events, time.time() - t0, False)

    return run_both


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important;
  font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif; background: #f3f2ee; }
.panel { display: flex; flex-direction: column; gap: 8px; padding: 16px 18px;
  background: #fff; border: 1px solid #dedbd2; border-radius: 8px; color: #1c1b18; }
/* Default ink for everything inside a panel: inline colours (muted, teal, ...)
   still win, but nothing inherits the host theme's light text on white cards. */
.panel, .panel * { color: #1c1b18; }
#title-bar, #title-bar * { color: #1c1b18; }
.chip { font-family: 'IBM Plex Mono', Menlo, monospace; font-size: 11px; padding: 4px 8px;
  border: 1px solid #dedbd2; border-radius: 4px; color: #6b685f; background: #fff; }
#title-bar { display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0 10px 0; border-bottom: 1px solid #dedbd2; margin-bottom: 8px; }
footer { display: none !important; }
"""

HEAD = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">'
)

# The demo is designed light; ignore the browser's dark-mode preference.
FORCE_LIGHT_JS = (
    "() => { document.documentElement.classList.remove('dark');"
    " document.body.classList.remove('dark'); }"
)


def build_app(demo: Demo) -> gr.Blocks:
    run_both = make_run_both(demo)

    with gr.Blocks(title="Agentic Multimodal RAG") as app:
        gr.HTML(
            '<div id="title-bar">'
            '<div style="font-size:16px;font-weight:600;color:#1c1b18">Agentic Multimodal RAG</div>'
            '<div style="display:flex;gap:8px">'
            f'<div class="chip">{html.escape(demo.model_name)} · frozen · temp {demo.cfg.model.temperature}</div>'
            f'<div class="chip">{html.escape(demo.encoder_name)} + FAISS · {demo.index_size} items</div>'
            "</div></div>"
        )
        with gr.Row():
            question = gr.Textbox(
                show_label=False,
                container=False,
                placeholder="Ask a question about a landmark…",
                scale=5,
                lines=1,
            )
            run_button = gr.Button("Run both", variant="primary", scale=1)
        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                baseline_html = gr.HTML(render_baseline(None, None, False))
            with gr.Column(scale=4, min_width=480):
                timeline_html = gr.HTML(render_timeline([], 0, False))

        run_button.click(run_both, inputs=question, outputs=[baseline_html, timeline_html])
        question.submit(run_both, inputs=question, outputs=[baseline_html, timeline_html])
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=PROJECT_ROOT / "config.yaml", type=Path)
    parser.add_argument(
        "--no-thread",
        action="store_true",
        help="run the loop synchronously (no streaming); fallback if the backend "
        "does not tolerate a worker thread",
    )
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    print("Loading the model and retriever once…")
    demo = Demo(args.config, threaded=not args.no_thread)
    build_app(demo).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        css=CSS,
        head=HEAD,
        js=FORCE_LIGHT_JS,
        footer_links=[],
        show_error=True,
    )


if __name__ == "__main__":
    main()
