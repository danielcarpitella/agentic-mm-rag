---
name: demo-ui-gradio
overview: Demo UI Gradio per la presentazione: cronologia del loop agentico in streaming (eventi tipizzati emessi dall'orchestratore) affiancata alla risposta dello stesso modello senza retrieval. Nessuna modifica alla logica del loop; struttura pronta per un futuro evento "thinking".
todos:
  - id: event-hook
    content: Aggiungere on_event/_emit in orchestrator.py accanto ai _log esistenti, con test deterministico della sequenza di eventi
    status: completed
  - id: baseline-prompts
    content: Aggiungere BASELINE_SYSTEM_PROMPT e BASELINE_ANSWER_INSTRUCTION in prompts.py
    status: completed
  - id: gradio-app
    content: Scrivere app.py (baseline + timeline in streaming via thread e queue, thumbnail base64, eventi salvati in logs/events_*.jsonl)
    status: completed
  - id: verify
    content: Test unitari verdi, evaluate_minimal invariato, prova dei 4 preset nella UI
    status: completed
  - id: sync-docs
    content: Aggiornare README, ARCHITECTURE, project-context.mdc e stato di questo piano
    status: completed
isProject: true
---

# Piano: demo UI Gradio (timeline agentica + confronto con il modello "puro")

## Contesto

Il loop agentico (SEARCH/READY, dedup per `hit.id`, cap token sulle decisioni, risposta
finale in contesto pulito con validazione delle citazioni) è al checkpoint del commit
`4430547`. Per la presentazione serve una demo live che faccia vedere a una giuria tecnica
**come** il modello decide, non solo cosa risponde. La UI scelta mescola due idee:

- a destra la **cronologia** del loop agentico, evento per evento e in streaming
  (decisione del modello → retrieval con thumbnail e score → guard dell'orchestratore →
  READY → validatore → risposta con citazioni evidenziate);
- a sinistra la risposta dello **stesso modello senza retrieval**, per il confronto diretto.

Vincoli da `.cursor/rules/project-context.mdc`: nessuna modifica alla logica del loop,
invariante "una retrieval per turno" intatto, repo cross-platform, niente astrazioni o
polish inutili, docs sincronizzate dopo una modifica stabile.

Requisito esplicito: la UI deve essere **strutturalmente pronta** per un futuro evento
"thinking" senza rifare nulla. La UI consuma un **flusso di eventi tipizzati**: aggiungere
il thinking = un tipo di evento in più + un ramo di rendering in più.

## Architettura

```
Orchestrator.run()  --on_event(dict)-->  queue  --> app.py generator --> gr.HTML (timeline)
                                                 \-> logs/events_*.jsonl (replay futuro)
LMM.generate([system, question])  ----------------> gr.HTML (colonna "Model alone")
```

## Fase 1 — Hook eventi in `src/orchestrator.py` (additivo)

- `Orchestrator.__init__` accetta `on_event: Callable[[dict], None] | None = None`
  (default `None`: test esistenti, `main.py` ed `evaluate_minimal.py` non cambiano).
- Helper `_emit(type, **data)`: no-op senza sink. Chiamato **accanto** ai `_log`, mai al
  posto di un branch.

| type | dove | payload |
|---|---|---|
| `question` | inizio `run` | `text` |
| `decision` | dopo ogni `RAW OUTPUT` di step | `step`, `raw` |
| `search` | dopo `EXTRACTED TRIGGER` | `step`, `query` |
| `retrieval` | dopo l'inserimento dei `new_hits` | `step`, `hits=[{label,id,score,image_path,caption}]` |
| `duplicate` | dopo `DUPLICATE HITS` | `step`, `labels`, `new_evidence` |
| `ready` | dopo `STEP n - READY` | `step` |
| `invalid_decision` / `decision_failed` | rami di retry | `step`, `raw` |
| `limit` | image limit / step limit | `step`, `reason` |
| `final_prompt` | prima della generazione finale | `labels` |
| `invalid_answer` | dopo `FINAL ANSWER - INVALID` | `text`, `labels` |
| `answer` | ogni `return` di `_generate_final` | `text`, `valid`, `corrected`, `no_evidence` |

Il futuro evento `thinking` (`step`, `text`) si aggiungerà qui, prima di `decision`.

## Fase 2 — Prompt del baseline in `src/prompts.py`

`BASELINE_SYSTEM_PROMPT` e `BASELINE_ANSWER_INSTRUCTION`: la condizione "no-RAG",
riutilizzabile per la valutazione (Direzione C di `docs/DIRECTIONS.md`).

## Fase 3 — `app.py` (Gradio, radice del repo)

Dipendenza `gradio` pinnata in `requirements-common.txt`.

1. Startup: stessi `os.environ.setdefault` di `main.py`, `LMM` e `Retriever` caricati una volta.
2. `run_baseline(question)`: `lmm.generate([system, user])`, nessun retrieval.
3. `run_agentic(question)` (generator): `Orchestrator(..., on_event=queue.put)` in un thread,
   consumo della coda, `yield` dell'HTML a ogni evento; eventi salvati in
   `logs/events_<timestamp>.jsonl`.
4. `render_event(event)`: un ramo per `type`; tipo sconosciuto → card grigia generica (così un
   futuro `thinking` compare comunque). Citazioni evidenziate riusando
   `PARENTHETICAL_IMAGE_LABEL_PATTERN`. Thumbnail base64 via Pillow con cache.
5. Layout `gr.Blocks`: domanda + "Run both" + 4 preset (importati da
   `scripts/evaluate_minimal.py`); colonna "Model alone" (~35 %) e colonna "Agentic loop"
   (~65 %) con stats nel header; footer "reasoning trace: off". Prima il baseline, poi lo
   streaming della timeline.

## Fase 4 — Verifica

1. `python -m unittest tests.test_orchestrator` verde, incluso il test sugli eventi.
2. `HF_HUB_OFFLINE=1 python scripts/evaluate_minimal.py` invariato rispetto al run 19:58.
3. `python app.py` → 4 preset: sinistra riempita prima, eventi uno alla volta, thumbnail,
   guard ambra su Colosseo/Sydney, citazioni evidenziate, stats coerenti, JSONL scritto.
4. Fallback se MLX non tollera il thread: run sincrono con eventi raccolti in lista.

## Fase 5 — Sincronizzazione docs

README (Running, Requirements), ARCHITECTURE (§2.3 hook, §3.4 riga `app.py`),
`project-context.mdc` (riga su `app.py`), stato dei todo di questo piano.

## Fuori scope

Replay da log nella UI, evento `thinking`, pannello raw prompts, modalità single-pass RAG,
modifiche a dataset/retriever/prompt decisionali.

## Risultato (2026-09-03, MLX sul Mac)

- `tests/test_orchestrator.py`: 8 test verdi (7 esistenti + sequenza eventi).
- `scripts/evaluate_minimal.py` con l'hook attivo: log identici al run del 2026-09-02
  19:58 in tutti e quattro i casi (decisioni, hit, duplicati, READY, risposte finali).
- `app.py` verificato nel browser sui casi Sydney e Colosseo–Sydney: colonna "Model alone"
  riempita prima, timeline in streaming un evento alla volta (thread + queue funziona con
  MLX, `--no-thread` non è servito), thumbnail e score CLIP, guard ambra sui duplicati,
  READY verde, card del validatore, risposta con badge rosso quando le citazioni mancano
  (mostrata così com'è, nessuna citazione aggiunta). Loop completo in 5–9 s.
- Eventi salvati in `logs/events_*.jsonl` (run_start, baseline_answer, eventi del loop,
  run_end).
- Tema chiaro forzato via `js` al launch: senza, i testi senza colore esplicito ereditavano
  il tema scuro del browser ed erano invisibili sulle card bianche.
- La risposta finale su Sydney è uscita senza citazioni sia nel CLI sia nella UI: la UI
  non introduce differenze, riflette il limite già noto del modello.
