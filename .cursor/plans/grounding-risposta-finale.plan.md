---
name: grounding risposta finale
overview: "Continuazione del piano “Stabilizzare il loop agentico”: la state machine di retrieval resta congelata, mentre questa fase ridisegna la generazione finale per ottenere citazioni complete e risposte basate sulle immagini anziché sulle caption."
todos:
  - id: checkpoint-baseline
    content: Registrare il run 17:57 in NOTES.md e committare lo stato corrente come checkpoint
    status: completed
  - id: isolate-final-context
    content: Costruire la risposta finale da immagini e label in un contesto multimodale pulito
    status: pending
  - id: strengthen-final-contract
    content: Rimuovere esempi copiabili e imporre osservazioni visive con citazioni complete
    status: pending
  - id: replace-correction-pass
    content: Usare una sola rigenerazione pulita senza risposta precedente o caption
    status: pending
  - id: test-final-grounding
    content: Aggiornare ed estendere i test e rieseguire la valutazione MLX sui quattro casi
    status: pending
  - id: document-observed-result
    content: Aggiornare note e stato del piano solo con il risultato verificato
    status: pending
isProject: true
---

# Grounding della risposta finale

## Collegamento al piano precedente

Questo piano continua
[`stabilizzare-loop-agentico.plan.md`](./stabilizzare-loop-agentico.plan.md).
Il piano precedente ha stabilizzato retrieval sequenziale, dedup per `hit.id`,
label/budget, terminazione limitata, `READY.` e cap decisionale. Queste parti sono
un checkpoint e non vanno riprogettate in questa iterazione.

Il run MLX delle 17:57 costituisce il nuovo baseline (log `run_20260902_1757*.log`):

- retrieval attesi 4/4 e nessuna immagine duplicata reinserita;
- Sydney e Colosseo–Sydney emettono `READY` dopo il feedback dedup;
  Liberty–Cristo termina correttamente a `max_steps`;
- soltanto Sydney produce una risposta realmente valida;
- Stonehenge copia il few-shot del tetto e passa erroneamente la validazione;
- i due casi multi non citano le immagini e il retry resta identico;
- caption parroting e contaminazione tra immagini restano forti, soprattutto in
  Stonehenge e Liberty–Cristo.

## Intenzione e confine

Rendere affidabile la fase successiva a `READY` senza modificare dataset,
retriever o protocollo di una retrieval per turno. Il problema da risolvere non è
più trovare le immagini, ma sintetizzare una risposta visiva con attribuzione
corretta delle label.

Diagnosi che motiva l'approccio: oggi la risposta finale eredita l'intera
conversazione decisionale (system prompt del protocollo, cronologia
`SEARCH/READY`, caption complete, esempio few-shot e, nella correzione, la
risposta sbagliata). Per un 2B è più facile copiare quel testo che guardare le
immagini; il contesto pulito rimuove la scorciatoia invece di aggiungere altre
istruzioni sopra.

## Implementazione

### 1. Contesto finale pulito

- In [`src/orchestrator.py`](../../src/orchestrator.py), conservare gli hit unici
  insieme alla label assegnata (lista ordinata `(label, hit)`), non soltanto
  `images_used`.
- Al momento della risposta, costruire una nuova conversazione multimodale
  contenente esclusivamente: un system prompt minimo dedicato alla risposta
  (rispondere solo da ciò che è visibile, citare le label), la domanda, le
  immagini e le associazioni minimali di identificazione. I ruoli reali sono
  necessari: è documentato che il 2B ignora le istruzioni concatenate senza ruolo.
- Allegare tutte le immagini a un solo messaggio user, nell'ordine delle label
  (entrambi i backend in [`src/lmm.py`](../../src/lmm.py) preservano l'ordine dei
  messaggi).
- Per l'identificazione usare un nome leggibile derivato da `hit.id`
  (underscore → spazi, es. `Image 1 — sydney opera house`): serve solo a dire
  quale landmark è quale, non come fonte di fatti, ed evita che il modello ricopi
  l'id grezzo nella risposta della demo.
- Escludere dalla generazione finale cronologia `SEARCH/READY`, system prompt
  decisionale e caption descrittive. Le caption restano disponibili nel retrieval
  loop, ma non diventano una scorciatoia testuale per la risposta.
- Loggare l'intero contesto finale inviato (system, testi, elenco immagini), non
  soltanto l'istruzione: senza il prompt letterale il debugging della nuova fase
  è impossibile.
- Caso degenere `images_used == 0` (limite step senza alcuna immagine recuperata):
  mantenere il comportamento attuale, nessuna generazione a contesto pulito.

### 2. Prompt strutturale senza contenuto copiabile

- In [`src/prompts.py`](../../src/prompts.py), sostituire il few-shot concreto
  con un contratto di formato privo di esempi semantici.
- Il contratto elenca sempre le label concrete disponibili (`Image 1`,
  `Image 2`, …), mai il placeholder generico `Image N`: è documentato in NOTES.md
  che il modello copia i placeholder alla lettera.
- Richiedere una breve osservazione visibile per ogni immagine pertinente, con la
  relativa `(Image N)` nella stessa frase, seguita dal confronto quando la domanda
  è multi-image.
- Vietare esplicitamente date, autori, misure, ubicazioni, funzione e altri fatti
  non direttamente visibili.

### 3. Validazione completa delle label

- Estendere la validazione in [`src/orchestrator.py`](../../src/orchestrator.py):
  la forma richiesta è parentetica `(Image N)`, le label inesistenti vengono
  rifiutate e devono comparire **tutte** le label da `Image 1` a `Image N`
  presenti nel contesto. La regola è generale, non hardcodata sul benchmark: dopo
  la dedup il contesto contiene solo hit realmente richiesti dal modello.
- Nei casi multi del benchmark questo implica che una risposta è valida soltanto
  se contiene sia `(Image 1)` sia `(Image 2)`; la sola presenza di una label non
  basta.
- Limite accettato: se il retriever restituisse un'immagine non pertinente, anche
  quella andrebbe citata. Si documenta il limite invece di introdurre euristiche
  semantiche.
- Aggiungere una protezione deterministica contro risposte copiate o degenerate:
  rifiutare il placeholder letterale `Image N`, risposte costituite quasi solo da
  citazioni o troppo corte per contenere un'osservazione, e riproduzioni verbatim
  di frammenti lunghi dell'istruzione. Evita un altro falso positivo come
  Stonehenge.

### 4. Rigenerazione correttiva pulita

- Sostituire l'attuale correzione che appende la risposta errata alla cronologia
  con una sola rigenerazione dalla stessa domanda e dalle stesse immagini.
- Usare un prompt più vincolante ma ancora privo di esempi concreti; non
  reinserire né la vecchia risposta né le caption. A temperatura zero la
  rigenerazione produce un output diverso solo se il contesto cambia: sostituire
  l'istruzione è il meccanismo, mentre il vecchio retry appendeva testo a un
  contesto quasi identico e riproduceva la stessa risposta carattere per
  carattere.
- Se anche la rigenerazione fallisce, loggare il fallimento e fermarsi: niente
  retry aggiuntivi o citazioni aggiunte deterministicamente a contenuto non
  verificato.

### 5. Test e valutazione

- Aggiornare ed estendere
  [`tests/test_orchestrator.py`](../../tests/test_orchestrator.py). Due asserzioni
  esistenti dipendono dal vecchio comportamento e vanno riviste: la chiamata
  finale non conterrà più il feedback dedup ("already available as Image 1") né
  la vecchia risposta appesa durante la correzione.
- Nuovi test: isolamento del contesto finale (nessuna caption, nessuna
  cronologia, immagini in ordine), insieme completo delle citazioni,
  rigenerazione singola a contesto pulito e rifiuto di placeholder o risposte
  degenerate.
- Mantenere invariati i quattro casi in
  [`scripts/evaluate_minimal.py`](../../scripts/evaluate_minimal.py) per
  confrontare direttamente il nuovo run con quello delle 17:57.
- Controllare automaticamente retrieval e label; valutare manualmente pertinenza
  visiva e caption parroting.

## Criteri di accettazione

- Retrieval, dedup e terminazione non regrediscono.
- 4/4 risposte contengono tutte e sole le citazioni richieste; i due confronti
  citano entrambe le immagini.
- Nessuna risposta copia un esempio o una struttura placeholder.
- Le risposte descrivono forma, colore, disposizione o posa visibili, senza
  recitare metadata.
- Se il 2B fallisce ancora con il contesto pulito, il limite viene documentato
  invece di mascherarlo con post-processing semantico non affidabile.

## Checkpoint e documentazione

Prima dell'implementazione, due azioni concrete: aggiungere a
[`docs/NOTES.md`](../../docs/NOTES.md) la voce con i risultati osservati del run
17:57 e committare lo stato corrente del repository — le modifiche del piano
precedente (`config.yaml`, `src/`, `tests/`, `docs/`) sono ancora non committate,
e quel commit è il checkpoint a cui poter tornare.

Dopo il nuovo run, aggiornare NOTES.md soltanto con risultati osservati,
mantenendo il promemoria per la demo: privilegiare domande la cui risposta non è
già nella caption, con Sydney roof come caso positivo.

Stima realistica: 2–3 ore per implementazione e test, più 1–2 ore per run MLX e
un'eventuale iterazione mirata.
