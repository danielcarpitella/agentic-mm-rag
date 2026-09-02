---
name: stabilizzare loop agentico
overview: Correggere duplicati, terminazione e citazioni preservando l’invariante di una sola retrieval per turno. Il codice e i test deterministici sono stati aggiornati; il run MLX finale ha confermato loop e dedup, ma non il grounding della risposta finale.
todos:
  - id: dedup-state-machine
    content: Implementare dedup per hit.id e feedback decisionale senza consumare label o budget
    status: completed
  - id: strengthen-citations
    content: Rendere prompt finale e correzione operativi, label-safe e focalizzati sull’evidenza visibile
    status: completed
  - id: cap-decisions
    content: Aggiungere un limite token dedicato alle generazioni decisionali su entrambi i backend
    status: completed
  - id: add-regression-tests
    content: Aggiungere test deterministici della state machine, dedup e retry limitato
    status: completed
  - id: verify-minimal-eval
    content: Verificare localmente e poi confrontare il nuovo run MLX 4-casi con il baseline delle 15:28
    status: completed
isProject: true
---

# Stabilizzare il loop agentico

## Intenzione e scopo

Portare l’attuale loop da prototipo parzialmente funzionante a comportamento
affidabile sui quattro casi minimi, intervenendo sui problemi osservati di
duplicazione, terminazione, citazioni ed evidenza visiva. Lo scopo non è
ridisegnare il sistema o ottimizzarlo per il solo dataset attuale, ma consolidare
il protocollo agentico preservando una retrieval per turno e una nuova decisione
del modello dopo ogni risultato.

## Fonte unica e baseline pre-implementazione

Questo piano è autosufficiente: non presuppone che il testo di Claude venga copiato
nel repository. Prima di questa implementazione, il codice aveva il protocollo
`SEARCH`/`READY`, una retrieval per turno e un solo retry delle citazioni, ma non
aveva dedup (evitare che `SEARCH` reinserisca un’immagine già presente), few-shot
di citazione, feedback sui duplicati o un limite token specifico per le decisioni.

Il riferimento è il doppio run deterministico delle 15:28 definito in
[`scripts/evaluate_minimal.py`](../../scripts/evaluate_minimal.py):

- Test 1, Sydney: recupera l’ID corretto, ripete la stessa ricerca e inserisce la
  stessa immagine come `Image 2`; poi emette `READY`. Risposta senza citazione e
  retry identico.
- Test 2, Stonehenge: una ricerca e `READY` immediato, quindi il loop è quasi
  ideale. Risposta senza citazione e retry identico, con misure e dettagli
  “mortise and tenon” quasi copiati dalla caption.
- Test 3, Colosseo–Sydney: recupera correttamente i due ID nei primi due step,
  poi li ripete fino a `IMAGE LIMIT`. È l’unico caso con citazioni valide,
  `Image 1` e `Image 2`.
- Test 4, Liberty–Cristo: recupera correttamente i due ID, poi li ripete fino a
  `IMAGE LIMIT`. Risposta senza citazioni, retry identico e confronto visivo
  generico/inaffidabile, incluso “pointed top” per entrambe le statue.

Baseline complessiva: primi retrieval corretti 4/4, duplicati inseriti in 3/4,
citazioni valide 1/4, echo del system prompt allo step iniziale in 3/4. I casi
multi terminano per limite immagini, non con `READY`.

## Valutazione critica del feedback esterno

- **Dedup:** diagnosi e correzione sono valide. La guardia deve operare
  sull’identità del risultato (`hit.id`), non solo sulla query testuale: query
  diverse possono restituire la stessa immagine.
- **Terminazione:** dopo la dedup i duplicati non consumano più il budget
  immagini. I casi insistenti arrivano quindi a `READY` oppure a `max_steps`.
- **Citazioni:** nei tre casi inizialmente falliti, output originale e corretto
  erano identici. Con temperatura zero, il retry era di fatto inutile.
- **Few-shot:** un esempio con `Image 1` e `Image 2` nei casi single rischia di
  introdurre label inesistenti; l’esempio deve usare soltanto label disponibili.
- **Caption parroting:** è un rischio reale davanti a una giuria tecnica.
  Stonehenge e il test 4 mostrano recitazione dei metadata; Sydney è il
  contro-esempio positivo perché la forma descritta non appare nella caption.
- **Echo del prompt:** non rompe il retrieval perché viene estratto il primo
  `SEARCH`, ma spreca token e mostra aderenza debole al protocollo.
- **Token cap:** limitare realmente la generazione decisionale è preferibile a
  troncare la stringa dopo la generazione. Uno stop su newline resta fuori scope.
- Il warning Transformers `Kwargs passed to processor...` è cosmetico.

## Implementazione

### 1. Dedup e terminazione sicura

- In [`src/orchestrator.py`](../../src/orchestrator.py), mantenere una mappa
  `hit.id -> Image N`, filtrare gli hit già visti e assegnare label contigue
  soltanto agli hit nuovi.
- Se una ricerca restituisce solo duplicati, conservare nella history il
  `SEARCH("...")` tentato, ma non reinserire immagine/caption e non incrementare
  `images_used`.
- In [`src/prompts.py`](../../src/prompts.py), comunicare che l’evidenza è già
  disponibile come `Image N` e chiedere `READY` o una ricerca realmente diversa.
- Tornare sempre al modello per una nuova decisione. Il tentativo duplicato
  consuma comunque uno step, preservando una sola azione per turno ed evitando
  loop infiniti.
- Accettare `READY.` quando è la decisione effettiva, mantenendo la priorità del
  primo `SEARCH` presente nell’output.

### 2. Citazioni e risposta visiva

- Rafforzare `FINAL_ANSWER_INSTRUCTION` con risposta concisa, citazione vicina al
  claim visivo ed esempio costruito sulle label esistenti.
- Rafforzare `CORRECT_ANSWER_INSTRUCTION` e mantenere un solo retry.
- Loggare distintamente una correzione identica o ancora invalida.
- Specificare che le caption servono come contesto/identificazione e non come
  fonte di misure, autori, storia o altri fatti non osservabili.
- Non cambiare le quattro domande durante il confronto col baseline. Per la demo
  finale, privilegiare domande la cui risposta non appare nella caption.

### 3. Output decisionali brevi

- In [`src/lmm.py`](../../src/lmm.py), aggiungere a `generate()` un override
  opzionale del limite token valido per MLX e CUDA.
- Esporre `decision_max_new_tokens: 48` in
  [`src/config.py`](../../src/config.py) e [`config.yaml`](../../config.yaml).
  Usarlo solo nei turni decisionali; risposta finale e correzione mantengono 512.

### 4. Regressioni deterministiche

- Aggiungere un test `unittest` con LMM e retriever finti per dedup con query
  diverse, label contigue, budget invariato, nuova decisione, terminazione a
  `max_steps`, `READY.` e retry singolo.
- Eseguire test, controllo sintattico e lint prima del run MLX.

## Risultato verificato alle 17:57

- Retrieval attesi 4/4 e nessuna immagine duplicata reinserita.
- Sydney e Colosseo–Sydney emettono `READY` dopo il feedback dedup;
  Liberty–Cristo termina in modo limitato a `max_steps`.
- Il cap limita l’echo ma non lo elimina.
- Sydney produce l’unica risposta realmente valida.
- Stonehenge copia il contenuto dell’esempio few-shot e passa erroneamente il
  controllo pur rispondendo sul tetto.
- I due casi multi non citano le immagini; il retry resta identico.
- Caption parroting e contaminazione tra immagini restano aperti.

Conclusione: la state machine di retrieval è stabilizzata; la generazione finale
resta parziale. Il lavoro continua nel piano
[`grounding-risposta-finale.plan.md`](./grounding-risposta-finale.plan.md).

## Fuori scope

- Nessun cambio a dataset, indice, retriever, `top_k`, modello o temperatura.
- Nessuna coda di `SEARCH`, esecuzione automatica di più retrieval o retry
  illimitato.
- Nessuna dichiarazione di “problema risolto” quando la valutazione resta
  parziale.
