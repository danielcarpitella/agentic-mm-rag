---
name: grounding risposta finale
overview: "Continuazione del piano “Stabilizzare il loop agentico”: la fase finale pulita è stata implementata e verificata. Il grounding visivo migliora, ma Qwen2-VL-2B continua a ignorare le citazioni; il limite è documentato senza post-processing ingannevole."
todos:
  - id: checkpoint-baseline
    content: Registrare il run 17:57 in NOTES.md e committare lo stato corrente come checkpoint
    status: completed
  - id: isolate-final-context
    content: Costruire la risposta finale da immagini e label in un contesto multimodale pulito
    status: completed
  - id: strengthen-final-contract
    content: Rimuovere esempi copiabili e imporre osservazioni visive con citazioni complete
    status: completed
  - id: replace-correction-pass
    content: Usare una sola rigenerazione pulita senza risposta precedente o caption
    status: completed
  - id: test-final-grounding
    content: Aggiornare ed estendere i test e rieseguire la valutazione MLX sui quattro casi
    status: completed
  - id: document-observed-result
    content: Aggiornare note e stato del piano solo con il risultato verificato
    status: completed
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

- Durante il retrieval, in [`src/orchestrator.py`](../../src/orchestrator.py),
  conservare una lista ordinata delle immagini uniche recuperate. Ogni elemento
  contiene sia la label assegnata sia l'intero `Hit`, per esempio
  `[(1, hit_colosseum), (2, hit_sydney)]`. Il solo contatore `images_used` dice
  quante immagini esistono, ma non permette di ricostruire quale file e quale
  landmark corrispondono a ciascuna label.
- Quando il loop arriva alla risposta finale, **non riutilizzare** la lista
  `messages` usata per `SEARCH/READY`. Creare invece una nuova conversazione
  `answer_messages`, indipendente dalla precedente.
- La nuova conversazione contiene soltanto due messaggi:
  1. un messaggio `system` breve, dedicato alla risposta, che ordina di descrivere
     esclusivamente elementi visibili e di citare le immagini;
  2. un messaggio `user` con la domanda originale, l'elenco delle associazioni
     tra label e landmark e tutte le immagini.
- Allegare le immagini al messaggio `user` nello stesso ordine delle label. Per
  esempio, nel confronto Colosseo–Sydney, il testo indica
  `Image 1 — colosseum` e `Image 2 — sydney opera house`, mentre l'elenco dei
  file contiene prima la foto del Colosseo e poi quella di Sydney. In questo modo
  il modello può associare senza ambiguità ogni citazione alla relativa immagine.
- Nell'elenco usare un nome leggibile derivato da `hit.id` (underscore → spazi):
  serve soltanto a identificare il landmark, non fornisce la risposta e non
  espone nella demo un id tecnico come `sydney_opera_house`.
- Mantenere separati i ruoli `system` e `user`: è già documentato che il 2B tende
  a ignorare le istruzioni quando vengono concatenate alla domanda come testo
  indistinto.
- Escludere dalla generazione finale cronologia `SEARCH/READY`, system prompt
  decisionale e caption descrittive. Le caption restano disponibili nel retrieval
  loop, ma non diventano una scorciatoia testuale per la risposta.
- Loggare l'intero contesto finale inviato (system, testi, elenco immagini), non
  soltanto l'istruzione: senza il prompt letterale il debugging della nuova fase
  è impossibile.
- Caso degenere `images_used == 0` (limite step senza alcuna immagine recuperata):
  mantenere il comportamento attuale, nessuna generazione a contesto pulito.

### 2. Prompt strutturale senza contenuto copiabile

- In [`src/prompts.py`](../../src/prompts.py), eliminare frasi di esempio che
  descrivono un contenuto specifico, come «The structure has a curved roof
  (Image 1)». Il modello 2B può copiarle come risposta anche quando l'immagine
  mostra tutt'altro, come è successo con Stonehenge.
- Non lasciare però il 2B senza guida: sostituire il few-shot semantico con uno
  **scaffold dinamico e molto esplicito**, costruito sulle label realmente
  disponibili. Lo scaffold mostra i token esatti obbligatori, per esempio
  `Required citation tokens: (Image 1), (Image 2)`, e assegna una funzione a
  ciascuna frase senza fornire una possibile risposta visiva.
- Con una sola immagine, ordinare di scrivere una breve osservazione visibile e
  iniziare la frase esattamente con `(Image 1):`. Con due immagini, ordinare una
  frase che inizi con `(Image 1):`, una con `(Image 2):` e una frase di confronto
  introdotta da entrambe le citazioni. Il prefisso è preferito al suffisso perché
  i run precedenti documentati in NOTES mostrano che il 2B segue meglio un inizio
  di frase concreto. In questo modo riceve label e struttura precise, ma non può
  copiare contenuti come “curved roof” in una risposta su Stonehenge.
- Questa soluzione è un'ipotesi da verificare nel run MLX, non una garanzia: i
  run precedenti mostrano che il 2B fatica senza guida, ma che gli esempi
  semantici possono essere copiati. Il confronto con il baseline stabilirà se lo
  scaffold risolve entrambe le debolezze.
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

## Risultato verificato alle 19:58

- I sette test deterministici passano e il controllo lint non segnala errori.
  Contesto finale, immagini e label sono isolati e ordinati; la rigenerazione è
  singola e non contiene risposta precedente, caption o cronologia decisionale.
- Due run MLX completi hanno mantenuto retrieval, dedup e terminazione del
  checkpoint. Il contesto pulito ha eliminato la copia dell'esempio del tetto e
  ha migliorato soprattutto le descrizioni iniziali di Sydney e Stonehenge.
- Il criterio sulle citazioni non è stato raggiunto: sia lo scaffold con label a
  fine frase sia quello più vincolante con label come prefisso hanno prodotto
  0/4 risposte valide nella forma parentetica richiesta. Il modello omette le
  label nei casi singoli e usa talvolta `Image 1:` senza parentesi nei multi.
- La rigenerazione non corregge le citazioni e tende ad allungare o peggiorare la
  risposta. Se resta invalida viene ora loggata e scartata, mantenendo l'output
  originale invece di sostituirlo con altro contenuto non validato.
- Il grounding multi-image resta debole: compaiono dettagli non visibili e
  contaminazioni tra monumenti anche senza caption. Come previsto dai criteri,
  il limite del 2B è registrato in NOTES.md e non viene nascosto aggiungendo
  citazioni deterministicamente.

## Checkpoint e documentazione

Il checkpoint precedente è stato salvato nel commit `a72e734` e il run 17:57 è
registrato in [`docs/NOTES.md`](../../docs/NOTES.md).

Il risultato dei run 19:56 e 19:58 è stato aggiunto alle note mantenendo il
promemoria per la demo: privilegiare domande la cui risposta non è già nella
caption, con Sydney roof come caso visivamente positivo ma non ancora conforme
nelle citazioni.
