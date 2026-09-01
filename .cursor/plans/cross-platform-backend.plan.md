---
name: cross-platform-backend
overview: Aggiungere un backend Transformers/CUDA per i due PC Windows, mantenendo MLX come percorso Mac già verificato. La modifica resta confinata al wrapper LMM e alle configurazioni/installazioni per piattaforma.
todos:
  - id: split-dependencies
    content: Separare dipendenze e istruzioni Mac/Windows CUDA
    status: completed
  - id: implement-transformers
    content: Aggiungere il backend Transformers CUDA nel wrapper LMM
    status: completed
  - id: add-platform-config
    content: Aggiungere configurazioni e selezione CLI per piattaforma
    status: completed
  - id: validate-windows
    content: Testare prima RTX 4060, poi RTX 5070 Ti, fino al loop completo
    status: pending
  - id: sync-docs
    content: Aggiornare README, architettura e contesto dopo i risultati verificati
    status: pending
isProject: true
---

# Piano: backend cross-platform

## Decisione
- Mantenere **due backend**: `mlx` sul Mac Apple Silicon e `transformers` sui PC Windows con NVIDIA CUDA.
- Non creare backend distinti per RTX 4060 e RTX 5070 Ti: condividono lo stesso percorso Transformers/CUDA.
- Tenere invariata l'interfaccia `LMM.generate(messages)`; orchestratore, retriever e prompt restano agnostici al backend.

## Implementazione
1. Separare le dipendenze comuni da quelle di piattaforma, così Windows non tenta di installare `mlx-vlm`. Fornire istruzioni Windows con PyTorch CUDA e la quantizzazione 4-bit richiesta dalla RTX 4060 da 8 GB.
2. Estendere [`src/lmm.py`](../../src/lmm.py) con il ramo `transformers`: caricare `Qwen/Qwen2-VL-2B-Instruct`, usare CUDA, trasformare gli stessi `Message` (ruoli, testo e path delle immagini) nel formato Qwen e restituire solo il testo generato. Conservare il ramo MLX senza modificarne il comportamento.
3. Aggiungere configurazioni per piattaforma e un'opzione CLI per sceglierle, mantenendo la configurazione Mac come default. La configurazione Windows deve usare il checkpoint Hugging Face nativo e limitare in modo prudente le immagini/VRAM per la RTX 4060; nessun percorso CPU viene considerato supportato per la demo.
4. Verificare prima il backend LMM con [`scripts/test_lmm.py`](../../scripts/test_lmm.py), poi il retriever e infine il ciclo completo con [`main.py`](../../main.py). Testare inizialmente sulla RTX 4060, poi ripetere sulla RTX 5070 Ti.
5. Aggiornare [`README.md`](../../README.md), [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) e questo piano con i risultati verificati.

## Criterio di completamento
- Mac: il comando esistente continua a funzionare con MLX.
- Ciascun PC Windows: installazione pulita, test LMM, retrieval e una domanda landmarks completano `SEARCH → retrieval → risposta` senza errori di VRAM.
- Il README consente a un compagno di ripetere la procedura senza modifiche manuali al codice.

## Stato di implementazione
- Completati: dipendenze separate, backend Transformers/CUDA, configurazione Windows e selezione `--config`.
- Verificato localmente: sintassi Python e opzioni CLI.
- Da verificare: esecuzione MLX fuori dal sandbox e test completi sui PC Windows.
