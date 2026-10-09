# Implementazione: revisione delle etichette sospette in `bee_al`

Lavori nel repository `bee_segmentation`, branch `claude/beautiful-lamport-vp4z28`. Il pacchetto `bee_al/` è l'hub di active learning sui crop di singole api (DINOv2 congelato → memory bank → k-NN pesato + label propagation → batch → etichettatura nell'hub → cluster, poi snapshot, audit e stime). Leggi prima `bee_al/README.md`, `bee_al/engine.py`, `bee_al/knn.py`, `bee_al/memory_bank.py`, `bee_al/label_store.py`, `bee_al/hub.py`, `bee_al/config.py`, `config/al_config.yaml`, `tests/conftest.py`, `tests/test_engine.py`, `tests/test_hub.py`.

## Obiettivo

Le etichette date a mano possono essere sbagliate (errori di clic, stanchezza, casi ambigui, criteri che cambiano nel tempo). Nell'hub un'etichetta sbagliata pesa molto: le etichette sono poche, le query per incertezza scelgono proprio le api ambigue, e nella propagazione le etichette umane sono bloccate (`F[labeled] = Y0`), quindi un errore si diffonde alle api simili senza mai essere corretto dal grafo.

Va aggiunta una **coda di revisione delle etichette sospette**: le api etichettate a mano su cui i vicini etichettati più simili indicano con forza un'altra classe (idea del *confident learning*, Northcutt et al. 2021). Le sospette vengono **mostrate a un umano**, che conferma o corregge. Niente viene corretto in automatico.

---

## Vincoli

- **Non modificare** strategia di query, k-NN, propagazione, re-train, snapshot, audit, stime, né il formato delle colonne di `al_labels.csv`.
- Solo numpy/scipy/sklearn/stdlib: niente torch nell'hub.
- Stile del codice esistente: commenti e docstring in italiano, testi dell'interfaccia in inglese, type hints, dataclass di config con `_check`, scritture atomiche dove già si usano, tutto sotto `self.lock` nell'engine.
- Le config esistenti devono caricarsi senza modifiche (default sensati per la nuova sezione). Aggiorna `config/al_config.yaml` con la sezione commentata.
- Stessi input → stessi output (nessuna casualità non controllata).

---

## 1. Quali etichette si controllano

**Solo le etichette di training**: le api con etichetta umana corrente nella memory bank (`bank.labeled_indices()`), comprese quelle di classi di scarto.

Sono **escluse**, e il motivo va scritto nel docstring:

- **Validation set e audit.** Servono a misurare il modello. Correggerle guardando l'opinione del modello le avvicinerebbe alle sue predizioni e farebbe sembrare il modello migliore di quanto è. Per loro l'unico controllo ammesso è l'accordo tra annotatori, già presente nell'audit.
- Le api messe da parte con "Scarta ape" (`UNKNOWN_LABEL`).

## 2. Come si calcola il sospetto

Nuovo modulo `bee_al/review.py` con:

```python
@dataclass
class Suspect:
    image_id: str
    index: int
    label: str                 # etichetta umana corrente
    suggested: str             # classe più votata dai vicini, diversa da label
    p_suggested: float         # probabilità k-NN della classe suggerita
    p_label: float             # probabilità k-NN dell'etichetta umana
    annotator: str
    confidence: str            # confidenza dichiarata dall'annotatore (High/Medium/Low/"")
    query_type: str
    timestamp: str
    neighbors: list[dict]      # i k vicini etichettati: image_id, label, similarità
    rare_class: bool           # label ha meno di review.min_class_size etichette umane
```

`find_suspects(bank, store, cfg, exclude_ids) -> list[Suspect]`:

1. Per ogni ape etichettata calcola il k-NN **escludendo l'ape stessa** (`knn_predict` lo fa già: un'ape etichettata non vota per sé). Usa `review.k` (default = `knn.k`) e la stessa temperatura `knn.temperature`.
2. È sospetta se la classe più votata è diversa dalla sua etichetta **e** `p_suggested >= review.min_disagreement` (default 0.8).
3. **Classi rare.** Se l'etichetta appartiene a una classe con meno di `review.min_class_size` (default 5) etichette umane, i vicini saranno quasi sempre di altre classi, anche se l'etichetta è giusta: è proprio il caso dei fenotipi rari appena scoperti. Queste api vanno marcate `rare_class = True` e mostrate in una sezione separata, con una nota che spiega perché il segnale è debole. Non devono essere mescolate con le altre.
4. **Ordine** della coda principale: prima le etichette con confidenza `Low`, poi `Medium`, poi le altre; a parità, `p_suggested` decrescente; poi `image_id` per stabilità.
5. Per ogni sospetta salva i `review.show_neighbors` (default 6) vicini etichettati più simili, con etichetta e similarità: servono come evidenza nella pagina.

Calcolo efficiente: un'unica ricerca dei vicini su tutte le etichettate (`bank.search_labeled`), niente cicli Python su tutta la bank. La coda viene ricalcolata a ogni `recompute()`/`refresh()` e messa in cache nell'engine, non a ogni richiesta HTTP.

## 3. Decisioni di revisione

Le decisioni vanno in un file separato, `results_dir/review_log.jsonl`, in sola aggiunta. Una riga per decisione:

```json
{"timestamp": "...", "annotator": "...", "image_id": "...", "decision": "confirmed|corrected|set_aside",
 "old_label": "...", "new_label": "...", "suggested": "...", "p_suggested": 0.91,
 "annotator_confidence": "Low", "model_version": "v000", "round": 42, "rare_class": false}
```

- **Confirm** ("l'etichetta è giusta"): nessun nuovo evento in `al_labels.csv`; solo la riga nel log. Un'ape confermata **non torna nella coda** finché restano uguali sia la sua etichetta sia la classe suggerita. Se una delle due cambia (nuova etichetta, nuovo re-train che sposta i vicini), può ricomparire.
- **Correct** ("la classe giusta è X"): passa da `engine.assign_many(annotator, [image_id], X)` (o dal percorso equivalente già usato per le correzioni), così l'evento finisce in `al_labels.csv` come una normale correzione e le predizioni si ricalcolano. In più la riga nel log con `decision = "corrected"`. X può essere la classe suggerita o qualsiasi altra.
- **Set aside**: l'ape diventa "Scarta ape" (`UNKNOWN_LABEL`) tramite lo stesso percorso; riga nel log con `decision = "set_aside"`.
- Al riavvio dell'hub lo stato delle conferme si ricostruisce dal log.
- La revisione **non deve** toccare validation set e audit (vedi punto 1): se arriva una richiesta per una di quelle api, `ALError` esplicito.

## 4. Config

```python
@dataclass
class ReviewConfig:
    enabled: bool = True
    k: int = 0                    # 0 = usa knn.k
    min_disagreement: float = 0.8 # probabilità minima della classe suggerita
    min_class_size: int = 5       # sotto questa soglia la classe è "rara": sezione a parte
    show_neighbors: int = 6
```

con i `_check` appropriati (0 < min_disagreement ≤ 1, min_class_size ≥ 1, k ≥ 0, show_neighbors ≥ 1).

## 5. Engine

- `ALEngine.suspects() -> dict` con `{"main": [...], "rare": [...], "n_checked": int, "n_confirmed_hidden": int}` dalla cache.
- `ALEngine.review(annotator, image_id, decision, new_label=None) -> None` che applica il punto 3.
- `ALEngine.review_stats() -> dict`: etichette controllate, sospette attuali, decisioni per tipo, correzioni per coppia (vecchia → nuova classe), quota di sospette tra le etichette `Low`/`Medium`/`High`. Sono i numeri per la tesi.
- Aggiungi a `status()` il numero di sospette in coda e `review_stats()`.
- Aggiungi a ogni riga di `al_rounds.jsonl` il campo `n_suspects` (letto con `.get()` nelle righe vecchie).

## 6. Hub

- Nuova voce **Review** nel menu, tra Clusters e Status, con il numero di sospette tra parentesi.
- Pagina `/al-review`, una sospetta per volta (come la pagina di etichettatura):
  - l'immagine dell'ape;
  - etichetta attuale, annotatore, data, confidenza dichiarata;
  - la classe suggerita con la sua probabilità e quella dell'etichetta attuale;
  - una riga con le miniature dei vicini etichettati più simili, ognuna con la sua classe e la similarità: è l'evidenza su cui decidere;
  - pulsanti **Keep "<etichetta attuale>"**, **Change to "<classe suggerita>"**, un campo con i pulsanti rapidi delle classi per scegliere un'altra classe, e **Scarta ape**;
  - avanzamento ("12 suspects left, 30 reviewed").
- Sotto la coda principale, una sezione **Rare classes** separata con la spiegazione del segnale debole.
- Pagina di stato: riquadro "Label review" con `review_stats()` e link al download di `review_log.jsonl` e di un `review_report.csv` (una riga per decisione).
- Le decisioni avvengono con `POST /al-review`; errori (`ALError`) mostrati come messaggio, mai un 500.

## 7. Test (`tests/test_review.py` + aggiunte in `tests/test_hub.py`)

Su dati sintetici (fixture `synth` e `cfg` di `conftest.py`):

- Etichette corrette di cluster ben separati → nessuna sospetta.
- Si inverte a mano l'etichetta di alcune api (rumore noto) → quelle api compaiono tra le sospette, con la classe suggerita uguale a quella vera; recall alta sul rumore iniettato, pochi falsi allarmi.
- L'ape non vota per se stessa (con k = 1 e un'unica ape di una classe isolata, la sospetta compare comunque correttamente).
- Ordinamento: a parità di disaccordo una `Low` viene prima di una `High`.
- Classe con meno di `min_class_size` etichette → `rare_class = True`, sezione separata.
- Validation set, audit e "Scarta ape" non compaiono mai; `review()` su quelle api solleva `ALError`.
- **Confirm**: l'ape sparisce dalla coda, nessun evento nuovo in `al_labels.csv`, una riga in `review_log.jsonl`; ricompare se cambia la classe suggerita.
- **Correct**: evento nuovo in `al_labels.csv`, etichetta aggiornata nella bank, predizioni ricalcolate, riga nel log.
- **Set aside**: l'ape diventa `UNKNOWN_LABEL`.
- Riavvio dell'engine: conferme ricostruite dal log.
- `review_stats()` coerente con le decisioni prese.
- Hub: `/al-review` risponde 200 con e senza sospette, le tre azioni funzionano, la voce di menu mostra il conteggio.
- Tutti i test esistenti restano verdi.

## 8. Documentazione

- Aggiungi a `bee_al/README.md` una sezione **"Revisione delle etichette sospette"**:
  - come vengono trovate;
  - perché validation set e audit sono esclusi;
  - cosa significano confirm, correct e set aside;
  - come leggere la sezione delle classi rare;
  - quali numeri riportare (sospette trovate, corrette, quota per confidenza dichiarata).
- Una frase nella sezione sull'etichettatura che consigli di usare la confidenza *Low* per le api dubbie, perché finiscono per prime in revisione.

## Commit

Un solo commit: `bee_al: review queue for suspicious hand labels (kNN disagreement)`, con `pytest` verde. Se trovi un'ambiguità che cambia il comportamento, fermati e chiedi invece di indovinare.
