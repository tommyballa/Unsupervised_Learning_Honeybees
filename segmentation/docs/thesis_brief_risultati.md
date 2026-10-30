# Brief per la tesi: tutti i risultati (pipeline completa, batch 3-6)

Questo file è un briefing da dare a Claude per scrivere la tesi magistrale. Contiene quello che è stato fatto e misurato fino al 17 ottobre 2026. Se il codice o i dati sono cambiati dopo questa data, fanno fede il codice e i dati.

## Istruzioni per chi scrive (Claude)

- Usa **solo** fatti e numeri presenti in questo file e nei due brief collegati (`ablation_sam3.md`, `thesis_brief_yolo_finetuning.md`). Se una frase non è supportata da qui, toglila oppure segnala `[AUTORE DA CONFERMARE]`.
- Lascia visibili nel testo i marcatori `[AUTORE DA CONFERMARE]` già presenti: sono fatti che non ho potuto verificare.
- Lingua: `[AUTORE DA CONFERMARE]`. Il brief precedente sulla segmentazione chiedeva l'inglese accademico.
- Nel testo usa nomi neutri per i run e le fasi ("modello base", "1376 px", "fase 1"). I nomi delle cartelle servono solo all'autore per ritrovare i risultati.
- Differenze di circa 0,003-0,008 mAP sono sotto la variabilità tra seed: vanno descritte come "entro il rumore", non come miglioramenti.
- Distingui sempre:
  - le metriche di segmentazione, misurate contro **pseudo-etichette SAM 3** e non contro una ground truth manuale;
  - i conteggi della classificazione per batch, che sono **output del modello senza validazione** (vedi §8 e §9).

## 1. Obiettivo e pipeline

L'obiettivo è stimare indicatori di salute dell'alveare da foto di telaini: profilo demografico (giovani, adulte, fuchi) e presenza di api malate. Le foto sono state scattate in quattro sessioni (batch 3-6).

La pipeline ha sei passi:

1. **Pseudo-etichette di segmentazione** sulle foto intere con SAM 3, guidato dal testo "bee".
2. **Fine-tuning di YOLO26s-seg** su queste pseudo-etichette, con un ablation su risoluzione, augmentation, freeze e seed.
3. **Estrazione delle singole api** con il modello YOLO scelto, dopo aver normalizzato luminosità e contrasto di ogni foto.
4. **Pulizia e normalizzazione dei crop:** filtri di qualità, crop 512×512 con l'asse lungo verticale, testa in alto grazie a un classificatore lineare su DINOv2.
5. **Active learning semi-supervisionato** (pacchetto `bee_al`, con un'interfaccia web per l'annotazione), in due fasi:
   - fase 1: crop utilizzabili o da scartare;
   - fase 2: classi demografiche e malattia, solo sui crop utilizzabili.
6. **Conteggi per batch** dalle classi finali, con intervalli bootstrap sulle foto.

È stato sviluppato anche un generatore di varroa sintetica sui crop per l'anomaly detection: vedi `brief_varroa_sintetica_su_crop.md`. Su quel lavoro non ho risultati quantitativi: `[AUTORE DA CONFERMARE: se e come entra nella tesi]`.

## 2. Dati

- Foto di telaini, cartella `DatasetApi_Ceschi`, batch 3, 4, 5 e 6. `[AUTORE DA CONFERMARE: apiario, alveari, date delle sessioni, fotocamera]`.
- Dalla pipeline sono state tolte le 22 foto di `batch_6/favi_phone_camera`, scattate con il telefono. Restano **791 foto**.

| Batch | Foto | Note |
|---|---:|---|
| batch_3 | 168 | |
| batch_4 | 241 | |
| batch_5 | 199 | |
| batch_6 | 183 | 180 in `favi/` e 3 in `sick/` |

Sulle 3 foto della cartella `sick` il modello finale trova solo 59 api valide in fase 2, di cui 2 classificate Malata. Non spiegano quindi l'aumento di Malata nel batch_6 (§9).

## 3. Pseudo-etichette SAM 3

Script: `generate_labels_sam3.py`. Modello: `facebook/sam3` (Hugging Face transformers), prompt testuale "bee". Output in `processed/bee_batch3+4+5+6_SAM3`, nel formato di segmentazione YOLO.

SAM 3 lavora a tile, con la scala adattata a ogni foto:

1. **Stima della scala.** La foto viene divisa in una griglia 3×3. Ogni cella, ridotta a 1008 px, passa da SAM 3 con soglia 0,5, e si prende la lunghezza mediana delle api trovate. Non si usa la miniatura intera perché, nelle foto larghe e affollate, SAM 3 non vedeva api: 59 foto su 791 avevano 0-5 box.
2. **Tiling multi-scala.** La foto viene riscalata in modo che un'ape sia lunga 120 px e poi 150 px, con un fattore di scala tra 0,05 e 5. Ogni versione viene divisa in tile di 500 px con sovrapposizione del 40%, e i risultati delle due scale si uniscono. Con una sola scala circa l'1% delle api restava senza etichetta, con due circa lo 0,25%.
3. **Soglie sullo score.**
   - Sopra 0,5 le maschere si tengono tutte: alzare la soglia a 0,6 perde circa l'8% delle api.
   - Tra 0,3 e 0,5 una maschera si tiene solo se la sua area è almeno il 60% di quella dell'ape mediana della foto. Sotto 0,5 SAM 3 restituisce quasi solo teste di api infilate nelle celle, celle con polline e spazi vuoti.
   - Area minima 600 px² nella tile e area massima il 25% della tile.
4. **NMS sulle maschere invece che sui box.** Una maschera viene scartata quando più del 60% della maschera più piccola sta dentro una con score più alto. La NMS sui box lasciava circa il 5% di duplicati e frammenti.

I parametri sono stati tarati a vista sulla recall, su 18 foto scelte per coprire zoom diversi (api lunghe da 150 a 680 px) e diverse messe a fuoco. **Le etichette non sono state corrette a mano** `[AUTORE DA CONFERMARE]`.

Totale: 260.586 api su 791 foto, circa 329 per foto; il massimo è circa 1.850 api in una foto.

## 4. Segmentazione: fine-tuning di YOLO26s-seg (ablation)

Il dettaglio completo, con tutte le tabelle e le metriche, è in **`ablation_sam3.md`**: va allegato insieme a questo file. Qui ne riporto la sintesi.

### Split e setup
- Split **casuale per foto**:

  | Split | Foto | Api |
  |---|---:|---:|
  | Train | 554 | 182.318 |
  | Validation | 118 | 39.076 |
  | Test | 119 | 39.192 |

- YOLO26s-seg pre-addestrato, fine-tuning completo.
- Iperparametri:
  - fino a 200 epoche, early stopping con patience 40;
  - batch 2;
  - AdamW, lr 0,001 con decadimento coseno fino all'1%;
  - mosaic disattivato, `max_det` 2.500.
- Hardware: una RTX A6000 da 48 GB per run.
- La risoluzione alta di training è 1376 px, perché 1440 px va in out of memory a batch 2.
- Test: conf 0,10, sia alla risoluzione di training sia a 2048 px.
- Criterio di scelta: seg mAP50-95 sul test a 2048 px.

### Risultati (test, 119 foto, a 2048 px)

| Variante | seg mAP50-95 | seg mAP50 | seg mAP75 | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| base, 1280 px | 0,688 | 0,926 | 0,798 | 0,924 | 0,882 | 0,902 |
| **1376 px (adottata)** | **0,692** | 0,927 | 0,803 | 0,931 | 0,881 | 0,905 |
| 1376 + rotazione ±180° | 0,688 | 0,925 | 0,800 | 0,929 | 0,871 | 0,899 |
| 1376 + rotazione + copy-paste | 0,690 | 0,919 | 0,802 | 0,931 | 0,873 | 0,901 |
| 1376, freeze 5 | 0,685 | 0,921 | 0,794 | 0,931 | 0,881 | 0,905 |
| 1376, freeze 10 | 0,682 | 0,918 | 0,792 | 0,926 | 0,874 | 0,899 |
| 1376, freeze 20 | 0,580 | 0,835 | 0,673 | 0,881 | 0,778 | 0,826 |
| 1376, freeze 23 | 0,484 | 0,760 | 0,554 | 0,777 | 0,701 | 0,737 |
| 1376, seed 1 | 0,695 | 0,931 | 0,805 | 0,941 | 0,884 | 0,912 |

### Conclusioni principali
1. **Inferenza a 2048 px:** vale +4/6 punti di mAP50-95 per ogni variante, senza riaddestrare.
2. **Training a 1376 px invece di 1280:** +1,5 punti alla risoluzione di training, ma a 2048 px solo +0,4, che è entro il rumore tra seed (0,003).
3. **Rotazione e copy-paste:** nessun effetto misurabile.
4. **Freeze:** 5-10 layer costano circa un punto. Da 20 layer in su il modello crolla (−11 e −21 punti), soprattutto in recall: il fine-tuning completo è necessario.
5. **mAP75** è la metrica che separa di più le varianti buone: conta la precisione del contorno, mentre la detection è simile tra le varianti.

### Confronto con il primo ablation (LangSAM, batch 3+4)
È descritto in `thesis_brief_yolo_finetuning.md`.
- **LangSAM:** 298 foto di train, 56 di test, alta risoluzione 1440 px. Migliore 0,628 a 2048 px.
- **SAM 3:** migliore 0,692.

**I valori assoluti non sono confrontabili.** Ogni modello è valutato contro le pseudo-etichette dello strumento che ha generato le sue etichette di training, e cambiano foto e numero di immagini. Gli split si sovrappongono: 40 foto del test LangSAM sono nel train SAM 3, e 45 foto del test SAM 3 sono nel train LangSAM. Sono confrontabili solo le tendenze, e coincidono.

**Modello adottato:** `runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt`, usato a 2048 px.

## 5. Estrazione delle singole api

- Script: `extract_anomaly_dataset.py`, lanciato con il modello adottato, `--imgsz 2048` e `--conf 0.35` (default).
- Sorgente: le 791 foto di `bee_batch3+4+5+6_SAM3/images`.
- **Le 554 foto di train del modello YOLO sono incluse:** su quelle foto la segmentazione è più ottimistica che su foto nuove.
- **Normalizzazione per foto** prima del ritaglio: trasformazione lineare del canale L in LAB, con un solo guadagno e un solo offset per tutta la foto. Così le api della stessa foto restano confrontabili tra loro.
  - Le statistiche si misurano sui pixel delle api (se sono almeno 5.000), perché lo sfondo cambia da foto a foto.
  - Target: media L 110 e deviazione standard 42, cioè le mediane di 90 foto di batch3+4. Prima della correzione la media variava tra 72 e 139 e la deviazione tra 33 e 51.
  - Guadagno limitato tra 0,6 e 1,8.
- Risultato: **249.982 crop grezzi** (PNG RGBA) in `processed/single_bee3+4+5+6_SAM3/raw_bees`.

## 6. Pulizia dei crop e orientamento

Script: `dataset_cleaning.py`, pipeline versione 4. Output in `processed/single_bee3+4+5+6_SAM3/cleaned_bees`.

### Filtri (motivi di scarto)

| Motivo | Regola | Api scartate |
|---|---|---:|
| border | ape tagliata dal bordo del crop: più di 15 px di maschera in una fascia di 8 px | 31.917 |
| blur | punteggio di nitidezza < 50, misurato a una scala comune | 25.015 |
| short | lato lungo del rettangolo ruotato < 105 px: sotto circa 100 px erano addomi nelle celle, teste o ali | 9.388 |
| aspect | rapporto lunghezza/larghezza > 3 | 205 |
| fragments | tenendo solo il corpo principale si perde più del 20% della maschera | 60 |
| **Totale scartate** | | **66.585** (26,6%) |
| **Accettate** | | **183.397** |

Altre soglie del filtro: area minima 1.500 px², fill ratio ≥ 0,30, lato corto ≥ 25 px.

| Batch | Foto | Crop estratti | Accettati | % accettati |
|---|---:|---:|---:|---:|
| batch_3 | 168 | 45.182 | 34.131 | 75,5% |
| batch_4 | 241 | 99.639 | 73.611 | 73,9% |
| batch_5 | 199 | 58.914 | 41.805 | 71,0% |
| batch_6 | 183 | 46.247 | 33.850 | 73,2% |

### Normalizzazione
- Ogni crop è un PNG 512×512 a sfondo nero, con l'asse lungo dell'ape in verticale e il lato lungo pari a circa l'85% del frame.
- La maschera passa per chiusura, dilatazione, apertura (che separa le api adiacenti), recupero di zampe e antenne e smussatura; i bordi sono sfumati per 12 px.
- I kernel sono proporzionali alla lunghezza dell'ape, con riferimento 190 px.

### Testa in alto (`bee_orientation.py`)
- Metodo: regressione logistica (C = 0,1) su embedding DINOv2 ViT-S/14. L'input è la differenza tra l'embedding dell'ape e quello della stessa ape ruotata di 180°: lo score è quindi antisimmetrico per costruzione e il passo è idempotente.
- Etichette: 214 api di batch 3-6 orientate a mano (`config/head_orientation_labels.json`), escluse le ambigue.
- **Cross-validation a 10 fold: 0 errori su 214.** Le api ambigue (di lato, parziali) non sono tra le etichettate, quindi lì qualche errore è da aspettarsi.
- Sulle 183.397 api accettate:
  - **57.501 sono state ruotate** di 180° (31,4%);
  - **9.164 hanno |score| < 1**, cioè orientamento incerto (5,0%). Con |score| < 2 diventano 18.572, con |score| < 3 28.836.
  - Le api incerte non sono state ricontrollate a mano.

## 7. Metodo di active learning (`bee_al`)

Descrizione completa in `bee_al/README.md`.

### Rappresentazione
- **Embedding:** DINOv2 ViT-S/14 a 224 px. Nessuna testa di classificazione: le classi si assegnano sui vicini nello spazio degli embedding.

### Assegnazione delle classi
- **Propagazione delle etichette:** label spreading (Zhou et al.) su un grafo kNN con 15 vicini, α = 0,8, al massimo 30 iterazioni.
- **kNN di ripiego** (k = 10, temperatura 0,07).
- **Non assegnate:** il 5% di api più lontane da ogni esempio etichettato resta "non assegnata" e non entra nei cluster.

### Selezione delle api da etichettare
- Batch di 30 api.
- Il 70% per incertezza: margine tra le prime due classi della propagazione.
- Il 20% per novelty e il 10% casuale.
- Cold start sotto le 50 etichette.

### Interfaccia
L'interfaccia web ha tre modi di etichettare:
- coda di active learning, singola o a griglia;
- pagina Clusters, dove si etichettano a mano gruppi di api simili;
- pagina Review, con le etichette sospette perché in disaccordo con i vicini.

C'è anche un pulsante "Scarta ape" per le api che non si vogliono assegnare.

### Re-train contrastivo (a pulsante)
- Fine-tuning degli ultimi 2 blocchi di DINOv2, 300 passi, batch 96.
- Il 50% di ogni batch è etichettato; le pseudo-etichette entrano con confidenza ≥ 0,95.
- Loss: SupCon sulle api etichettate e pseudo-etichettate, SimCLR sulle altre. Temperatura 0,1, proiezione a 128 dimensioni.
- Learning rate: 1e-5 per il backbone, 1e-3 per la testa di proiezione.
- Dopo il re-train si rigenerano gli embedding e il grafo.

### Validation set
100 api casuali, fisse (seed 123), escluse dalla coda. **In nessuna delle due fasi è stato etichettato**, quindi accuracy e macro-F1 non sono disponibili e i modelli sono stati accettati senza un confronto di metriche.

### Criteri di stop
- Stabilità delle predizioni: meno dell'1% di api cambia classe per 3 round.
- Nessuna classe nuova da 3 round.
- Plateau della F1 sul validation set.

### Snapshot
Lo stato si congela in snapshot immutabili (`predictions.csv`, `frames.csv`, `meta.json`, etichette) e i cluster si esportano in cartelle per classe.

## 8. Fase 1: crop utilizzabili o da scartare

- **Dati:** tutte le 183.397 api pulite. Snapshot `fase1_finale_okvsscarta` del 15/10/2026, modello v003, 14 round.
- **Classi:** Ok (crop utilizzabile) e Scarta (segmentazione o crop non utilizzabile).
- **Annotazione:** un solo annotatore (l'autore), 1.735 eventi di etichetta e 1.561 api etichettate.
  - Per tipo di query: manual 1.505, uncertainty 112, diversity 58, novelty 39, random 21.
  - **Quasi tutte le etichette sono state scelte dall'annotatore dalla pagina Clusters, non estratte a caso.**
- **Re-train contrastivi:**

  | Modello | Etichette | Pseudo-etichette | Loss iniziale | Loss finale | Durata |
  |---|---:|---:|---:|---:|---:|
  | v001 | 386 | 35.899 | 4,20 | 3,01 | 8 min |
  | v002 | 1.070 | 9.037 | 4,20 | 3,07 | 8 min |
  | v003 | 1.537 | 20.237 | 4,18 | 3,06 | 8 min |

- **Risultato:**
  - Ok 138.788 (824 a mano, 137.964 predette);
  - Scarta 35.422 (737 a mano, 34.685 predette);
  - 9.087 non assegnate e 100 nel validation set.
- **Criteri di stop:**
  - nessuna classe nuova: soddisfatto;
  - stabilità: **non soddisfatta**, nei 3 ultimi round è cambiato il 3,15%, l'1,85% e l'1,96% delle api;
  - F1: non valutabile, perché manca il validation set.

Quota di Ok per batch (`scripts/batch_counts.py`). Le quote sono sulle api assegnate, con intervallo bootstrap al 95% sulle foto:

| Batch | Foto | Api | Ok | Scarta | Non assegnate |
|---|---:|---:|---|---|---:|
| batch_3 | 168 | 34.131 | 26.566 (81,4% [78,4-84,8]) | 6.054 (18,6%) | 1.511 |
| batch_4 | 241 | 73.611 | 55.879 (79,1% [77,2-80,7]) | 14.783 (20,9%) | 2.949 |
| batch_5 | 199 | 41.805 | 29.777 (76,8% [75,0-78,9]) | 8.980 (23,2%) | 3.048 |
| batch_6 | 183 | 33.850 | 26.640 (82,6% [80,5-84,9]) | 5.631 (17,4%) | 1.579 |

Il bootstrap misura la variabilità da foto a foto, **non gli errori del modello**.

**Esperienza precedente (pilota su batch 3+4, archiviata).** C'era stata una prima run con etichette LangSAM, 68.559 api e classi Crop Utilizzabile / Ape Incompleta / Crop Non Utilizzabile. Lì il validation set era stato etichettato, e mostrava che il modello lasciava passare circa 2/3 dei crop non utilizzabili. È il motivo per cui la fase 1 esiste come passo a sé. `[AUTORE DA CONFERMARE: se citare il pilota nella tesi]`

## 9. Fase 2: profilo demografico e malattia

- **Dati:** solo le 138.788 api Ok della fase 1, con gli embedding DINOv2 originali (v000) come punto di partenza.
- **Snapshot** `fase2_finale` del 16/10/2026: modello v004, 9 round. Export in `single_bee3+4+5+6_SAM3_clusters/fase2_finale`.
- **Classi**, come definite dall'autore:
  - **Giovane, Adulte, Fuco:** profilo demografico.
  - **Malata:** api malate in generale.
  - **Non Classificabile:** api che non si possono classificare per segmentazione o posa. È una classe di scarto: il modello la impara, ma resta fuori dalle quote.
  - Criteri visivi usati per l'età: `[AUTORE DA CONFERMARE]`. Un'indicazione data durante il lavoro era pelo del torace e stato delle ali.
- **Annotazione:** un annotatore, 1.954 eventi e 1.837 api etichettate.
  - Per tipo di query: manual 1.786, diversity 65, uncertainty 61, novelty 25, random 17.
- **Re-train contrastivi:**

  | Modello | Etichette | Pseudo-etichette | Loss iniziale | Loss finale |
  |---|---:|---:|---:|---:|
  | v001 | 607 | 16.711 | 4,13 | 2,31 |
  | v002 | 1.019 | 3.184 | 4,04 | 2,37 |
  | v003 | 1.338 | 2.684 | 4,07 | 2,37 |
  | v004 | 1.685 | 1.745 | 4,02 | 2,39 |

  Le pseudo-etichette sopra 0,95 calano da un modello all'altro.
- **Classi finali:**

  | Classe | A mano | Predette | Totale | % delle 131.829 api nei cluster |
  |---|---:|---:|---:|---:|
  | Adulte | 1.250 | 122.432 | 123.682 | 93,8% |
  | Non Classificabile | 129 | 3.821 | 3.950 | 3,0% |
  | Giovane | 99 | 1.994 | 2.093 | 1,6% |
  | Fuco | 130 | 1.271 | 1.401 | 1,1% |
  | Malata | 229 | 474 | 703 | 0,5% |

  Inoltre 6.842 non assegnate, 100 nel validation set e 17 "Scarta ape".
- **Criteri di stop:**
  - nessuna classe nuova: soddisfatto;
  - stabilità: **lontana dalla soglia**, nei 3 ultimi round è cambiato il 9,49%, il 13,92% e il 9,14% delle api;
  - F1: non valutabile.

**Conteggi per batch** (`batch_counts.py`, bootstrap sulle foto, 1.000 repliche).
- Le quote del profilo sono calcolate sul totale Giovane + Adulte + Fuco.
- Malata è in % di tutte le api classificate, cioè senza Non Classificabile e senza le non assegnate.

| Batch | Foto | Api | Giovane (% profilo) | Adulte (% profilo) | Fuco (% profilo) | Malata (% classificate) | Non Classificabile | Non assegnate |
|---|---:|---:|---|---|---|---|---:|---:|
| batch_3 | 168 | 26.548 | 1,0% [0,8-1,2] | 98,7% [98,5-99,0] | 0,3% [0,2-0,3] | 0,2% [0,1-0,2] (42) | 685 | 1.392 |
| batch_4 | 241 | 55.850 | 2,1% [1,8-2,4] | 96,3% [95,9-96,7] | 1,6% [1,4-1,7] | 0,4% [0,3-0,4] (198) | 1.713 | 2.303 |
| batch_5 | 199 | 29.761 | 1,5% [1,3-1,8] | 97,2% [96,7-97,6] | 1,3% [1,0-1,6] | 0,4% [0,3-0,5] (109) | 898 | 1.921 |
| batch_6 | 183 | 26.629 | 1,5% [1,3-1,7] | 97,8% [97,5-98,0] | 0,8% [0,6-0,9] | 1,4% [1,1-1,8] (354) | 658 | 1.226 |

Il batch_6 ha una quota di Malata circa 3-7 volte più alta degli altri. Le foto con più api Malata sono tutte nel batch_6 (DSC_6797: 19, DSC_6796: 18, DSC_6785: 16). Non vengono dalla cartella `sick` (§2).

### Come leggere questi numeri (da scrivere nella tesi)
1. **Non sono validati.** Il validation set di 100 api casuali non è stato etichettato, e non è stato fatto l'audit alla cieca (`bee_al/audit.py`, `scripts/audit_report.py`, disponibili ma non usati). Gli intervalli mostrano solo la variabilità tra foto.
2. **Probabile sottostima delle classi rare.**
   - La propagazione su grafo favorisce la classe con più esempi: Adulte ha 1.250 esempi, le altre 99-229.
   - Per Malata un terzo del totale (229 su 703) sono le api etichettate a mano. Il modello quindi ne ha trovate solo 474 in più su circa 130.000 api.
   - Le quote di Giovane, Fuco e Malata vanno lette come limiti inferiori o come confronto relativo tra batch, non come prevalenze assolute.
3. **Campione di etichette non casuale.** Il 91% delle etichette è stato scelto dall'annotatore dalla pagina Clusters. Non si possono usare per stimare l'accuratezza.
4. **Modello non stabilizzato.** Tra un round e l'altro cambia classe il 9-14% delle api.
5. **Un solo annotatore**, senza misura di accordo tra annotatori.
6. **Le classi di fase 2 sono state definite dopo lo snapshot.** Lo snapshot `fase2_finale` è stato fatto con la lista di classi vuota nella config. Classi di scarto, demografiche e di malattia sono state applicate dopo, al momento del conteggio, leggendo `config/al_fase2_sam3.yaml`. Le etichette e le predizioni non cambiano.
7. **Effetto a cascata della fase 1.** Le api "Scarta" predette per errore non arrivano in fase 2, e le api "Ok" non utilizzabili sono finite in parte in Non Classificabile.

Passi che renderebbero i numeri difendibili, non ancora fatti: `[AUTORE DA CONFERMARE se farli prima della consegna]`
- etichettare i 100 api del validation set di fase 2;
- fare l'audit stratificato per confidenza e ottenere stime di prevalenza corrette;
- eventualmente bilanciare le masse delle classi nella propagazione.

## 10. Limiti generali (sezione limiti della tesi)

- **Segmentazione**
  - Ground truth = pseudo-etichette SAM 3 non corrette a mano.
  - Split casuale per foto e non per sequenza di scatto o favo: foto quasi identiche possono stare una in train e una in test. Le metriche assolute possono quindi essere ottimistiche, ma il confronto tra varianti resta valido.
  - Un solo seed ripetuto. Alcuni run (freeze10, seed1) sono stati ripresi da checkpoint dopo out of memory causati da altri utenti della GPU; i `best.pt` testati precedono le riprese.
- **Estrazione:** fatta anche sulle foto di train del modello YOLO.
- **Pulizia:**
  - soglie scelte a vista;
  - non c'è una misura di quante api buone vengano scartate;
  - 5% di orientamenti incerti non ricontrollati.
- **Classificazione:** vedi §9, punti 1-7.
- **Metadati:** non c'è una corrispondenza foto → alveare. La "sessione" coincide con il batch, quindi non si può separare l'effetto del tempo da quello dell'alveare `[AUTORE DA CONFERMARE: alveari fotografati in ogni batch]`.

## 11. File e riproducibilità

| Cosa | Dove |
|---|---|
| Ablation SAM 3, dettaglio completo | `ablation_sam3.md`, `logs/ablation_sam3/` |
| Ablation LangSAM (precedente) | `thesis_brief_yolo_finetuning.md` |
| Pseudo-etichette SAM 3 | `generate_labels_sam3.py`, `processed/bee_batch3+4+5+6_SAM3`, `logs/generate_labels_sam3.log` |
| Split | `split_dataset_sam3.py`, `data_sam3.yaml` |
| Estrazione | `extract_anomaly_dataset.py`, `logs/extract_single_bee3+4+5+6_SAM3.log` |
| Pulizia e orientamento | `dataset_cleaning.py`, `bee_orientation.py`, `logs/cleaning_single_bee3+4+5+6_SAM3.log`, manifest `cleaned_bees/_cleaning_manifest.json` |
| Active learning | `bee_al/` (README), config `config/al_config.yaml` (fase 1) e `config/al_fase2_sam3.yaml` (fase 2) |
| Snapshot fase 1 | `/mnt/disk1/tommaso_ballarin/bee_al/results_sam3_b3456/snapshots/fase1_finale_okvsscarta/` (con `batch_counts/`) |
| Snapshot fase 2 | `/mnt/disk1/tommaso_ballarin/bee_al/results_fase2_sam3/snapshots/fase2_finale/` (con `batch_counts/`: CSV per batch e per foto, grafico `batch_counts.png`) |
| Cluster esportati | `processed/single_bee3+4+5+6_SAM3_clusters/{fase1_finale_okvsscarta, fase2_finale}` |
| Modelli contrastivi | `/mnt/disk1/tommaso_ballarin/bee_al/models_{sam3_b3456,fase2_sam3}/v00N/` (`meta.json`, `train_log.json`) |
| Varroa sintetica | `brief_varroa_sintetica_su_crop.md`, `varroa_on_crops.py`, `build_varroa_split.py` |

Software: Ultralytics 8.4.138, PyTorch 2.14.0, Python 3.10, GPU NVIDIA RTX A6000 da 48 GB (server condiviso).

I grafici pronti sono `batch_counts/batch_counts.png` dei due snapshot e quelli di Ultralytics nelle cartelle dei run (`results.png`, curve PR).
