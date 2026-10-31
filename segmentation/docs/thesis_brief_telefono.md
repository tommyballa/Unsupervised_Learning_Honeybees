# Brief per la tesi: foto scattate con il telefono (cambio di dominio)

Questo è un briefing da dare a Claude per scrivere la tesi magistrale. Integra `thesis_brief_risultati.md` (pipeline completa su batch 3-6) e contiene quello che è stato fatto e misurato il 23 ottobre 2026 sulle 22 foto del telefono. Se il codice o i dati sono cambiati dopo questa data, fanno fede il codice e i dati.

## Istruzioni per chi scrive (Claude)

- Valgono le stesse regole di `thesis_brief_risultati.md`:
  - solo fatti e numeri presenti qui;
  - i marcatori `[AUTORE DA CONFERMARE]` restano visibili;
  - differenze piccole si descrivono come "entro il rumore".
- Le metriche di segmentazione sono misurate contro le **pseudo-etichette SAM 3**, non contro una ground truth manuale.
- Le quote delle classi sono **output del modello, non validate**: il validation set della fase dedicata non è stato etichettato (§6).
- Tutti i numeri sono in `visualizations/thesis_telefono/metrics.json`. Le figure sono nella stessa cartella.

## 1. Perché questo esperimento

Il modello YOLO e la fase 2 dell'active learning sono stati costruiti su foto di una fotocamera, nei batch 3-6. Sapere come si comporta la pipeline con un telefono interessa per l'uso pratico in apiario. Domande:

1. La segmentazione regge il cambio di dispositivo?
2. Lo spazio DINOv2 dell'active learning riconosce le api del telefono come simili a quelle già etichettate?
3. Una fase di annotazione dedicata, con pochi esempi, basta per organizzarle in classi?

## 2. Dati

- **22 foto** in `batch_6/favi_phone_camera`, scattate l'11/10/2026 tra le 17:59 e le 18:22 (dal nome dei file).
  - 21 foto sono a 4000×3000 e una a 4608×3456.
  - 4 foto hanno orientamento EXIF 3, cioè sono ruotate di 180°.
  - `[AUTORE DA CONFERMARE: modello del telefono, alveare, se sono gli stessi favi fotografati nel batch_6]`.
- **Non sono mai state usate** per allenare YOLO: erano state tolte dalle 791 foto della pipeline (`thesis_brief_risultati.md` §2).
- **Etichette:** pseudo-etichette SAM 3 in `processed/phone_captured`, prodotte con lo stesso procedimento di batch 3-6. In totale 6.959 api, in media 316 per foto.
- **Dimensione delle api.**
  - Lunghezza mediana di un'ape: 229 px su foto da 4000 px nel telefono, 264 px su foto da 6016 px in batch 3-6.
  - Riportate a 2048 px per YOLO, le api del telefono sono circa 1,5 volte più grandi: 117 px contro 90 px.
  - La foto `IMG_20261011_180019` è un primo piano con api molto più grandi.

**Problema trovato nelle pseudo-etichette.**
- Nelle 4 foto con EXIF 3, le etichette SAM 3 erano ruotate di 180° rispetto all'immagine che legge YOLO. YOLO, tramite OpenCV, applica l'orientamento EXIF.
- Le etichette originali non sono state modificate. Le metriche sotto usano una copia corretta (x→1−x, y→1−y), in `/mnt/disk1/tommaso_ballarin/phone_captured_labels_ruotate`.
- Le metriche con le etichette sbagliate sono riportate solo per completezza. Vanno citate al più come nota metodologica: bisogna sempre controllare l'orientamento EXIF quando si generano etichette.

## 3. Segmentazione: YOLO sulle foto del telefono

- **Modello:** il migliore dell'ablation, YOLO26s-seg addestrato a 1376 px (`thesis_brief_risultati.md` §4).
- **Protocollo:** lo stesso del test set, cioè `conf=0.10` e `max_det=2500` (`logs/phone/phone_eval_data.py`).

| Dati | Inferenza | seg mAP50-95 | seg mAP50 | seg mAP75 | P | R |
|---|---|---:|---:|---:|---:|---:|
| Test batch 3-6 (119 foto, riferimento) | 2048 px | 0,692 | 0,927 | 0,803 | 0,931 | 0,881 |
| Telefono (22 foto) | 2048 px | 0,527 | 0,804 | 0,589 | 0,884 | 0,765 |
| Telefono | 1376 px | 0,529 | 0,828 | 0,584 | 0,908 | 0,789 |
| Telefono, modello base 1280 px | 1280 px | 0,528 | 0,827 | 0,581 | 0,903 | 0,785 |
| *Telefono, etichette non corrette (da non usare)* | 2048 px | 0,425 | 0,651 | 0,472 | 0,793 | 0,677 |

- **Il calo dovuto al cambio di dominio è di circa 0,16 mAP50-95**, da 0,692 a 0,527.
- Sul telefono i tre modelli o risoluzioni sono **entro il rumore**: fra 0,527 e 0,529.
- Sul test di batch 3-6 l'inferenza a 2048 px dava +0,046 rispetto a 1376 px. Sul telefono questo vantaggio sparisce. Una spiegazione compatibile, non verificata, è che a 2048 px le api del telefono siano più grandi di quelle viste in training (§2).

**Abbinamento maschera per maschera con SAM 3** (IoU ≥ 0,5, greedy per confidenza; `logs/phone/phone_compare.py`):
- **Alla soglia usata per l'estrazione (conf ≥ 0,35):** YOLO trova 5.768 api, di cui 5.184 abbinate. La **precisione è 0,90**, la **recall è 0,74**.
- **Alla soglia della valutazione (conf ≥ 0,10):** precisione 0,68, recall 0,83.
- La mediana per foto dell'IoU medio delle api abbinate è 0,83. Quando YOLO trova un'ape, la maschera è buona: il calo viene soprattutto dalle api mancate.
- **Figura 1** (`fig1_seg_per_foto.png`): precisione e recall per foto. 19 foto su 22 hanno recall tra 0,63 e 0,87. Le peggiori sono:
  - il primo piano `180019`: recall 0,22, YOLO trova 34 api su 120;
  - `175932` e `180633`: recall 0,54 e 0,59.
- Esempi visivi in `visualizations/phone_yolo_vs_sam3_ruotate/`: SAM 3 a sinistra, YOLO a destra, in rosso le api che l'altro non ha.
  - A vista, le api perse da YOLO sono soprattutto api ad ali aperte, sovrapposte o sul bordo della foto.
  - Le api in più trovate da YOLO sono in parte omissioni di SAM 3: le pseudo-etichette non sono una verità perfetta.

## 4. Estrazione e pulizia

Pipeline identica a batch 3-6: estrazione a 2048 px con conf 0,35, normalizzazione per foto, pulizia v4 con testa in alto. Output in `processed/single_bee_phone_SAM3/`.

- **Api:** 5.768 crop grezzi, 4.689 accettati (81,3%).
- **Motivi di scarto:**
  - bordo 915;
  - troppo corte 159;
  - proporzioni 4;
  - sfocatura 1.
- **Il filtro di sfocatura non scatta quasi mai, anche se a vista molti crop del telefono sono sfocati o "impastati"** (vedi le figure con i crop). La soglia è stata tarata sulla fotocamera.
- Testa in alto: ruotate di 180° 1.547 api su 4.689.

## 5. Distanza di dominio nello spazio DINOv2

Le api del telefono sono state passate al **modello della fase 1** (Ok / Scarta, embedding v003, addestrato su batch 3-6) con lo stesso criterio usato per il batch 7 (`scripts/add_dataset_to_stage.py`).

- Un'ape è "non assegnata" quando la sua similarità coseno con l'ape etichettata più vicina non supera **0,674**. È la soglia che nella fase 1 separava le non assegnate.

| Dati | Api | Ok | Scarta | Non assegnate | Similarità mediana col più vicino etichettato |
|---|---:|---:|---:|---:|---:|
| batch 3-6 (dati della fase 1) | 183.397 | | | 5,0% | 0,764 |
| batch 7 (fotocamera, sessione nuova) | 44.810 | 34.645 | 8.108 | **4,6%** | 0,766 |
| telefono | 4.689 | 895 | 167 | **77,4%** | 0,626 |

- **Figura 2** (`fig2_similarita_dominio.png`): distribuzione della similarità per batch 3-6 e per il telefono. Quella del telefono è spostata a sinistra, in gran parte sotto la soglia. Batch 7 non è nella figura, ma ha una distribuzione quasi identica a batch 3-6 (mediana 0,766 contro 0,764, tabella sopra).
- **Lettura:** una sessione nuova con la stessa fotocamera (batch 7) non è un cambio di dominio per DINOv2. Il telefono lo è.
- Questo fa capire che la quota di "non assegnate" è un indicatore semplice e senza etichette per accorgersi di un cambio di dominio.

Le 895 api Ok sono state anche proiettate nella fase 2 di batch 3-6, come dataset tenuto da parte (`embedding_map/fase2_v004_b3456_b7_phone/solo_phone`). Lì:
- il 33% resta non assegnato, contro il 4,5% del batch 7 e il 4,9% di batch 3-6;
- il 65% va in Adulte.

Anche le api del telefono che sembrano buone restano lontane dagli esempi della fotocamera.

## 6. Fase dedicata: active learning solo sul telefono

Per vedere se poche etichette bastano, è stata fatta una fase di annotazione a parte (`config/al_fase2_phone.yaml`, hub separato):
- su **tutte le 4.689 api pulite**, senza il filtro della fase 1;
- **da zero**, con embedding DINOv2 originali e nessuna etichetta;
- con le stesse classi della fase 2: Giovane, Adulte, Fuco, Malata, Non Classificabile (classe di scarto).

**Snapshot** `foto_telefono` del 23/10/2026 14:21: modello v002, 4 round, circa 20 minuti di annotazione (prima etichetta 13:58, ultima 14:17).

**Annotazione:** un annotatore (l'autore), 200 api etichettate.
- Per classe: Non Classificabile 94, Adulte 77, Giovane 18, Malata 11, Fuco 0.
- Per tipo di query: manual 151 (scelte dalla pagina Clusters), diversity 44, random 5.
- Review delle etichette sospette: 13 controllate, tutte confermate. Il modello suggeriva sempre Adulte, l'annotatore ha tenuto Non Classificabile (6), Giovane (4) e Malata (3).

**Re-train contrastivi** (supervised contrastive + SimCLR, 300 passi):

| Modello | Etichette | Pseudo-etichette | Loss iniziale | Loss finale | Durata |
|---|---:|---:|---:|---:|---:|
| v001 | 178 | 90 | 3,94 | 2,20 | 72 s |
| v002 | 200 | 110 | 3,89 | 2,27 | 69 s |

**Stabilità:** cambia classe il 18,7% delle api da v000 a v001 e il 6,0% da v001 a v002. Non è stabilizzato: la soglia del criterio di stop è l'1%.

**Classi finali** (`batch_counts.py`; le api del validation set contano con la classe predetta):

| Classe | Api | % delle 4.689 | % delle classificate (3.215) |
|---|---:|---:|---:|
| Adulte | 3.115 | 66,4% | 96,9% [96,2-97,7] |
| Giovane | 87 | 1,9% | 2,7% [2,1-3,3] |
| Malata | 13 | 0,3% | 0,4% [0,2-0,7] |
| Fuco | 0 | 0 | 0 |
| Non Classificabile | 1.255 | 26,8% | (scarto) |
| Non assegnate | 219 | 4,7% | |

Gli intervalli vengono dal bootstrap sulle 22 foto e **non misurano gli errori del modello**.

**Confronto con batch 3-6** (fase 2, `thesis_brief_risultati.md` §9):

| | batch 3-6 | telefono |
|---|---|---|
| Non Classificabile, % di tutte le api | 3,0% | **26,8%** |
| Non assegnate | 4,9% | 4,7% |
| Giovane, % profilo | 1,0-2,1% | 2,7% |
| Fuco, % profilo | 0,3-1,6% | 0% (nessuna etichetta) |
| Malata, % classificate | 0,2-1,4% | 0,4% |

- **Dopo la fase dedicata le non assegnate tornano al livello di batch 3-6**, dal 77% al 4,7%. Con circa 200 etichette dello stesso dominio, lo spazio DINOv2 copre di nuovo le api del telefono.
- **Circa un'ape su quattro è Non Classificabile**, contro il 3% della fotocamera. Questo è coerente con la qualità visiva dei crop (sfocati, impastati) e con le foto più affollate. Va letto insieme al punto 2 dei limiti (§7).
- I numeri di Giovane, Malata e Fuco **non sono confrontabili** con batch 3-6. Le etichette sono poche (18 e 11) e per Fuco nessuna, quindi il modello non può trovarne.

**Figure:**
- **Figura 3** (`fig3_fase1_vs_fase_telefono.png`) mette a confronto il giudizio del modello della fotocamera (filtro della fase 1) con la classe data nella fase dedicata, sulle stesse api:
  - delle **895 Ok** per il modello della fotocamera, il 94% è Adulte nella fase dedicata;
  - delle **167 Scarta**, il 57% è Non Classificabile;
  - delle **3.627 non assegnate**, il 61% è Adulte e il 31% Non Classificabile.
  
  Quindi il filtro della fase 1, quando decide, è coerente con l'annotazione dedicata. Ma scartava come "non assegnate" anche circa 2.200 api che, con etichette del telefono, risultano Adulte. **Usare il modello della fotocamera sul telefono avrebbe perso quasi tre quarti dei dati.**
- **Figura 4** (`fig4_composizione_per_foto.png`): classi per foto. Non Classificabile va dall'11% al 46%. È più alta nel primo piano `180019` e nelle ultime foto della sequenza (`182238`, `182242`: 44-45%).
- **Mappa UMAP** (`map_overview.png`, `map_by_cluster.png`; embedding v002, 4.689 api):
  - Non Classificabile occupa una regione propria, a destra;
  - Adulte occupa il resto;
  - Giovane è sparsa, con un piccolo gruppo in basso;
  - le non assegnate stanno soprattutto ai margini della regione Non Classificabile.
- **Crop di esempio:** `crops_<classe>.png`, 24 a caso per classe.

**Coerenza interna delle etichette** (leave-one-out k-NN sulle 200 api etichettate, k=10, come il kNN dell'hub):

| Embedding | Accuratezza | macro-F1 | F1 Adulte | F1 Giovane | F1 Malata | F1 Non Class. |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 originale (v000) | 0,69 | 0,47 | 0,71 | 0,10 | 0,31 | 0,76 |
| dopo re-train (v002) | 0,72 | 0,51 | 0,75 | 0,09 | 0,40 | 0,80 |

- Giovane viene confusa quasi sempre con Adulte: 17 su 18 con v000. Malata viene presa per Adulte 7 volte su 11.
- **Questa non è una stima dell'accuratezza**:
  - le etichette sono state scelte quasi tutte a mano dalla pagina Clusters;
  - la riga v002 è ottimistica, perché il re-train ha visto proprio queste etichette.
- Serve solo a dire che, con queste poche etichette, DINOv2 separa bene Adulte da Non Classificabile ma **non distingue l'età né la malattia**.

## 7. Limiti (da riportare)

1. **Segmentazione contro pseudo-etichette.** Le pseudo-etichette SAM 3 non sono corrette a mano, e per 4 foto erano anche ruotate (§2). Le metriche sono "accordo con SAM 3", non accuratezza.
2. **Classi non validate.**
   - Il validation set (100 api casuali) non è stato etichettato e non c'è audit.
   - Il 76% delle etichette è stato scelto dall'annotatore, quindi non è un campione casuale.
   - Il 47% delle etichette è Non Classificabile. Questo può spingere la propagazione verso quella classe.
3. **Modello non stabilizzato:** il 6% delle api cambia classe all'ultimo round.
4. **Poche foto:** 22, di una sola sessione e di un solo telefono `[AUTORE DA CONFERMARE]`. Non si può separare l'effetto del dispositivo da quello di luce, distanza e favo.
5. **Un solo annotatore**, nessuna misura di accordo.
6. **Pulizia tarata sulla fotocamera.** Il filtro di sfocatura non scarta i crop sfocati del telefono: questi finiscono in Non Classificabile invece che nello scarto della pulizia.

## 8. Messaggi per la tesi

- La **segmentazione si trasferisce in parte**: −0,16 mAP50-95. Le maschere trovate sono buone (IoU mediano 0,83), ma circa un'ape su quattro viene persa. Il problema peggiora con i primi piani.
- In DINOv2 **il telefono è un dominio diverso, la stessa fotocamera in un'altra sessione no** (batch 7). La quota di api non assegnate (77% contro 5%) lo segnala senza etichette.
- **Una piccola fase di annotazione dedicata** (200 etichette, circa 20 minuti) basta a coprire di nuovo il dominio: le non assegnate scendono dal 77% al 5%. Non basta però per le classi fini (Giovane, Malata, Fuco).
- Con il telefono **circa un quarto dei crop non è classificabile**, contro il 3% della fotocamera. Per l'uso in campo conta più la qualità dell'immagine che il modello.

## 9. File e riproducibilità

| Cosa | Dove |
|---|---|
| Valutazione YOLO | `logs/phone/phone_eval_data.py`, `logs/phone/phone_eval_ruotate.json` (etichette corrette), `logs/phone/phone_eval.json` (etichette originali), `logs/phone/data_phone*.yaml` |
| Confronto per foto con SAM 3 | `logs/phone/phone_compare.py`, `logs/phone/phone_compare_ruotate.csv`, immagini in `visualizations/phone_yolo_vs_sam3_ruotate/` |
| Etichette SAM 3 corrette | `/mnt/disk1/tommaso_ballarin/phone_captured_labels_ruotate/` (con `LEGGIMI.txt`) |
| Estrazione e pulizia | `logs/phone/run_extract_clean_phone.sh`, `logs/phone/extract_clean_phone.log`, `processed/single_bee_phone_SAM3/` |
| Filtro della fase 1 | `/mnt/disk1/tommaso_ballarin/bee_al/results_sam3_b3456/added_phone_predictions.csv` (e `added_b7_predictions.csv`) |
| Telefono nella fase 2 di batch 3-6 | dataset `phone` in `config/al_fase2_sam3.yaml` (tenuto da parte), mappa in `results_fase2_sam3/embedding_map/fase2_v004_b3456_b7_phone/` |
| Fase dedicata | `config/al_fase2_phone.yaml`, snapshot `/mnt/disk1/tommaso_ballarin/bee_al/results_fase2_phone/snapshots/foto_telefono/` (con `batch_counts/` e `report_tesi/`), modelli in `models_fase2_phone/` |
| Figure e metriche di questo brief | `logs/phone/thesis_report_phone.py` → `visualizations/thesis_telefono/` (copia di `report_tesi/`) |
