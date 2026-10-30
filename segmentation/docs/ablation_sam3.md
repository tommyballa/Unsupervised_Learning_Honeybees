# Ablation YOLO26s-seg sulle etichette SAM 3 (batch 3-6)

Fine-tuning di YOLO26s-seg per la segmentazione delle api, rifatto sulle pseudo-etichette SAM 3 dei batch 3-6. Ripete l'ablation fatto in precedenza sulle etichette LangSAM (batch 3+4, vedi `thesis_brief_yolo_finetuning.md`), con le stesse varianti e lo stesso protocollo di test.

Run svolti dal 12 al 16 ottobre 2026.

## Dataset

- Etichette: maschere poligonali generate con SAM 3 (`generate_labels_sam3.py`), cartella `bee_batch3+4+5+6_SAM3`. 791 foto, dopo aver tolto le 22 scattate con il telefono.
- Split: `split_dataset_sam3.py`, **casuale per foto**, in `processed/bee_batch3+4+5+6_SAM3_split`. Config: `data_sam3.yaml`.

| Split | Foto | Api | Media per foto |
|---|---|---|---|
| Train | 554 | 182 318 | 329 |
| Validation | 118 | 39 076 | 331 |
| Test | 119 | 39 192 | 329 |

Con SAM 3 una foto arriva a circa 1 850 api (con LangSAM il massimo era circa 1 370), per questo `max_det` è 2 500.

## Setup comune

- Modello: YOLO26s-seg pre-addestrato (`yolo26s-seg.pt`), fine-tuning completo; script `train_yolo.py`.
- Ultralytics 8.4.138, PyTorch 2.14.0, una NVIDIA RTX A6000 da 48 GB per run (server condiviso).
- Iperparametri identici all'ablation LangSAM: fino a 200 epoche, early stopping con patience 40, batch 2, AdamW (lr0 0.001, decadimento coseno fino all'1%, 5 epoche di warm-up, weight decay 0.0005), mosaic disattivato, `cache=True`, seed 0.
- Augmentation di base: HSV (h 0.015, s 0.7, v 0.4), rotazione ±10°, traslazione 0.1, scala ±50%, flip orizzontale 0.5, flip verticale 0.2.
- `max_det` 2 500.
- Nessun run ha avuto il batch ridotto per out of memory: sono tutti a batch 2.

**Risoluzione alta: 1376 px invece di 1440.** Nell'ablation LangSAM la risoluzione alta era 1440 px. Con SAM 3 le foto sono più affollate e 1440 px va in out of memory a batch 2. Prova di memoria con le due foto più affollate nello stesso batch (3 579 istanze):

| Risoluzione | Picco di memoria |
|---|---|
| 1280 px | 36.7 GB |
| 1344 px | 42.3 GB |
| 1376 px | 41.4 GB |
| 1440 px | out of memory |

## Protocollo di test

- Modello scelto: `best.pt`, cioè il checkpoint migliore sulla validation (fitness Ultralytics).
- Test: 119 foto, `split=test`, conf 0.10, `max_det` 2 500, script `logs/ablation_sam3/abl_eval.py`.
- Ogni modello è testato a due risoluzioni: quella di training e 2048 px.
- Metriche di segmentazione (maschere): mAP50-95, mAP50, mAP75, precision, recall, F1. In più mAP50-95 e mAP50 dei box.
- Criterio per il "modello migliore": seg mAP50-95 sul test a 2048 px.

## Run

| Run | Cartella (`runs/segment/runs/segment/`) | Training | Variazione rispetto alla base | Epoche (best) | Val seg mAP50-95 | Note |
|---|---|---|---|---|---|---|
| base | `bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3` | 1280 px | — | 147 (85) | 0.647 | early stop, 3.9 h |
| img1376 | `..._SAM3_img1376` | 1376 px | risoluzione 1376 | 144 (104) | 0.664 | early stop, 5.0 h |
| img1376_rot | `..._SAM3_img1376_rot` | 1376 px | rotazione ±180°, flip verticale 0.5 | 176 (136) | 0.656 | early stop, 4.8 h |
| img1376_rot_cp | `..._SAM3_img1376_rot_cp` | 1376 px | come sopra + copy-paste 0.3 | 198 (158) | 0.659 | early stop, 5.6 h |
| freeze5 | `..._SAM3_img1376_freeze5` | 1376 px | primi 5 layer congelati | 137 (97) | 0.649 | early stop, 3.7 h |
| freeze10 | `..._SAM3_img1376_freeze10` | 1376 px | primi 10 layer congelati | 200 (154) | 0.644 | ripreso 2 volte ¹ |
| freeze20 | `..._SAM3_img1376_freeze20` | 1376 px | primi 20 layer congelati | 200 (164) | 0.530 | |
| freeze23 | `..._SAM3_img1376_freeze23` | 1376 px | primi 23 layer congelati | 200 (164) | 0.476 | |
| seed1 | `..._SAM3_img1376_seed1` | 1376 px | img1376 con `seed=1` | 200 (82) | 0.662 | ripreso 7 volte ² |

¹ freeze10 è stato ripreso da `last.pt` alle epoche 161 e 184, dopo out of memory causati da altri processi sulla GPU. Il primo tentativo era stato interrotto il 13/10 dal disco di sistema pieno: è conservato come `..._freeze10__interrotto_disco_pieno_1013`.

² seed1 è stato ripreso alle epoche 99, 136, 145, 154, 163, 172 e 181, sempre per out of memory causati da altri processi. Il `best.pt` testato è dell'epoca 82, salvato prima della prima ripresa, quindi è identico a quello di un run senza interruzioni. Ogni ripresa però azzera il conteggio dell'early stopping: per questo il run è arrivato a 200 epoche invece di fermarsi verso la 122.

freeze10, freeze20, freeze23 e seed1 sono stati addestrati su un server condiviso: i tempi di training non sono confrontabili e non li riporto. I tentativi falliti sono conservati con il suffisso `__interrotto_*`, sia in `runs/` sia nei log.

## 1. Risoluzione e augmentation

### Test a 2048 px

| Run | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|
| base (1280) | 0.688 | 0.926 | 0.798 | 0.924 | 0.882 | 0.902 | 0.699 | 0.914 |
| **img1376** | **0.692** | **0.927** | **0.803** | **0.931** | 0.881 | **0.905** | 0.703 | **0.915** |
| img1376_rot | 0.688 | 0.925 | 0.800 | 0.929 | 0.871 | 0.899 | 0.703 | 0.914 |
| img1376_rot_cp | 0.690 | 0.919 | 0.802 | **0.931** | 0.873 | 0.901 | **0.709** | **0.915** |

### Test alla risoluzione di training

| Run | Test | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| base | 1280 | 0.631 | 0.912 | 0.724 | 0.913 | 0.864 | 0.888 | 0.669 | 0.900 |
| **img1376** | 1376 | **0.646** | **0.914** | **0.743** | 0.920 | **0.865** | **0.891** | **0.675** | 0.903 |
| img1376_rot | 1376 | 0.640 | 0.912 | 0.735 | 0.915 | 0.863 | 0.888 | 0.674 | **0.909** |
| img1376_rot_cp | 1376 | 0.641 | **0.914** | 0.737 | **0.921** | 0.860 | 0.889 | **0.675** | 0.903 |

## 2. Freeze (configurazione img1376)

`freeze=N` congela i primi N layer del modello: il backbone fino al layer 10, fino alla testa con 20-23.

### Test a 2048 px

| Run | freeze | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| **img1376** | 0 | **0.692** | **0.927** | **0.803** | **0.931** | **0.881** | **0.905** | **0.703** | **0.915** |
| freeze5 | 5 | 0.685 | 0.921 | 0.794 | **0.931** | **0.881** | **0.905** | 0.696 | **0.915** |
| freeze10 | 10 | 0.682 | 0.918 | 0.792 | 0.926 | 0.874 | 0.899 | 0.699 | 0.913 |
| freeze20 | 20 | 0.580 | 0.835 | 0.673 | 0.881 | 0.778 | 0.826 | 0.546 | 0.822 |
| freeze23 | 23 | 0.484 | 0.760 | 0.554 | 0.777 | 0.701 | 0.737 | 0.451 | 0.738 |

### Test a 1376 px

| Run | freeze | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| **img1376** | 0 | **0.646** | **0.914** | **0.743** | **0.920** | **0.865** | **0.891** | **0.675** | **0.903** |
| freeze5 | 5 | 0.631 | 0.904 | 0.719 | 0.916 | 0.854 | 0.884 | 0.658 | 0.899 |
| freeze10 | 10 | 0.628 | 0.903 | 0.718 | 0.915 | 0.853 | 0.883 | 0.661 | 0.899 |
| freeze20 | 20 | 0.496 | 0.756 | 0.558 | 0.841 | 0.706 | 0.768 | 0.466 | 0.744 |
| freeze23 | 23 | 0.446 | 0.715 | 0.500 | 0.775 | 0.666 | 0.716 | 0.418 | 0.700 |

## 3. Variabilità tra training (seed)

Stesso training di img1376 con `seed=1`.

| Run | Test | mAP50-95 | mAP50 | mAP75 | P | R | F1 | box mAP50-95 | box mAP50 |
|---|---|---|---|---|---|---|---|---|---|
| img1376 (seed 0) | 2048 | 0.692 | 0.927 | 0.803 | 0.931 | 0.881 | 0.905 | 0.703 | 0.915 |
| img1376 seed1 | 2048 | 0.695 | 0.931 | 0.805 | 0.941 | 0.884 | 0.912 | 0.705 | 0.918 |
| img1376 (seed 0) | 1376 | 0.646 | 0.914 | 0.743 | 0.920 | 0.865 | 0.891 | 0.675 | 0.903 |
| img1376 seed1 | 1376 | 0.645 | 0.915 | 0.741 | 0.925 | 0.863 | 0.892 | 0.670 | 0.903 |

Tra i due seed la differenza è di 0.003 in seg mAP50-95, sia a 2048 sia a 1376 px. Con LangSAM era 0.008: 0.628 contro 0.620 a 2048 px.

## 4. Confronto con l'ablation LangSAM

Seg mAP50-95 sul test a 2048 px. Per LangSAM: batch 3+4, test di 56 foto e 19 314 api, alta risoluzione a 1440 px. Per SAM 3: batch 3-6, test di 119 foto, alta risoluzione a 1376 px.

| Variante | LangSAM | SAM 3 |
|---|---|---|
| base 1280 | 0.609 | 0.688 |
| alta risoluzione (1440 / 1376) | **0.628** | 0.692 |
| + rotazione | 0.617 | 0.688 |
| + rotazione + copy-paste | 0.620 | 0.690 |
| freeze5 | 0.619 | 0.685 |
| freeze10 | 0.619 | 0.682 |
| freeze20 | 0.511 | 0.580 |
| freeze23 | 0.470 | 0.484 |
| seed1 | 0.620 | **0.695** |

Precision, recall e F1 a 2048 px, migliore configurazione:

| | P | R | F1 | mAP75 |
|---|---|---|---|---|
| LangSAM img1440 | 0.895 | 0.855 | 0.875 | 0.738 |
| SAM 3 img1376 | 0.931 | 0.881 | 0.905 | 0.803 |

**I due ablation non sono direttamente confrontabili in valore assoluto.** Cambiano le foto di test (56 contro 119), il numero di foto di training (298 contro 554) e, soprattutto, la ground truth. Ogni modello è valutato contro le pseudo-etichette dello stesso strumento che ha generato le sue etichette di training. Il +6/8 punti di SAM 3 può quindi venire da etichette più coerenti, da più dati o da un test più facile, e non dimostra maschere più vicine alle api reali. Per un confronto equo servirebbero i due modelli valutati sulle stesse foto, non viste da nessuno dei due in training, possibilmente con etichette corrette a mano. Gli split si sovrappongono: a giudicare dai nomi dei file, 40 foto del test LangSAM sono nel train SAM 3 e 45 foto del test SAM 3 sono nel train LangSAM.

Le **tendenze** invece sono confrontabili, e coincidono.

## Cosa mostrano i risultati

1. **Risoluzione di inferenza.** Testare a 2048 px vale +4/6 punti di seg mAP50-95 per ogni variante, con precision e recall entrambe più alte. Le api sono piccole: la mediana è circa 49 px sul lato lungo a 1280.
2. **Risoluzione di training.** 1376 px contro 1280 dà +1.5 punti alla risoluzione di training. A 2048 px la differenza (+0.4) è nell'ordine della variabilità tra seed.
3. **Augmentation geometrica e copy-paste.** Nessun effetto misurabile: con SAM 3 le varianti stanno entro 0.004 a 2048 px, sotto la differenza tra seed, mentre con LangSAM facevano perdere circa un punto. Le api sono già in tutte le orientazioni nelle foto. Il copy-paste trova poco spazio libero dove incollare, dato che le foto sono fitte di api e Ultralytics incolla solo dove la sovrapposizione è sotto il 30%.
4. **Freeze.** Congelare 5-10 layer costa circa un punto, poco ma sistematico in entrambi gli ablation. Da 20 layer in su il modello crolla: −11 punti con freeze20 e −21 con freeze23. Il crollo riguarda soprattutto la recall, da 0.88 a 0.78 e poi 0.70. Le feature pre-addestrate su COCO vanno adattate alle api: il fine-tuning completo è necessario.
5. **Contorni delle maschere.** La metrica che separa di più le configurazioni buone è mAP75, cioè la precisione del contorno. La detection è simile tra le varianti buone: box mAP50 sempre circa 0.91-0.92.

## Configurazione adottata

**`img1376`**: fine-tuning completo (nessun freeze), training a 1376 px, augmentation di base, inferenza a 2048 px.

Pesi: `runs/segment/runs/segment/bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3_img1376/weights/best.pt`

Il `best.pt` di seed1 è equivalente: è leggermente avanti in quasi tutte le metriche, ma entro la variabilità tra seed.

## Limiti

- **Ground truth:** sono pseudo-etichette SAM 3, non corrette a mano. Le metriche misurano l'accordo con SAM 3, non con le api reali.
- **Split:** è casuale per foto, non per sequenza di scatto o per favo. Foto dello stesso favo scattate a pochi secondi di distanza possono finire una nel train e una nel test, quindi le metriche assolute possono essere ottimistiche. Il confronto tra varianti resta valido, perché tutte usano lo stesso split.
- **Seed:** un solo seed ripetuto. La variabilità tra training è stimata da una sola coppia di run (0.003 qui, 0.008 con LangSAM).
- **Riprese:** freeze10 e seed1 sono stati ripresi da checkpoint dopo out of memory causati da altri utenti della GPU. Per seed1 il modello testato precede tutte le riprese. Per freeze10 il `best.pt` è dell'epoca 154, prima delle riprese, quindi anche in questo caso il modello testato non ne è influenzato.

## File

- Risultati di test: `logs/ablation_sam3/abl_<run>.json` e `.log`.
- Log di training:
  - `logs/train_batch3+4+5+6_SAM3_*.log` per base, img1376, rot, rot_cp e freeze5;
  - `logs/ablation_sam3/freeze_disk1/` per freeze10, freeze20, freeze23 e seed1, insieme ai tentativi interrotti, allo stato della coda (`queue_gpu0.out`) e agli script.
- Code di lancio: `logs/ablation_sam3/queue.sh` (risoluzione e augmentation) e `logs/ablation_sam3/freeze_disk1/queue_freeze_gated.sh` (freeze e seed).
- Copia originale dei run freeze10/20/23 e seed1: `/mnt/disk1/tommaso_ballarin/bee_ablation_sam3_freeze/`.
