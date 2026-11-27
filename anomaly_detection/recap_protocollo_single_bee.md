# Recap protocollo sperimentale — single-bee varroa

## 1. Dataset e divisione

**Dataset finale:** `single_bee3+4+5+6_SAM3_varroa_split` (crop 512×512 segmentati con SAM3).

- **Train:** 102.303 api sane (Young/Adult/Drone/Sick).
- **Test originale:** 24.297 sane + 1.279 con varroa sintetica (1 o 2 per ape), con maschere pixel-level.

**Test ridotto (usato fino al 18/10):** 5.000 sane estratte con seed 0 più tutte le 1.279 anomale, quindi circa il 20% di anomale. Senza il sottocampionamento il test completo avrebbe solo il 5% di anomale, e AP e F1 cambierebbero molto. Per comparabilità tutti i modelli usano il test ridotto.

**Split val / test held-out** (`experiments/single_bee/make_val_split.py`, seed 0):

- **Val:** il 25% delle anomale (322: 210 con una varroa, 112 con due), stratificato per cluster × numero di varroa. A queste si aggiungono 1.251 sane prese, stratificate per cluster, tra quelle **escluse** dal test ridotto. Anche qui le anomale sono circa il 20%.
- **Test held-out:** le altre 957 anomale (639 / 318) e le **stesse 5.000 sane** del test ridotto.
- Non ci sono immagini in comune tra i due split (c'è un assert nello script). Le 18.046 sane rimanenti non sono usate.

## 2. Come è stata scelta la configurazione migliore

Il processo ha avuto due livelli.

### a) Ricetta (iperparametri): sweep per modello su validation set

Script `sweep_<model>.py`, log in `logs/<model>_sweep/`.

- Varianti di backbone, layer, risoluzione, lr, numero di step, blur della mappa, score immagine, mascheramento dell'ape, ecc.
- Per velocità le metriche erano calcolate a 128 px, con le varianti di post-processing valutate sulle stesse mappe senza riaddestrare.
- Il criterio era l'AUROC image-level, guardando anche la pixel AP.
- Le scoperte valide per quasi tutti i modelli:
  - Risoluzione 512 px (nativa) o più alta.
  - Elaborazione solo sull'ape: loss mascherata e mappa limitata alla maschera erosa. Lo sfondo nero è circa l'86% del crop.
  - Blur della mappa con σ intorno a 8–16 px (la varroa misura circa 25–43 px).
  - Score immagine = massimo della mappa.
  - Training breve, perché i modelli addestrati raggiungono il picco presto e poi peggiorano.

### b) Scelta dello step/epoca e delle soglie, sul validation set

- Durante il training, `BestTracker` valuta il modello sul **val** a intervalli regolari. Tiene il checkpoint migliore con un criterio **lessicografico**: prima `img_roc_auc`, a parità `pxl_avg_prec`.
- I modelli senza training (PatchCore, PaDiM) non hanno step da scegliere: si fa un unico fit.
- Sul val si fissano anche:
  - la **soglia image-level** che massimizza l'F1;
  - la **soglia pixel-level** e la normalizzazione delle heatmap.
- Il **test held-out viene valutato una sola volta**, sul checkpoint scelto e con le soglie del val. `img_f1` e `pxl_f1` sono quindi alla soglia del val. Per confronto è riportato anche `*_f1_oracle`, cioè l'F1 alla soglia ottimale sul test.

## 3. Ricette finali e risultati sul test set

| Modello | Ricetta (sintesi) | Best | img AUROC | img AP | img F1 | pxl AP | AU-PRO |
|---|---|---|---|---|---|---|---|
| PatchCore | WRN50 layer2-3, bank 30k, σ 8, score = max | fit | **0.9986** | 0.993 | 0.954 | **0.777** | 0.998 |
| CFA | WRN50 layer2-3, 768 px, 32k centroidi su patch dell'ape, lr 1e-5 | step 348 | 0.9959 | 0.976 | 0.926 | 0.755 | 0.998 |
| Dinomaly | ViT-B ufficiale (layer 2-9), 10k iterazioni | iter 10000 | 0.9958 | 0.974 | 0.930 | 0.768 | 0.999 |
| PaDiM | WRN50 layer3, pool 3×3, Gaussiana condivisa, eps 1e-4, σ 4 | fit | 0.9886 | 0.933 | 0.881 | 0.521 | 0.994 |
| STFPM | R18, mappa solo layer3, 600 step, SGD lr 0.4 | step 540 | 0.9832 | 0.920 | 0.835 | 0.534 | 0.996 |
| FastFlow | WRN50, 512 px, loss NLL sui pixel dell'ape, z² su tutte le scale, erosione 24, σ 16 | step 125 | 0.9818 | 0.910 | 0.834 | 0.469 | 0.987 |
| PaSTe | R18, lr 0.4, 400 step, mappa layer2+3, σ 16 | step 350 | 0.9652 | 0.810 | 0.763 | 0.393 | 0.993 |
| RD4AD | WRN50, mappa layer2, 400 step, σ 8 | step 400 | 0.9531 | 0.787 | 0.732 | 0.349 | 0.989 |

PatchCore con MobileNetV2 (0.9987) e ResNet18 (0.9983) ha un AUROC equivalente ma una pixel AP più bassa (0.754 / 0.735). GANomaly è escluso dal confronto finale: non è stato rifatto sul dataset finale con il protocollo val (su batch3+4 la sua AUROC era 0.897).

## 4. Limiti da dichiarare nella tesi


2. **Le varroa sintetiche derivano da sole 2 texture sorgente**, condivise tra val e test. Val e test misurano quindi la stessa distribuzione di anomalie, e la generalizzazione a varroa reali non è dimostrata.
3. **Il rumore tra run è circa ±0.01 AUROC** per i modelli addestrati (RD4AD oscilla di ±0.02 tra una valutazione e l'altra). Le differenze sotto questa soglia, per esempio tra CFA e Dinomaly, non sono significative. C'è un solo seed.
4. **Le anomale sono circa il 20%** nel val e nel test ridotto, una proporzione scelta da noi e non quella naturale. Per questo AP e F1 vanno letti in questo contesto.
