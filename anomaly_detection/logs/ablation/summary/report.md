# Ablation: dimensione del training set e backbone

## Protocollo

Le ablation riguardano otto modelli: PatchCore, CFA, Dinomaly, PaDiM, STFPM, FastFlow, PaSTe e RD4AD.
Il dataset è quello finale (SAM3, batch 3–6) e il protocollo è lo stesso degli esperimenti principali:

- il checkpoint migliore è scelto sul validation set (322 api con varroa e 1251 sane);
- le metriche sono calcolate una sola volta sul test held-out (957 api con varroa e 5000 sane);
- le mappe e le maschere sono a 512×512.

In ogni ablation cambia un solo fattore. Il resto del recipe di ciascun modello resta invariato: learning rate, numero di step, layer allo stesso stride, post-processing della mappa. Non è stata fatta alcuna ri-ottimizzazione e c'è un solo seed per configurazione.

- **Numero di api di training.** Si usano sottoinsiemi casuali di N ∈ {500, 5 000, 20 000} api sane, oltre al training completo di 102 303 api.
  - I sottoinsiemi sono annidati (500 ⊂ 5 000 ⊂ 20 000) e ottenuti da una permutazione fissa.
  - Per CFA il numero di patch campionate per immagine nella costruzione della memory bank cresce al diminuire di N, in modo da mantenere circa 1,6 M patch candidate al clustering.
- **Backbone.** Con il training completo si confrontano tre backbone CNN pre-addestrati su ImageNet: WideResNet-50-2, ResNet-18 e MobileNetV2.
  - Per MobileNetV2 si usano i blocchi `features.3/6/13`, che hanno lo stesso stride dei layer ResNet 1/2/3.
  - FastFlow e RD4AD sono stati provati solo con ResNet-18 e WideResNet-50.
  - Per Dinomaly si confronta il ViT-B/14 con il ViT-S/14, entrambi DINOv2 con register.
- **Costo di inferenza.** È misurato sullo stesso calcolo della valutazione: modello in fp32, post-processing e mappa riportata a 512×512. Le grandezze misurate sono:
  - **GFLOP per immagine**, con `torch.utils.flop_counter`. La ricerca k-NN di PatchCore (`torch.cdist`) non è vista dal contatore ed è aggiunta come 2·n·m·d.
  - **Parametri**: tutti i tensori del modello, compresi memory bank e matrici di covarianza.
  - **Latenza a batch 1**: mediana su GPU NVIDIA e su CPU con 4 thread.

  Le GPU erano condivise con job di altri utenti, quindi le latenze vanno lette come indicative; il confronto principale si basa sui GFLOP.

## Risultati: numero di api di training

Tabella 1 – AUROC immagine / AP pixel sul test held-out al variare di N (backbone del recipe).

| Modello | N = 500 | N = 5 000 | N = 20 000 | N = 102 303 |
|---|---|---|---|---|
| PatchCore | 0.987 / 0.608 | 0.997 / 0.736 | 0.998 / 0.762 | 0.999 / 0.777 |
| CFA | 0.985 / 0.586 | 0.995 / 0.739 | 0.996 / 0.749 | 0.996 / 0.755 |
| Dinomaly | 0.991 / 0.731 | 0.995 / 0.763 | 0.996 / 0.762 | 0.996 / 0.768 |
| PaDiM | 0.986 / 0.480 | 0.989 / 0.518 | 0.989 / 0.522 | 0.989 / 0.521 |
| STFPM | 0.985 / 0.554 | 0.984 / 0.562 | 0.984 / 0.558 | 0.983 / 0.534 |
| FastFlow | 0.982 / 0.441 | 0.975 / 0.423 | 0.982 / 0.482 | 0.982 / 0.469 |
| PaSTe | 0.963 / 0.373 | 0.966 / 0.398 | 0.972 / 0.433 | 0.965 / 0.393 |
| RD4AD | 0.964 / 0.469 | 0.963 / 0.431 | 0.964 / 0.399 | 0.953 / 0.349 |

Figura: `train_size.pdf`.

I modelli basati su memoria sono quelli che risentono di più del numero di api normali disponibili: PatchCore, CFA e PaDiM modellano direttamente la distribuzione delle feature normali.

- **Con 500 api** PatchCore e CFA perdono circa un punto di AUROC e circa 17 punti di AP pixel rispetto al training completo.
- **Già con 5 000 api** recuperano quasi tutto il divario. Oltre 20 000 api il guadagno è marginale.
- **Dinomaly è il più robusto con pochi dati:** con 500 api raggiunge ancora AUROC 0.991 e AP pixel 0.731.

I modelli addestrati per discesa del gradiente (STFPM, FastFlow, PaSTe, RD4AD) hanno un numero di step fissato dal recipe. In pratica vedono qualche migliaio di immagini anche quando il training set è completo, e infatti le loro prestazioni non dipendono in modo sistematico da N: le differenze rientrano nella variabilità tra run.

RD4AD è un'eccezione: peggiora al crescere di N, soprattutto nell'AP pixel, che scende da 0.469 a 0.349. È coerente con un recipe ottimizzato per il training completo che non trae beneficio dalla maggiore varietà, ma con un solo seed il risultato va preso con cautela.

Dal punto di vista applicativo, poche migliaia di api sane bastano per installare il sistema su un nuovo alveare senza perdite rilevanti.

## Risultati: backbone e costo di inferenza

Tabella 2 – Training completo, backbone alternativi (★ = backbone del recipe). Latenze indicative (GPU condivisa).

| Modello | Backbone | AUROC img | AP pxl | Param. (M) | GFLOP | GPU bs1 (ms) | CPU (ms) |
|---|---|---|---|---|---|---|---|
| PatchCore | WRN50 ★ | 0.999 | 0.777 | 71.0 | 476 | 48 | 7590 |
| | R18 | 0.998 | 0.735 | 14.3 | 111 | 20 | 3040 |
| | MNv2 | 0.999 | 0.754 | 4.4 | 36 | 15 | 1460 |
| CFA | WRN50 ★ | 0.996 | 0.755 | 52.4 | 392 | 54 | 1990 |
| | R18 | 0.980 | 0.571 | 9.2 | 73 | 33 | 1510 |
| | MNv2 | 0.991 | 0.553 | 2.7 | 18 | 32 | 1430 |
| Dinomaly | ViT-B ★ | 0.996 | 0.768 | 148.0 | 424 | 59 | 1350 |
| | ViT-S | 0.996 | 0.678 | 37.4 | 121 | 21 | 640 |
| PaDiM | WRN50 ★ | 0.989 | 0.521 | 27.0 | 99 | 11 | 420 |
| | R18 | 0.641 | 0.305 | 2.9 | 15 | 3 | 290 |
| | MNv2 | 0.993 | 0.586 | 0.6 | 2.8 | 5 | 340 |
| STFPM | R18 ★ | 0.983 | 0.534 | 5.6 | 32 | 6 | 190 |
| | WRN50 | 0.985 | 0.568 | 49.8 | 194 | 18 | 960 |
| | MNv2 | 0.940 | 0.371 | 1.1 | 6.6 | 9 | 250 |
| FastFlow | WRN50 ★ | 0.982 | 0.469 | 102.9 | 341 | 49 | 1700 |
| | R18 | 0.913 | 0.304 | 10.4 | 30 | 14 | 470 |
| PaSTe | R18 ★ | 0.965 | 0.393 | 5.6 | 37 | 13 | 630 |
| | WRN50 | 0.916 | 0.196 | 49.8 | 199 | 27 | 1210 |
| | MNv2 | 0.795 | 0.092 | 1.1 | 13 | 16 | 570 |
| RD4AD | WRN50 ★ | 0.953 | 0.349 | 161.2 | 269 | 24 | 1000 |
| | R18 | 0.921 | 0.266 | 27.6 | 41 | 8 | 270 |

Figura: `efficiency.pdf`, che riporta AUROC immagine e AP pixel in funzione dei GFLOP, in scala logaritmica.

**I backbone leggeri.** Sulla detection a livello di immagine un backbone leggero è spesso sufficiente; la localizzazione, misurata con l'AP pixel, è più sensibile alla qualità delle feature.

- **PatchCore** con MobileNetV2 ha lo stesso AUROC del WideResNet-50 (0.999) e un'AP pixel di poco inferiore (0.754 contro 0.777), con 13 volte meno operazioni (36 contro 476 GFLOP) e 16 volte meno parametri. È il miglior compromesso tra accuratezza e costo e il candidato naturale per un'implementazione su dispositivo edge.
- **PaDiM** con MobileNetV2 supera addirittura il backbone del recipe, con AUROC 0.993 e solo 2,8 GFLOP.
- **Dinomaly** con ViT-S mantiene l'AUROC ma perde 9 punti di AP pixel.

**Dove il costo è concentrato.** Nei metodi a memoria come PatchCore buona parte del costo non è nel backbone ma nella ricerca k-NN: con WideResNet-50 sono 377 dei 476 GFLOP. Ridurre la dimensione delle feature con un backbone più piccolo abbassa quindi anche il costo della ricerca.

**I modelli addestrati.** Il comportamento è meno regolare, perché learning rate e numero di step sono stati ottimizzati per il backbone del recipe:

- FastFlow e RD4AD perdono molto passando a ResNet-18;
- per PaSTe, ottimizzato su ResNet-18, sia WideResNet-50 sia MobileNetV2 peggiorano.

Due casi negativi vanno letti con questa avvertenza:

- **PaSTe con MobileNetV2** (AUROC 0.795) ha il checkpoint migliore all'ultimo step, segno di un training non arrivato a convergenza con il recipe di ResNet-18.
- **PaDiM con ResNet-18** (AUROC 0.641) dipende dalle feature di layer3 combinate con una Gaussiana condivisa. Anche scegliendo sul validation set la regolarizzazione migliore della covarianza, l'AUROC di validazione non supera 0.82. Il problema è quindi legato alle feature e non a un errore di implementazione.

## Limiti

- C'è un solo seed per configurazione, quindi differenze sotto circa 0.005 di AUROC non sono significative.
- I recipe non sono stati ri-ottimizzati per ogni N e per ogni backbone. I risultati misurano la robustezza del recipe, non il massimo raggiungibile da ciascuna combinazione.
- Le latenze sono state misurate su GPU e CPU condivise; i GFLOP sono la misura di costo indipendente dal carico.

## File

- `logs/ablation/summary/ablation.csv`: tutte le metriche, cioè AUROC, AP e F1 immagine e AP, AUPRO e F1 pixel, insieme alle misure di costo.
- `logs/ablation/summary/train_size.pdf` e `efficiency.pdf`: le figure.
- Codice:
  - `experiments/single_bee/ablation.py`, con le variabili `ABL_TRAIN_N` e `ABL_BACKBONE`;
  - `measure_efficiency.py`;
  - `summarize_ablation.py`.
