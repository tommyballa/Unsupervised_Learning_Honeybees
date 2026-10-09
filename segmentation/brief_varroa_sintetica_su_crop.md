# Brief: come viene aggiunta la varroa sintetica (versione sui crop)

Documento di contesto per Claude. Descrive il metodo attuale con cui si generano
api "anomale" con varroa sintetica per l'anomaly detection. Scritto l'8 ottobre 2026:
se il codice è cambiato dopo, fa fede il codice.

## In breve

La varroa viene incollata **direttamente sul crop della singola ape già segmentata**
(PNG 512×512 a sfondo nero), non più sull'immagine intera del favo. Per ogni ape
anomala si salva anche una maschera binaria della varroa, da usare come ground truth
per le metriche pixel-level.

## Cosa è cambiato rispetto a prima

| | Metodo precedente | Metodo attuale |
|---|---|---|
| Dove si incolla | immagine intera del favo | crop 512×512 della singola ape |
| Script | `varroa_generator.py` + `extract_synthetic_anomaly_dataset.py` | `varroa_on_crops.py` + `build_varroa_split.py` |
| Segmentazione | YOLO gira *dopo* l'inserimento, poi `dataset_cleaning.clean_bee` | nessuna: i crop sono già segmentati e puliti |
| GT mask | maschera sul favo, trasformata con l'affine di `clean_bee` e intersecata con l'ape | generata già nel frame del crop |
| Posizione | dentro il poligono dell'ape, zona più satura/scura | solo sul corpo (vedi sotto) |

Conseguenze da tenere presenti:

- La varroa non passa più per YOLO né per la pulizia, quindi non può alterare la
  segmentazione e non viene mai tagliata dalla maschera dell'ape.
- La varroa non subisce più lo stesso ridimensionamento dell'ape (i crop sono api
  piccole ingrandite a 512). Per questo l'adattamento di nitidezza e grana è
  essenziale: senza, la varroa è visibilmente più nitida dell'ape.

I due script vecchi non sono stati modificati e restano utilizzabili.
`varroa_on_crops.py` importa da `varroa_generator.py` le funzioni comuni
(caricamento sorgenti, adattamento colore, ombre, compositing).

## File

- `varroa_on_crops.py` — inserimento su un singolo crop (`add_varroa_to_crop`) e
  CLI per prove visive con foglio di confronto.
- `build_varroa_split.py` — costruzione del dataset train/test.
- `tests/test_varroa_split.py` — test su conteggi e divisione.
- Sorgenti varroa: `varroa_alpha.png`, `varroa_alpha2.png`, `varroa_alpha3.png` in
  `/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/` (PNG RGBA).
- Crop di input (default): `.../processed/train_single_bee_batch3+4_clusters/Prova1/Crop_Utilizzabile`
  (54.182 PNG 512×512, asse lungo dell'ape verticale, testa in alto **o** in basso).

## Inserimento su un crop, passo per passo

Funzione: `varroa_on_crops.add_varroa_to_crop(crop, varroa_sources, ratio_range, ..., num_varroa=None)`.
Restituisce `(crop_sintetico, gt_mask, zona)`; i primi due sono `None` se fallisce.

### 1. Sorgenti varroa (`vg.load_varroa_sources`)
Per ogni PNG: alpha binarizzato a 128, eroso per togliere la frangia chiara dello
scontorno, apertura morfologica per eliminare zampe e peli, tenuto solo il contorno
più grande riempito. Ritaglio sul corpo, buchi riempiti con inpainting, corpo
normalizzato a 128 px di larghezza.

### 2. Maschera dell'ape (`bee_mask_from_crop`)
Pixel con canale massimo > 10, componente connessa più grande, buchi riempiti.

### 3. Zona ammessa (`body_zone`)
La varroa può stare solo su torace e addome, fino a subito sotto gli occhi.

1. **Parte spessa**: apertura morfologica della maschera con raggio = 60% dello
   spessore massimo (distance transform). Elimina ali aperte, zampe, antenne.
2. **Asse del corpo**: direzione principale (PCA) della parte spessa.
3. **Estremità escluse**: 22% della lunghezza a ciascun capo. La testa non è
   distinguibile dall'addome, quindi si escludono entrambe le estremità: via gli
   occhi, ma anche la punta dell'addome.
4. **Fascia centrale**: distanza dall'asse ≤ 70% della semilarghezza del corpo.
5. **Ali chiare**: esclusi i pixel con saturazione < 75 e valore > 125 (HSV, dopo
   sfocatura), cioè membrana traslucida e riflessi.

### 4. Dimensione e rotazione
- Larghezza varroa = larghezza dell'ape × U(0.20, 0.24). La larghezza dell'ape è il
  lato corto del rettangolo orientato minimo della maschera **intera** (ali comprese).
- Rotazione uniforme 0–360°.
- Dalla varroa ruotata si ricava la maschera binaria (soglia 128, componente più
  grande) ritagliata stretta: è questa che diventa GT.

### 5. Posizione (`find_position_in_mask`)
Fino a 200 centri candidati presi a caso tra i pixel della zona. Un candidato è
scartato se esce dal frame, se si sovrappone a una varroa già inserita o se la
varroa non cade **al 100%** nella zona. Tra i validi vince quello con punteggio più
alto: `saturazione media + 0.5 × (255 − valore medio)` sotto la varroa, cioè zone
sature e scure.

### 6. Adattamento del colore (`vg.adapt_color_to_region`, spazio LAB)
- luminosità media spostata verso quella dell'ape (peso 0.60);
- contrasto avvicinato a quello dell'ape (0.45);
- crominanza spostata verso quella dell'ape (0.45);
- la varroa eredita luci e ombre locali dell'ape sottostante (0.60).

Le statistiche sono calcolate su una copia del crop in cui lo sfondo nero è
sostituito dal colore medio dell'ape, così il nero non le falsa.

### 7. Nitidezza e grana (`match_sharpness_and_noise` in `varroa_on_crops.py`)
- La nitidezza è il 90° percentile del Laplaciano. Quella di riferimento è misurata
  nell'intorno della varroa (3× il suo rettangolo), solo sui pixel dell'ape.
- Si sfoca la varroa con sigma crescente (da 0.6, passo 0.25) finché scende alla
  nitidezza di riferimento, con tetto `0.12 × sqrt(area varroa)`. Alpha sfocato con
  lo stesso sigma.
- Grana: deviazione stimata dal crop (MAD del residuo ad alta frequenza, tetto 6).
  Il rumore aggiunto viene sfocato con lo stesso sigma e riscalato, perché la grana
  dei crop è "larga" (sono ingrandimenti).

### 8. Ombre e compositing
- Bordo sfumato (feather) con raggio ≈ 8% della varroa.
- Ombra di contatto: anello ≈ 10% della varroa, opacità 0.25.
- Ombra spostata: offset ≈ 8%, sfocatura ≈ 20%, opacità 0.20.
- Alpha compositing diretto sul crop (niente Poisson blending).

### 9. GT mask
OR delle maschere binarie strette di ogni varroa, 512×512, valori 0/255. È la
sagoma netta della varroa: non include sfumatura del bordo né ombre, che però
modificano i pixel dell'ape entro pochi pixel dal contorno.

### Numero di varroa
- `num_varroa=None`: 1 o 2 a caso, si tengono quelle che entrano.
- `num_varroa=N`: esattamente N, con fino a 5 tentativi per varroa (dimensione e
  rotazione nuove a ogni tentativo); se non ci riesce l'inserimento fallisce.

## Costruzione del dataset (`build_varroa_split.py`)

1. Divisione casuale dei crop: 80% train, 20% test.
2. Nel test il 3% delle api riceve la varroa: 2/3 con una, 1/3 con due.
3. Si scorre il test in ordine casuale finché le quote sono piene. Se su un'ape la
   varroa non entra, quell'ape resta normale e si passa alla successiva.
4. Le api normali sono copie identiche dei crop sorgente (o symlink con `--mode symlink`).

Conteggi sui 54.182 crop: train 43.346, test 10.836, di cui 325 con varroa
(217 con una, 108 con due).

Output (default `.../processed/single_bee_batch3+4_varroa_split/`):

```
train/normal/        api normali di train
test/normal/         api normali di test
test/anomalous/      api di test con varroa
test/masks/          <bee_id>_mask.png, 0/255
test/labels.csv      bee_id, label (0/1), num_varroa, image_path, mask_path
split.csv            file, split, label, num_varroa
config.json          parametri e conteggi
```

Riproducibilità: seed 42. Ogni ape ha un seed proprio derivato dal nome del file,
quindi il risultato non dipende dall'ordine di elaborazione.

Comandi:

```
venv/bin/python build_varroa_split.py --dry-run          # solo i conteggi
venv/bin/python build_varroa_split.py                    # dataset completo
venv/bin/python build_varroa_split.py --crops-dir <dir> --output <dir> --limit 2000
venv/bin/python varroa_on_crops.py --name <prova> --num-bees 24   # prova visiva
```

Lo script non scrive in una cartella di output non vuota.

## Parametri principali

In cima a `varroa_on_crops.py`: `VARROA_BEE_RATIO_MIN/MAX` (0.20–0.24),
`BODY_CORE_RATIO` (0.60), `BODY_END_MARGIN` (0.22), `BODY_AXIS_BAND` (0.70),
`WING_MAX_SATURATION` (75), `WING_MIN_VALUE` (125), `MIN_ZONE_COVERAGE` (1.0),
`MAX_BLUR_RATIO` (0.12), `ATTEMPTS_PER_VARROA` (5), `CROPS_DIR`.

In cima a `build_varroa_split.py`: `TEST_FRACTION` (0.20), `ANOMALOUS_FRACTION`
(0.03), `TWO_VARROA_FRACTION` (1/3), `SEED` (42), `OUTPUT_DIR`.

In `varroa_generator.py` (condivisi col metodo vecchio): `BLEND_LUMINANCE`,
`BLEND_CONTRAST`, `BLEND_CHROMA`, `BLEND_SHADING`, `MIN_BLUR_SIGMA`,
`MAX_NOISE_SIGMA`, `CONTACT_SHADOW_OPACITY`, `DROP_SHADOW_OPACITY`.

## Limiti noti

- **Ali brune ripiegate sull'addome** non vengono riconosciute come ali: lì la
  varroa può finire sopra l'ala.
- **Dimensione non costante**: la larghezza dell'ape include le ali, quindi sulle
  api con le ali aperte la varroa esce più grande.
- **Bordino chiaro** attorno alla varroa in alcuni casi, più visibile sulle zone scure.
- **Punta dell'addome esclusa** come effetto collaterale dell'esclusione degli occhi.
- **Solo tre varroa sorgente**: la varietà di aspetto viene da rotazione, scala e
  adattamento del colore.
- **Realismo valutato solo a occhio** su qualche decina di api. Non è stato
  verificato l'effetto sul modello di anomaly detection né fatto un confronto con
  il metodo precedente.

## Stato

- Pipeline testata end-to-end su 3.000 crop (conteggi, gruppi disgiunti, maschere
  binarie con il numero giusto di varroa, normali identiche ai sorgenti,
  riproducibilità).
- Al momento della scrittura il dataset completo **non è ancora stato generato**.
