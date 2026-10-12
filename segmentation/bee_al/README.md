# bee_al — dai cluster alle stime sulle colonie

`bee_al` è l'hub di active learning sui crop di singole api: DINOv2 congelato → memory bank →
k-NN pesato + label propagation → batch di query → etichettatura nell'hub → cluster finali.
Questo documento descrive il flusso che porta dai cluster a stime affidabili della
composizione delle colonie.

```
etichettatura ──► criteri di stop ──► snapshot ──► audit alla cieca ──► stime corrette ──► indicatori di colonia
```

Perché serve: contare le api che il modello assegna a ogni classe ("classify and count") dà
numeri distorti, soprattutto per le classi rare. Se il 5% delle api sane viene scambiato per
malato e le malate vere sono il 2%, il conteggio grezzo dice 5,6%. L'audit misura gli errori
del modello su un campione mai visto e li usa per correggere il conteggio.

## Pipeline a fasi

L'etichettatura si fa in più fasi, ognuna con la propria config, le proprie etichette e i propri modelli:

1. **Fase 1 – qualità della segmentazione.** Su tutte le api si separano quelle segmentate bene da
   quelle segmentate male (`labels.classes` suggerisce `Segmentata Bene` e `Segmentata Male`).
2. **Fase 2 – profilo demografico e anomaly detection.** Si lavora solo sulle api che la fase 1 ha
   tenuto. Si prepara da uno snapshot della fase 1:

```
python scripts/new_stage.py --snapshot fase1_finale --keep "Segmentata Bene" --name fase2
cd active_learning_tool && BEE_AL_CONFIG=config/al_fase2.yaml ../venv/bin/python -u app.py
```

Lo script scrive l'elenco delle api tenute (`results_fase2/subset.json`) e `config/al_fase2.yaml`, con
`paths.subset_file` che punta all'elenco. La fase 1 resta intatta: per tornarci basta avviare l'hub
senza `BEE_AL_CONFIG`. Con `--min-confidence` si lasciano fuori le api assegnate dal modello con poca
sicurezza.

## Etichettatura a griglia

`/al-grid` mostra 12 api alla volta (4 × 3, a sinistra) e i comandi a destra. Ogni colore è un
gruppo con la sua classe: si sceglie un colore, si cliccano le api di quella classe, si passa a un
altro colore e si salva tutto insieme. I tasti 1–6 scelgono il colore, 0 sceglie "Scarta ape". Le api
lasciate senza colore tornano nella griglia successiva. `/al-grid?mode=validation` fa lo stesso sul
validation set.

Il pulsante **Shuffle: propose other bees** (griglia e pagina singola, non in validation né in audit)
sostituisce le api del batch non ancora etichettate con altre, scelte dalla stessa strategia di query e
dallo stesso modello: non parte un round nuovo e le etichette già date restano. Le api tolte non vengono
riproposte fino al round successivo; quelle che un altro annotatore sta guardando restano sue. Va usato
con parsimonia: saltare di proposito le api difficili toglie al modello proprio gli esempi più utili.

## Ispezione dei cluster

`/al-clusters` mostra **un gruppo alla volta**, scelto da un menù a tendina: i cluster assegnati dal
modello, le "non assegnate" e le classi etichettate a mano. Le api sono grandi come nella griglia di
etichettatura, dodici per volta su tre righe:

| Riga | Api |
|---|---|
| 1 | le più simili al centro del gruppo |
| 2 | quelle intorno alla mediana |
| 3 | le meno simili |

La similarità è il coseno tra l'embedding dell'ape e il centro (la media) del gruppo; per le "non
assegnate" è quella con l'esempio etichettato più vicino. *Other bees* passa alle dodici successive: le
seguenti dall'alto, dal basso e accanto alla mediana, senza mai ripetere un'ape. Le correzioni si fanno
con i colori, come nella griglia; tra le etichette date a mano c'è anche *Remove label*.

Due menù scelgono quante api vedere (12, 24, 48 o 96) e in che ordine: le tre fasce (default), dalla
più o dalla meno simile, dalla più o dalla meno sicura per il modello (solo per i cluster predetti) o un
campione casuale. La scelta resta valida cambiando gruppo.

La home dell'hub elenca solo le pagine di active learning; gli strumenti precedenti sono raccolti
sotto il tasto *Old tools*.

## 1. Etichettatura e criteri di stop

Si etichetta come sempre da `/al-labeling`. Per le api su cui si è in dubbio conviene dichiarare la
confidenza *Low*: sono le prime a finire nella revisione delle etichette sospette (vedi sotto). La pagina `/al-status` mostra tre criteri di stop,
valutati sugli ultimi `stopping.window` round che hanno ricevuto etichette nuove:

| Criterio | Significato |
|---|---|
| `f1_plateau` | la macro-F1 di validazione non migliora più di `f1_min_delta` |
| `predictions_stable` | meno di `max_pred_change` delle api cambia classe da un round all'altro |
| `no_new_classes` | nessuna classe nuova da almeno `window` round |

Sono un'indicazione: niente si ferma da solo. Un riavvio dell'hub o un ricalcolo senza
etichette nuove creano un round ma non contano. Dopo un re-train le predizioni non sono
confrontabili con il round precedente, e quel round è escluso dal criterio di stabilità.
Senza etichette nel validation set il primo criterio resta "non ancora noto".

```yaml
stopping:
  window: 3
  max_pred_change: 0.01
  f1_min_delta: 0.01
```

## 2. Freeze dello snapshot

Uno snapshot congela le predizioni di ogni ape. Non cambia più: audit e stime si calcolano
su di lui, anche se nel frattempo l'etichettatura continua.

- Dall'hub: `/al-status` → **Freeze snapshot**.
- Da riga di comando, senza avviare l'hub e senza aggiungere round:

```
python scripts/export_snapshot.py --name 2026-10-05_round42
```

In `<results_dir>/snapshots/<nome>/`:

| File | Contenuto |
|---|---|
| `predictions.csv` | una riga per ape: stato, classe finale, predizioni (propagazione e k-NN), frame, sessione, alveare |
| `frames.csv` | una riga per frame: api grezze estratte, accettate e scartate dalla pulizia, motivi di scarto |
| `meta.json` | round, modello, config completa, conteggi, metriche di validazione, criteri di stop, affidabilità delle mappe dei frame |
| `al_labels.csv`, `validation_set.json` | copie dello stato al momento del freeze |

Stato di un'ape (`status`): `labeled` (etichetta umana), `predicted` (assegnata dal modello),
`rejected` (classe finale di scarto), `unassigned` (troppo lontana da ogni esempio), `unknown`
("Scarta ape"), `validation`, `no_prediction`.

### Da quale frame viene ogni ape

Il collegamento si ricava dal manifest dell'estrazione (`raw_bees/_extraction_manifest.json`):
gli id delle api crescono di 1 frame dopo frame, quindi i conteggi per frame danno gli
intervalli di id. Se la somma dei conteggi non coincide con `next_bee_id` la mappa di quella
cartella viene dichiarata non affidabile e non usata: non si tira a indovinare.

Sessione e alveare si ricavano dal nome del frame con una regex. Senza regex, o se non
corrisponde, la sessione è il nome della cartella di dataset e l'alveare resta vuoto.

```yaml
frames:
  manifests: {}        # {nome cartella: percorso del manifest}; vuoto = <cartella pulita>/../raw_bees/
  frame_pattern: ""    # es. "(?P<colony>hive\\d+)_(?P<session>\\d{8})"
```

## 3. Creazione ed etichettatura dell'audit

Da `/al-status`, nella riga dello snapshot: **Create audit set** (una volta sola per snapshot,
come il validation set). Il campione è estratto tra le api assegnate dal modello e mai
etichettate a mano, diviso in strati:

- classe predetta × fascia di confidenza (`confidence_bins`);
- uno strato per ogni classe di scarto predetta;
- uno strato per le non assegnate;
- se si carica un CSV `image_id,score`, gli strati di `anomaly_split_class` si dividono anche
  per anomaly score.

Da ogni strato si estraggono `per_stratum` api (tutte, se lo strato ne ha meno; `large_stratum`
se ne ha almeno `large_threshold`). Una quota `double_fraction` va etichettata da due
annotatori diversi, per misurare l'accordo.

L'audit si etichetta dal menu **Audit** (`/al-labeling?mode=audit`), alla cieca: nessun
suggerimento, nessuna classe predetta, nessuno strato. Le api sono mescolate.

Regole che il codice garantisce:

- le etichette di audit non entrano mai nella memory bank, nel k-NN, nella propagazione, nel
  re-train né nelle metriche di validazione;
- le api dell'audit attivo non vengono più proposte in modalità normale e non si possono
  modificare dalla pagina dei cluster;
- lo stesso annotatore non riceve due volte la stessa ape.

```yaml
audit:
  per_stratum: 40
  large_stratum: 250
  large_threshold: 5000
  confidence_bins: [0.9]
  double_fraction: 0.1
  anomaly_split_class: ""
  anomaly_quantile: 0.9
  seed: 7
```

## 4. Report e indicatori

- Dall'hub: `/al-status` → **results**, oppure `/al-audit-results?snapshot=NOME`.
- Da riga di comando:

```
python scripts/audit_report.py --snapshot 2026-10-05_round42
python scripts/audit_report.py --snapshot NOME --anomaly-scores scores.csv
```

In `<snapshot>/report/`: `strata.csv`, `prevalence.csv`, `groups.csv`,
`colony_indicators.json`, `colony_indicators.csv`, `agreement.json`, `report.md`.
Il risultato resta in cache finché non arrivano nuove risposte di audit.

### Stime

Per lo strato `h`: `N_h` api nella popolazione, `n_h` risposte di audit (i "Scarta ape" sono
contati a parte), `r_hk` quota di risposte di classe `k`. `H_k` sono le api etichettate a mano
come `k`: contano esatte e non vengono ricorrette.

| Quantità | Formula |
|---|---|
| Precisione di uno strato | `r_h,c` per la classe predetta `c`, con intervallo di Wilson |
| Totale stimato | `Ñ_k = Σ_h N_h · r_hk + H_k` |
| Prevalenza | `p_k = Ñ_k / Σ_{j non di scarto} Ñ_j` |
| Varianza | `Var(Ñ_k) = Σ_h N_h² (1 − n_h/N_h) r_hk (1 − r_hk) / (n_h − 1)` |
| Classe mai vista in uno strato | limite superiore `≈ 3 / n_h` (regola del tre) |
| Per gruppo `g` | `Ñ_g,k = Σ_h N_g,h · r_hk + H_g,k` |

Gli intervalli principali vengono da un bootstrap: a ogni replica si ricampionano le risposte
di audit dentro ogni strato e, per i gruppi, i frame del gruppo.

Cose da sapere quando si leggono i numeri:

- **Le stime per gruppo assumono che gli errori del modello dentro uno strato siano uguali in
  tutti i gruppi.** Se un alveare ha api "strane" che il modello sbaglia più spesso, la stima di
  quel gruppo non lo vede.
- Se uno strato non ha ancora risposte di audit, per quello strato si usa la classe predetta
  così com'è, e il report lo segnala.
- Il bootstrap ricampiona da frequenze leggermente lisciate (una pseudo-osservazione per
  strato). Senza, una classe rara mai vista in uno strato grande avrebbe incertezza zero e
  l'intervallo coprirebbe il valore vero circa il 77% delle volte invece del 95%.
- Per le api doppie vale la risposta del primo annotatore; la seconda serve solo al kappa.
- Con pochi frame per gruppo (due o tre) il ricampionamento dei frame sottostima la variabilità:
  gli intervalli per gruppo e i trend "significant" vanno presi con cautela.
- Le api di validazione, i "Scarta ape" e quelle senza predizione restano fuori dalle stime.

### Indicatori di colonia

Calcolati per gruppo (`colony.group_by`: `frame`, `session`, `colony` o `dataset`).

| Indicatore | Contenuto |
|---|---|
| I1 – profilo demografico | quote corrette, con intervallo, delle `demographic_classes` |
| I2 – segni di malattia | per ogni classe di `disease_classes`: `detected`, `uncertain` o `not_detected` (con limite superiore) |
| I3 – divergenza di Jensen–Shannon | distanza del profilo del gruppo dal profilo di riferimento, in [0, 1] |
| I4 – quota non spiegata | (non assegnate + "Scarta ape") / api del gruppo |
| I5 – trend | differenza tra sessioni consecutive di I1, I2, I3, con intervallo |
| Forza della colonia | api grezze per frame |
| Qualità | quota di scarto stimata e scarti della pulizia automatica |
| Semaforo | verde, giallo o rosso, con l'elenco delle regole scattate |

Un segno di malattia è `detected` quando il limite inferiore dell'intervallo è sopra zero e
almeno `min_detected` api del gruppo sono state confermate da un umano (risposte di audit più
etichette date a mano). La soglia di I3 è media + 2 deviazioni standard della divergenza dei
gruppi di riferimento: con pochi gruppi è poco informativa.

```yaml
colony:
  group_by: session
  demographic_classes: []
  disease_classes: []
  min_detected: 3
  reference_groups: []        # vuoto = media di tutti i gruppi
  time_order: {}              # {alveare: [sessione1, sessione2, ...]}; "*" = tutti gli alveari insieme
  anomaly_threshold: 0.0      # 0 = spento
  n_boot: 1000
  seed: 11
  light_rules: [disease_detected, profile_divergent]   # anche: unexplained_high
  red_min_rules: 2
  max_unexplained: 0.0
```

## Revisione delle etichette sospette

Un'etichetta sbagliata pesa molto: le etichette sono poche, le query per incertezza scelgono proprio
le api ambigue, e nella propagazione le etichette umane sono bloccate, quindi un errore si diffonde alle
api simili senza che il grafo lo corregga. La pagina **Review** (`/al-review`) mostra le etichette che
vale la pena riguardare. Niente viene corretto in automatico.

**Come vengono trovate.** Per ogni ape etichettata a mano si fa votare il k-NN sulle *altre* api
etichettate (l'ape non vota per sé), con gli stessi pesi dell'hub. L'etichetta è sospetta se la classe
più votata è un'altra e raccoglie almeno `review.min_disagreement` del voto (default 80%). È l'idea del
*confident learning* (Northcutt et al., 2021). La coda si ricalcola a ogni ricalcolo delle predizioni;
prima vengono le etichette date con confidenza *Low*, poi *Medium*, poi le altre.

**Perché validation set e audit sono esclusi.** Servono a misurare il modello. Correggerli guardando
l'opinione del modello li avvicinerebbe alle sue predizioni e lo farebbe sembrare migliore di quanto è.
Per loro l'unico controllo ammesso è l'accordo tra annotatori, già previsto nell'audit. Restano fuori
anche le api messe da parte con "Scarta ape".

**Le tre decisioni.** Accanto all'ape si vedono l'etichetta attuale, chi l'ha data e con che
confidenza, la classe suggerita e le api etichettate più simili, che sono l'evidenza.

| Decisione | Effetto |
|---|---|
| Keep (*confirm*) | l'etichetta è giusta: nessun evento nuovo; l'ape non torna in coda finché restano uguali la sua etichetta e la classe suggerita |
| Change (*correct*) | la classe giusta è un'altra, quella suggerita o una a scelta: è una normale correzione in `al_labels.csv`, e le predizioni si ricalcolano |
| Scarta ape (*set aside*) | l'ape non è giudicabile: viene messa da parte |

Ogni decisione è una riga di `review_log.jsonl`, in sola aggiunta; al riavvio le conferme si
ricostruiscono da lì.

**Classi rare.** Una classe con meno di `review.min_class_size` etichette (default 5) ha vicini quasi
tutti di altre classi anche quando l'etichetta è giusta: è proprio come appare un fenotipo raro appena
scoperto. Queste api stanno in una sezione separata, in fondo alla pagina: vanno guardate, ma ci si
aspetta che siano in gran parte corrette. Il segnale diventa utile quando la classe ha più esempi.

**Numeri da riportare.** In `/al-status`, riquadro *Label review* (e in `review_report.csv`, una riga per
decisione): etichette controllate, sospette trovate, quante confermate, corrette e scartate, le
correzioni per coppia di classi (vecchia → nuova), e la quota di sospette per confidenza dichiarata. Se
le *Low* sono sospette molto più spesso delle *High*, la confidenza dichiarata è informativa. Ogni riga
di `al_rounds.jsonl` ha anche `n_suspects`.

```yaml
review:
  enabled: true
  k: 0                    # 0 = knn.k
  min_disagreement: 0.8
  min_class_size: 5
  show_neighbors: 6
```

## Classi di scarto

`labels.reject_classes` elenca le classi dei crop che non si possono giudicare (sfocati,
tagliati male, più api, nessuna ape). Regola per gli annotatori: **se il difetto impedisce di
giudicare il fenotipo è scarto; se il fenotipo si riconosce comunque si etichetta il fenotipo.**

| Dove | Cosa succede alle api di scarto |
|---|---|
| k-NN e propagazione | restano, così il modello impara a riconoscerle |
| Cluster finali | escluse |
| Snapshot | stato `rejected`, sia se scartate a mano sia dal modello |
| Audit | uno strato per classe di scarto predetta: misura quante api valide vengono scartate per errore |
| Stime | stimate come le altre classi, poi escluse dal denominatore delle prevalenze |
| Report | quota di scarto stimata per gruppo, accanto agli scarti della pulizia automatica |
| `/al-status` | conteggio per classe: etichettate a mano e assegnate dal modello |

## File del pacchetto

| File | Ruolo |
|---|---|
| `engine.py` | stato dell'hub: batch, round, criteri di stop, snapshot, audit |
| `frames.py` | collegamento ape → frame dai manifest |
| `snapshot.py` | scrittura e lettura degli snapshot |
| `audit.py` | campionamento stratificato e stime corrette |
| `colony.py` | indicatori di colonia |
| `report.py` | file del report e cache |
| `review.py` | coda delle etichette sospette |
| `export.py` | chiusura di una fase: cluster su disco |
| `hub.py` | pagine Flask |

I test si lanciano con `venv/bin/python -m pytest tests`.

## Fase corrente e chiusura della fase

La sezione `stage` della config dice in che fase si è (`name`, `title`, `description`): l'hub la mostra
in una fascia blu in cima a ogni pagina, così chi etichetta sa che cosa deve giudicare.

A fase finita, in `/al-status` il pulsante **Save configuration & export clusters** congela uno
snapshot e scrive i cluster su disco in `export.clusters_dir/<nome>/`:

| Contenuto | Descrizione |
|---|---|
| una cartella per classe | le immagini delle api di quella classe, etichettate a mano o assegnate dal modello |
| `clusters.csv` | tutte le api: cluster, stato, origine, confidenza, frame, sessione, percorso originale |
| `config.yaml` | la configurazione con cui la fase è stata fatta |
| `meta.json` | fase, data, chi ha esportato, api per cluster, snapshot di partenza |

Le api scartate ("Scarta ape"), quelle non assegnate e quelle di validazione restano solo nel CSV.
`export.mode` sceglie tra copia dei file (`copy`), `hardlink` e `symlink`. L'esportazione gira in
background e non ferma l'etichettatura. Quali cluster portare nella fase successiva si decide dopo,
con `scripts/new_stage.py --keep ...`.

L'hub lavora solo sulle cartelle elencate in `paths.dataset_dirs`: togliere una cartella (per esempio
il test set) esclude le sue api anche se i loro embedding sono già stati estratti.
