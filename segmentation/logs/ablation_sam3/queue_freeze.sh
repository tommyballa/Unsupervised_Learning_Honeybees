#!/bin/bash
# Ablation sul freeze sulle etichette SAM 3: riallena la configurazione migliore di queue.sh con
# diversi livelli di freeze (piu' un secondo seed come misura del rumore), come nel primo ablation
# (logs/ablation/queue_gpu*.sh), e valuta ogni run sul test set a risoluzione di training e a 2048 px.
#
# Uso: logs/ablation_sam3/queue_freeze.sh <gpu> <config> <variante> [<variante> ...]
#   config:   base | img1376 | img1376_rot | img1376_rot_cp   (quella risultata migliore)
#   variante: freeze5 | freeze10 | freeze20 | freeze23 | seed1
# L'avanzamento finisce in logs/ablation_sam3/queue_gpu<gpu>.out
cd /home/tommaso_ballarin/bee_segmentation/bee_segmentation
GPU=$1; CONF=$2; shift 2
PREFIX=bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3
RUNS=runs/segment/runs/segment
OUT=logs/ablation_sam3
MAX_DET=2500        # come in abl_eval.py: sopra il massimo di api per foto (~1850 con SAM 3)

say() { echo "$(date '+%F %T') $*" >> $OUT/queue_gpu$GPU.out; }

case $CONF in
    base)           SIZE=1280; ARGS="" ;;
    img1376)        SIZE=1376; ARGS="" ;;
    img1376_rot)    SIZE=1376; ARGS="degrees=180 flipud=0.5" ;;
    img1376_rot_cp) SIZE=1376; ARGS="degrees=180 flipud=0.5 copy_paste=0.3" ;;
    *) say "$CONF: configurazione sconosciuta, coda freeze non partita"; exit 1 ;;
esac
BASE=$PREFIX; [ $CONF != base ] && BASE=${PREFIX}_$CONF

for VAR in "$@"; do
    case $VAR in
        freeze*) EXTRA="freeze=${VAR#freeze}" ;;
        seed*)   EXTRA="seed=${VAR#seed}" ;;
        *) say "$VAR: variante sconosciuta, saltata"; continue ;;
    esac
    RUN=${CONF}_$VAR
    NAME=${BASE}_$VAR
    LOG=logs/train_batch3+4+5+6_SAM3_$RUN.log

    # Ultralytics non sovrascrive: con la cartella gia' presente allenerebbe in <nome>-2
    if [ -e "$RUNS/$NAME" ]; then
        say "$RUN: $RUNS/$NAME esiste gia', saltata (cancellala per rifare il training)"
        continue
    fi

    say "$RUN: training (imgsz=$SIZE $ARGS $EXTRA)"
    venv/bin/python train_yolo.py data=data_sam3.yaml device=$GPU imgsz=$SIZE max_det=$MAX_DET $ARGS $EXTRA name=$NAME > $LOG 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        say "$RUN: training FALLITO (exit $STATUS), vedi $LOG"
        continue
    fi
    # al primo out of memory Ultralytics dimezza il batch e riparte senza fermarsi
    grep -q "Reducing to batch" $LOG && say "$RUN: ATTENZIONE batch ridotto per out of memory, non e' piu' la configurazione prevista"

    say "$RUN: test a $SIZE e 2048 px"
    venv/bin/python $OUT/abl_eval.py $GPU $OUT/abl_$RUN.json $NAME:$SIZE $NAME:2048 > $OUT/abl_$RUN.log 2>&1
    say "$RUN: finita (exit del test $?)"
done
say "coda freeze finita: $CONF $*"
