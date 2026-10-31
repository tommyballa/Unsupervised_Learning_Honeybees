#!/bin/bash
# Coda del freeze che parte solo quando una GPU ha abbastanza memoria libera, e riprende dall'ultimo
# checkpoint i run interrotti. Serve perche' il server e' condiviso: con 12-17 GB di altri utenti sulla GPU
# questi training (34-41 GB di picco) vanno in out of memory a meta'.
# Sostituisce queue_freeze.sh e queue_freeze_resume.sh; stesse configurazioni e stesso test.
#
# Per ogni variante, nell'ordine:
#   - risultati di test gia' presenti              -> saltata
#   - run interrotto dopo almeno MIN_EPOCHS epoche -> ripresa da weights/last.pt
#   - run interrotto prima (o senza checkpoint)    -> cartella rinominata *__interrotto_*, training da capo
#   - nessuna cartella                             -> training da capo
# Prima di ogni lancio aspetta una GPU con al massimo MAX_USED MiB occupati.
#
# Uso: queue_freeze_gated.sh <config> <variante> [<variante> ...]
B=/mnt/disk1/tommaso_ballarin/bee_ablation_sam3_freeze
cd /home/tommaso_ballarin/bee_segmentation/bee_segmentation
export TMPDIR=/dev/shm/tb_tmp
export ABL_RUNS=$B/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # come in train_yolo.py
CONF=$1; shift
PREFIX=bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3
MAX_DET=2500        # come in abl_eval.py: sopra il massimo di api per foto (~1850 con SAM 3)
GPUS=${GPUS:-"0 1 2"}
MAX_USED=${MAX_USED:-6000}       # MiB degli altri tollerati: con ~4 GB di altri freeze10 (39 GB) ha girato senza problemi
MIN_EPOCHS=10
MAX_ERRORS=3        # fallimenti che non sono out of memory prima di rinunciare a un run

say() { echo "$(date '+%F %T') $*" >> $B/queue_gpu0.out; }
used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

# stampa l'indice della prima GPU rimasta sotto MAX_USED per tre letture a un minuto di distanza
wait_gpu() {
    local told=0 g u ok i
    while true; do
        for g in $GPUS; do
            ok=1
            for i in 1 2 3; do
                u=$(used $g)
                if [ -z "$u" ] || [ "$u" -gt $MAX_USED ]; then ok=0; break; fi
                [ $i -lt 3 ] && sleep 60
            done
            if [ $ok -eq 1 ]; then echo $g; return; fi
        done
        if [ $told -eq 0 ]; then say "$1: in attesa di una GPU con meno di $MAX_USED MiB occupati"; told=1; fi
        sleep 120
    done
}

case $CONF in
    base)           SIZE=1280; ARGS="" ;;
    img1376)        SIZE=1376; ARGS="" ;;
    img1376_rot)    SIZE=1376; ARGS="degrees=180 flipud=0.5" ;;
    img1376_rot_cp) SIZE=1376; ARGS="degrees=180 flipud=0.5 copy_paste=0.3" ;;
    *) say "$CONF: configurazione sconosciuta, coda non partita"; exit 1 ;;
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
    DIR=$ABL_RUNS/$NAME
    LOG=$B/logs/train_batch3+4+5+6_SAM3_$RUN.log
    [ -e $B/abl_$RUN.json ] && continue

    ERRORS=0; DONE=0
    while [ $DONE -eq 0 ]; do
        EPOCHS=0; [ -e $DIR/results.csv ] && EPOCHS=$(( $(wc -l < $DIR/results.csv) - 1 ))
        if [ -e "$DIR" ] && { [ ! -e $DIR/weights/last.pt ] || [ $EPOCHS -lt $MIN_EPOCHS ]; }; then
            TS=$(date +%m%d_%H%M%S)
            mv "$DIR" "${DIR}__interrotto_$TS"
            [ -e "$LOG" ] && mv "$LOG" "${LOG%.log}__interrotto_$TS.log"
            EPOCHS=0
        fi
        GPU=$(wait_gpu $RUN)
        if [ -e $DIR/weights/last.pt ]; then
            say "$RUN: ripresa dall'epoca $EPOCHS sulla GPU $GPU"
            venv/bin/python -c "import sys; from ultralytics import YOLO; YOLO(sys.argv[1]).train(resume=True, device=int(sys.argv[2]))" $DIR/weights/last.pt $GPU >> $LOG 2>&1
        else
            say "$RUN: training sulla GPU $GPU (imgsz=$SIZE $ARGS $EXTRA)"
            venv/bin/python train_yolo.py data=data_sam3.yaml device=$GPU imgsz=$SIZE max_det=$MAX_DET $ARGS $EXTRA project=$ABL_RUNS name=$NAME >> $LOG 2>&1
        fi
        STATUS=$?
        if [ $STATUS -eq 0 ]; then DONE=1; break; fi
        # training gia' arrivato in fondo: resta solo il test
        if tail -c 3000 $LOG | grep -q "nothing to resume"; then DONE=1; break; fi
        if tail -c 6000 $LOG | grep -q "OutOfMemoryError"; then
            say "$RUN: interrotto per out of memory (GPU presa da altri), riparte quando c'e' memoria"
        else
            ERRORS=$((ERRORS + 1))
            say "$RUN: training interrotto (exit $STATUS, errore $ERRORS di $MAX_ERRORS), vedi $LOG"
            if [ $ERRORS -ge $MAX_ERRORS ]; then DONE=2; fi
            sleep 60
        fi
    done
    if [ $DONE -ne 1 ]; then
        say "$RUN: training FALLITO, vedi $LOG"
        continue
    fi
    # al primo out of memory Ultralytics dimezza il batch e riparte senza fermarsi
    grep -q "Reducing to batch" $LOG && say "$RUN: ATTENZIONE batch ridotto per out of memory, non e' piu' la configurazione prevista"

    GPU=$(wait_gpu $RUN)
    say "$RUN: test a $SIZE e 2048 px sulla GPU $GPU"
    venv/bin/python $B/abl_eval.py $GPU $B/abl_$RUN.json $NAME:$SIZE $NAME:2048 > $B/abl_$RUN.log 2>&1
    say "$RUN: finita (exit del test $?)"
done
say "coda freeze finita: $CONF $*"
