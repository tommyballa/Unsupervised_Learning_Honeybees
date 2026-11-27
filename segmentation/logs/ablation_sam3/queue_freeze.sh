#!/bin/bash
# Freeze ablation on the SAM 3 labels: retrains the best configuration from queue.sh with
# different freeze levels (plus a second seed as a measure of the noise), as in the first ablation
# (logs/ablation/queue_gpu*.sh), and evaluates each run on the test set at training resolution and at 2048 px.
#
# Usage: logs/ablation_sam3/queue_freeze.sh <gpu> <config> <variant> [<variant> ...]
#   config:  base | img1376 | img1376_rot | img1376_rot_cp   (the one that turned out best)
#   variant: freeze5 | freeze10 | freeze20 | freeze23 | seed1
# Progress is written to logs/ablation_sam3/queue_gpu<gpu>.out
cd "$(dirname "$0")/../.."   # segmentation/
GPU=$1; CONF=$2; shift 2
PREFIX=bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3
RUNS=runs/segment/runs/segment
OUT=logs/ablation_sam3
MAX_DET=2500        # as in abl_eval.py: above the maximum number of bees per photo (~1850 with SAM 3)

say() { echo "$(date '+%F %T') $*" >> $OUT/queue_gpu$GPU.out; }

case $CONF in
    base)           SIZE=1280; ARGS="" ;;
    img1376)        SIZE=1376; ARGS="" ;;
    img1376_rot)    SIZE=1376; ARGS="degrees=180 flipud=0.5" ;;
    img1376_rot_cp) SIZE=1376; ARGS="degrees=180 flipud=0.5 copy_paste=0.3" ;;
    *) say "$CONF: unknown configuration, freeze queue not started"; exit 1 ;;
esac
BASE=$PREFIX; [ $CONF != base ] && BASE=${PREFIX}_$CONF

for VAR in "$@"; do
    case $VAR in
        freeze*) EXTRA="freeze=${VAR#freeze}" ;;
        seed*)   EXTRA="seed=${VAR#seed}" ;;
        *) say "$VAR: unknown variant, skipped"; continue ;;
    esac
    RUN=${CONF}_$VAR
    NAME=${BASE}_$VAR
    LOG=logs/train_batch3+4+5+6_SAM3_$RUN.log

    # Ultralytics does not overwrite: if the folder already exists it would train into <name>-2
    if [ -e "$RUNS/$NAME" ]; then
        say "$RUN: $RUNS/$NAME already exists, skipped (delete it to redo the training)"
        continue
    fi

    say "$RUN: training (imgsz=$SIZE $ARGS $EXTRA)"
    venv/bin/python train_yolo.py data=data_sam3.yaml device=$GPU imgsz=$SIZE max_det=$MAX_DET $ARGS $EXTRA name=$NAME > $LOG 2>&1
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        say "$RUN: training FAILED (exit $STATUS), see $LOG"
        continue
    fi
    # on the first out of memory Ultralytics halves the batch and restarts without stopping
    grep -q "Reducing to batch" $LOG && say "$RUN: WARNING batch reduced because of out of memory, this is no longer the intended configuration"

    say "$RUN: test at $SIZE and 2048 px"
    venv/bin/python $OUT/abl_eval.py $GPU $OUT/abl_$RUN.json $NAME:$SIZE $NAME:2048 > $OUT/abl_$RUN.log 2>&1
    say "$RUN: done (test exit $?)"
done
say "freeze queue finished: $CONF $*"
