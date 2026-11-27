#!/bin/bash
# YOLO ablation on the SAM 3 labels: trains the given configurations one after the other on the
# same GPU and evaluates each one on the test set, at its training resolution and at 2048 px.
# The configurations are those of the first ablation (logs/ablation); only the dataset and max_det change.
#
# Usage: logs/ablation_sam3/queue.sh <gpu> <run> [<run> ...]
#   run: base | img1376 | img1376_rot | img1376_rot_cp   (1376 px: at 1440 the SAM 3 labels do not fit in 48 GB)
# Progress is written to logs/ablation_sam3/queue_gpu<gpu>.out
cd /home/tommaso_ballarin/bee_segmentation/bee_segmentation
GPU=$1; shift
PREFIX=bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3
RUNS=runs/segment/runs/segment
OUT=logs/ablation_sam3
MAX_DET=2500        # as in abl_eval.py: above the maximum number of bees per photo (~1850 with SAM 3)

say() { echo "$(date '+%F %T') $*" >> $OUT/queue_gpu$GPU.out; }

for RUN in "$@"; do
    case $RUN in
        base)           SIZE=1280; ARGS="" ;;
        img1376)        SIZE=1376; ARGS="" ;;
        img1376_rot)    SIZE=1376; ARGS="degrees=180 flipud=0.5" ;;
        img1376_rot_cp) SIZE=1376; ARGS="degrees=180 flipud=0.5 copy_paste=0.3" ;;
        *) say "$RUN: unknown configuration, skipped"; continue ;;
    esac
    NAME=$PREFIX; [ $RUN != base ] && NAME=${PREFIX}_$RUN
    LOG=logs/train_batch3+4+5+6_SAM3_$RUN.log

    # Ultralytics does not overwrite: if the folder already exists it would train into <name>-2
    if [ -e "$RUNS/$NAME" ]; then
        say "$RUN: $RUNS/$NAME already exists, skipped (delete it to redo the training)"
        continue
    fi

    say "$RUN: training (imgsz=$SIZE $ARGS)"
    venv/bin/python train_yolo.py data=data_sam3.yaml device=$GPU imgsz=$SIZE max_det=$MAX_DET $ARGS name=$NAME > $LOG 2>&1
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
say "queue finished: $*"
