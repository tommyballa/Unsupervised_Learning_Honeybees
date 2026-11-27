#!/bin/bash
# Freeze queue that starts only when a GPU has enough free memory, and resumes interrupted runs from
# their last checkpoint. Needed because the server is shared: with 12-17 GB from other users on the GPU
# these trainings (34-41 GB peak) run out of memory halfway through.
# Replaces queue_freeze.sh and queue_freeze_resume.sh; same configurations and same test.
#
# For each variant, in order:
#   - test results already present                     -> skipped
#   - run interrupted after at least MIN_EPOCHS epochs -> resumed from weights/last.pt
#   - run interrupted earlier (or without checkpoint)  -> folder renamed *__interrotto_*, training from scratch
#   - no folder                                        -> training from scratch
# Before each launch it waits for a GPU with at most MAX_USED MiB in use.
#
# Usage: queue_freeze_gated.sh <config> <variant> [<variant> ...]
B=/mnt/disk1/tommaso_ballarin/bee_ablation_sam3_freeze
cd /home/tommaso_ballarin/bee_segmentation/bee_segmentation
export TMPDIR=/dev/shm/tb_tmp
export ABL_RUNS=$B/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # as in train_yolo.py
CONF=$1; shift
PREFIX=bee_model_finetuned_yolo26s_batch3+4+5+6_SAM3
MAX_DET=2500        # as in abl_eval.py: above the maximum number of bees per photo (~1850 with SAM 3)
GPUS=${GPUS:-"0 1 2"}
MAX_USED=${MAX_USED:-6000}       # MiB used by others that we tolerate: with ~4 GB from others, freeze10 (39 GB) ran without problems
MIN_EPOCHS=10
MAX_ERRORS=3        # non-out-of-memory failures before giving up on a run

say() { echo "$(date '+%F %T') $*" >> $B/queue_gpu0.out; }
used() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

# prints the index of the first GPU that stayed below MAX_USED for three readings one minute apart
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
        if [ $told -eq 0 ]; then say "$1: waiting for a GPU with less than $MAX_USED MiB in use"; told=1; fi
        sleep 120
    done
}

case $CONF in
    base)           SIZE=1280; ARGS="" ;;
    img1376)        SIZE=1376; ARGS="" ;;
    img1376_rot)    SIZE=1376; ARGS="degrees=180 flipud=0.5" ;;
    img1376_rot_cp) SIZE=1376; ARGS="degrees=180 flipud=0.5 copy_paste=0.3" ;;
    *) say "$CONF: unknown configuration, queue not started"; exit 1 ;;
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
            say "$RUN: resuming from epoch $EPOCHS on GPU $GPU"
            venv/bin/python -c "import sys; from ultralytics import YOLO; YOLO(sys.argv[1]).train(resume=True, device=int(sys.argv[2]))" $DIR/weights/last.pt $GPU >> $LOG 2>&1
        else
            say "$RUN: training on GPU $GPU (imgsz=$SIZE $ARGS $EXTRA)"
            venv/bin/python train_yolo.py data=data_sam3.yaml device=$GPU imgsz=$SIZE max_det=$MAX_DET $ARGS $EXTRA project=$ABL_RUNS name=$NAME >> $LOG 2>&1
        fi
        STATUS=$?
        if [ $STATUS -eq 0 ]; then DONE=1; break; fi
        # training already reached the end: only the test is left
        if tail -c 3000 $LOG | grep -q "nothing to resume"; then DONE=1; break; fi
        if tail -c 6000 $LOG | grep -q "OutOfMemoryError"; then
            say "$RUN: interrupted by out of memory (GPU taken by others), will restart when memory is available"
        else
            ERRORS=$((ERRORS + 1))
            say "$RUN: training interrupted (exit $STATUS, error $ERRORS of $MAX_ERRORS), see $LOG"
            if [ $ERRORS -ge $MAX_ERRORS ]; then DONE=2; fi
            sleep 60
        fi
    done
    if [ $DONE -ne 1 ]; then
        say "$RUN: training FAILED, see $LOG"
        continue
    fi
    # on the first out of memory Ultralytics halves the batch and restarts without stopping
    grep -q "Reducing to batch" $LOG && say "$RUN: WARNING batch reduced because of out of memory, this is no longer the intended configuration"

    GPU=$(wait_gpu $RUN)
    say "$RUN: test at $SIZE and 2048 px on GPU $GPU"
    venv/bin/python $B/abl_eval.py $GPU $B/abl_$RUN.json $NAME:$SIZE $NAME:2048 > $B/abl_$RUN.log 2>&1
    say "$RUN: done (test exit $?)"
done
say "freeze queue finished: $CONF $*"
