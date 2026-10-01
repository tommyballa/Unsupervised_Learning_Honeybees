#!/bin/bash
cd /home/tommaso_ballarin/moviad_acquario/tesi_v2
while pgrep -f "tag r1_r18" >/dev/null || pgrep -f "tag dbg_masked" > /dev/null; do sleep 20; done
exec logs/fastflow_sweep/queue.sh logs/fastflow_sweep/lane2.txt
