#!/bin/bash
# Start the four guided GR00T N1.7 policy servers used by the benchmark (one checkpoint x guidance mode per GPU):
#   gpu0 :5561 libero_spatial g1     gpu1 :5562 libero_spatial none
#   gpu2 :5563 libero_object  g1     gpu3 :5564 libero_object  none
# Baseline rollouts (eval_libero.py --policy groot) send no cost context, so they are unguided on either server.
# Usage: bash groot_servers.sh [start|stop|status]
source /workspace/mnt/mywang87/0Xuehui/oc_guidance_libero/env.sh
export HF_HOME=$WS/.cache/huggingface HF_ENDPOINT=https://hf-mirror.com
GR=$WS/0Xuehui/Isaac-GR00T
CKR=$GR/checkpoints/GR00T-N1.7-LIBERO
mkdir -p $PROJ/runs/groot_servers
declare -a SPEC=("0 5561 libero_spatial_pubbb g1" "1 5562 libero_spatial_pubbb none" "2 5563 libero_object_pubbb g1" "3 5564 libero_object_pubbb none")
case ${1:-start} in
  start)
    cd $GR
    for s in "${SPEC[@]}"; do set -- $s; gpu=$1; port=$2; ck=$3; mode=$4
      if pgrep -f "groot_serve[r].py.*--port $port" > /dev/null; then echo "port $port already running"; continue; fi
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PROJ setsid nohup .venv/bin/python -u $PROJ/safeguide/server/groot_server.py \
          --model-path $CKR/$ck --port $port --guidance $mode $([ $mode = g1 ] && echo "--recovery 1") > $PROJ/runs/groot_servers/${ck}_${mode}_$port.log 2>&1 < /dev/null &
      echo "started gpu$gpu :$port $ck $mode"
    done
    for i in $(seq 1 90); do
      n=$(grep -l "listening on" $PROJ/runs/groot_servers/*.log 2>/dev/null | wc -l); [ "$n" -ge 4 ] && break; sleep 5
    done
    grep -H "listening on\|Traceback" $PROJ/runs/groot_servers/*.log | cut -c1-160 ;;
  stop) pkill -f "groot_serve[r].py"; sleep 2; echo "remaining: $(pgrep -fc 'groot_serve[r].py')" ;;
  status) ps -eo pid,etime,args | grep "groot_serve[r].py" | grep -oE "^ *[0-9]+ +[0-9:-]+|--port [0-9]+|libero_[a-z]+_pubbb|--guidance [a-z0-9]+" | paste - - - - ;;
esac
