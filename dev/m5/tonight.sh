#!/bin/zsh
# splash-m5 overnight evaluation (2026-09-26). Stock = vanilla Splash 1.1.0 (the fork's
# base); fork = this build with the v8 choices. Test servers on 8041/8042 only; the live
# Swift service (8029) is never touched. Everything lands in $OUT; one log line per phase.
#
#   A  steady-state serving (dev/m5/serve_bench.py): official Qwen3.8-27B and Swift-1.5,
#      C = 1-4, 4,096 tokens, sampled, two rounds alternating stock/fork, macmon power and
#      temperature; the fork server writes the acceptance log (draft-length study).
#   B  quality: the 95-task set (converter/bench_quality.py --task-set all) on the official
#      27B, fork then stock.
#   C  Splash's own ABBA harness (dev/benchmarks/http_regression.py), official 27B, stock
#      baseline vs fork candidate, contexts 2,048 and 32,768 (decode ms/token and TTFT).
set -u
FORK=$HOME/Models/splash/splash-m5
STOCK=$HOME/Models/splash/splash-1.1.0
PROJECT=$HOME/Models/splash/swift-splash-project
OUT=${OUT:-$FORK/build/m5/night-2026-09-26}
PY=/opt/homebrew/opt/splash/libexec/python/bin/python3
CHOICES=$FORK/tuning/m5max-40c-swift15-v8.choices
OFFICIAL=$(ls -d $HOME/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-Splash/snapshots/* | head -1)
SWIFT=$PROJECT/output/swift15-splash
export SPLASH_API_KEY=$(cat ~/.splash/api-key)
mkdir -p $OUT
log() { print -r -- "$(date +%H:%M:%S) $*" | tee -a $OUT/night.log; }

start() {  # start LABEL PORT PACKAGE MODEL_ID [ENV...]
  local label=$1 port=$2 pkg=$3 model=$4; shift 4
  local server=$FORK/server/server.py binary=$FORK/build/splash
  [ $label = stock ] && binary=$STOCK/build/splash && server=$STOCK/server/server.py
  env "$@" nohup $PY -u $server $pkg/target $pkg/draft --tokenizer $pkg/tokenizer --model $model \
    --binary $binary --max-memory 30G --max-context auto --port $port > $OUT/server-$label-$port.log 2>&1 &
  local i=0
  until nc -z 127.0.0.1 $port 2>/dev/null || [ $i -gt 200 ]; do sleep 3; i=$((i+1)); done
  nc -z 127.0.0.1 $port 2>/dev/null && log "up $label :$port $model" || log "FAILED to start $label :$port"
}
stop() { for p in "$@"; do for pid in $(pgrep -f "port $p"); do kill $pid; done; done; sleep 8; }

phase_a() {  # phase_a NAME PACKAGE MODEL_ID
  local name=$1 pkg=$2 model=$3
  log "A $name: start"
  start stock 8041 $pkg $model
  start fork 8042 $pkg $model SPLASH_KERNEL_CHOICES=$CHOICES SPLASH_M5_ACCEPT_LOG=$OUT/accept-$name.log
  for r in 1 2; do
    order=(stock:8041 fork:8042); [ $r = 2 ] && order=(fork:8042 stock:8041)
    for e in $order; do
      python3 $FORK/dev/m5/serve_bench.py --label ${e%%:*} --port ${e#*:} --model $model \
        --concurrency 1,2,3,4 --round $r --out $OUT/steady-$name.jsonl 2>&1 | tee -a $OUT/night.log
    done
  done
  python3 $FORK/dev/m5/accept_hist.py $OUT/accept-$name.log > $OUT/accept-$name.txt 2>&1
  stop 8041 8042
  log "A $name: done"
}

log "night start (waits for nothing; run at 22:00 by the caller)"
phase_a official $OFFICIAL incoai/Qwen3.8-27B-Splash
phase_a swift $SWIFT local/Swift-1.5-4bit-MLX-Splash

log "B quality: start"
for e in fork:8042 stock:8041; do
  label=${e%%:*}; port=${e#*:}
  if [ $label = fork ]; then start fork $port $OFFICIAL incoai/Qwen3.8-27B-Splash SPLASH_KERNEL_CHOICES=$CHOICES
  else start stock $port $OFFICIAL incoai/Qwen3.8-27B-Splash; fi
  (cd $PROJECT && .venv/bin/python -m converter.bench_quality --base-url http://127.0.0.1:$port/v1 \
     --model incoai/Qwen3.8-27B-Splash --api-key-file ~/.splash/api-key --task-set all \
     --label official-$label --out $OUT/quality-official-$label.json) > $OUT/quality-official-$label.txt 2>&1
  log "B quality $label: $(tail -2 $OUT/quality-official-$label.txt | tr '\n' ' ')"
  stop $port
done

log "C harness: start"
(cd $FORK && SPLASH_M5_ALLOW_TRANSCRIPT_DIFF=1 SPLASH_KERNEL_CHOICES=$CHOICES .venv/bin/python -m dev.benchmarks.http_regression \
   --baseline-binary $STOCK/build/splash --binary $FORK/build/splash --package $OFFICIAL \
   --model incoai/Qwen3.8-27B-Splash --max-memory 30G --contexts 2048,32768 --samples 3 \
   --output $OUT/http-regression.json) > $OUT/http-regression.txt 2>&1
log "C harness: exit $? $(tail -3 $OUT/http-regression.txt | tr '\n' ' ')"
log "night done"
