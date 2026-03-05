#!/bin/bash
# Run data profiler on the 26 competitions from profiler dir
# Saves results to profiler_v2/

DATA_ROOT="/data/yiming/mlebench-data"
OUTPUT_ROOT="/data/yiming/project/data_analysis/profiler_v2"
PROFILER="/data/yiming/project/data-profiler/main.py"
MAX_PARALLEL=16
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"

# The 26 competitions from the original profiler run
COMPETITIONS=(
  aptos2019-blindness-detection
  denoising-dirty-documents
  detecting-insults-in-social-commentary
  dog-breed-identification
  google-quest-challenge
  jigsaw-toxic-comment-classification-challenge
  leaf-classification
  learning-agency-lab-automated-essay-scoring-2
  lmsys-chatbot-arena
  mlsp-2013-birds
  new-york-city-taxi-fare-prediction
  nomad2018-predict-transparent-conductors
  petfinder-pawpularity-score
  plant-pathology-2020-fgvc7
  predict-volcanic-eruptions-ingv-oe
  random-acts-of-pizza
  spooky-author-identification
  stanford-covid-vaccine
  statoil-iceberg-classifier-challenge
  tabular-playground-series-dec-2021
  tensorflow-speech-recognition-challenge
  tgs-salt-identification-challenge
  the-icml-2013-whale-challenge-right-whale-redux
  tweet-sentiment-extraction
  us-patent-phrase-to-phrase-matching
  ventilator-pressure-prediction
)

# Skip already completed
TODO=()
for comp in "${COMPETITIONS[@]}"; do
  if [ -f "$OUTPUT_ROOT/$comp/mle_report.md" ]; then
    echo "[SKIP] $comp (already done)"
  else
    TODO+=("$comp")
  fi
done

total=${#TODO[@]}
echo "============================================================"
echo "Batch Data Profiler Run (v2)"
echo "============================================================"
echo "Total: ${#COMPETITIONS[@]}, To run: $total, Skipped: $((${#COMPETITIONS[@]} - total))"
echo "Output: $OUTPUT_ROOT"
echo "Max parallel: $MAX_PARALLEL"
echo "============================================================"

running=0
for comp in "${TODO[@]}"; do
  data_dir="$DATA_ROOT/$comp/prepared/public"
  out_dir="$OUTPUT_ROOT/$comp"
  log_file="$LOG_DIR/$comp.log"

  echo "[START] $comp"
  (
    python "$PROFILER" "$data_dir" -o "$out_dir" -v \
      --max-turns-per-layer 3 \
      --stop-threshold 0.80 \
      --layer-timeout-sec 300 \
      > "$log_file" 2>&1
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
      echo "[DONE]  $comp (success)"
    else
      echo "[FAIL]  $comp (exit code $exit_code)"
    fi
  ) &

  running=$((running + 1))
  if [ $running -ge $MAX_PARALLEL ]; then
    wait -n 2>/dev/null || { for pid in $(jobs -p); do wait "$pid" 2>/dev/null && break; done; }
    running=$((running - 1))
  fi
done

wait

echo ""
echo "============================================================"
echo "Batch Run Complete"
echo "============================================================"

success=0
fail=0
for comp in "${COMPETITIONS[@]}"; do
  if [ -f "$OUTPUT_ROOT/$comp/mle_report.md" ]; then
    success=$((success + 1))
  else
    fail=$((fail + 1))
    echo "  FAILED: $comp"
  fi
done

echo "Success: $success / ${#COMPETITIONS[@]}"
echo "Failed:  $fail / ${#COMPETITIONS[@]}"
echo "Logs:    $LOG_DIR/"
