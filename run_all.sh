#!/bin/bash
# Run data profiler on all mlebench competitions
# Saves results to output_v2/<competition_name>/
# Runs up to N jobs in parallel

DATA_ROOT="/data/yiming/mlebench-data"
OUTPUT_ROOT="/data/yiming/project/data-profiler/output_v2"
PROFILER="/data/yiming/project/data-profiler/main.py"
MAX_PARALLEL=16
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"

# All competitions with prepared/public data, skip already completed
COMPETITIONS=()
for comp in $(ls "$DATA_ROOT"); do
  if [ -d "$DATA_ROOT/$comp/prepared/public" ]; then
    if [ -f "$OUTPUT_ROOT/$comp/mle_report.md" ]; then
      echo "[SKIP] $comp (already done)"
      continue
    fi
    COMPETITIONS+=("$comp")
  fi
done

total=${#COMPETITIONS[@]}

echo "============================================================"
echo "Batch Data Profiler Run"
echo "============================================================"
echo "Competitions: $total"
echo "Output: $OUTPUT_ROOT"
echo "Max parallel: $MAX_PARALLEL"
echo "============================================================"

# Simple parallel execution with job control
running=0
pids=()
names=()

for comp in "${COMPETITIONS[@]}"; do
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

  pids+=($!)
  names+=("$comp")
  running=$((running + 1))

  # Wait for one to finish if at max parallel
  if [ $running -ge $MAX_PARALLEL ]; then
    wait -n 2>/dev/null || { for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null && break; done; }
    running=$((running - 1))
  fi
done

# Wait for all remaining jobs
wait

echo ""
echo "============================================================"
echo "Batch Run Complete"
echo "============================================================"

# Summary
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

echo "Success: $success / $total"
echo "Failed:  $fail / $total"
echo "Logs:    $LOG_DIR/"
