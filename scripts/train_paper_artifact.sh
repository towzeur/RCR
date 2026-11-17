#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/train_paper_artifact.sh RUN_NAME [--data-root PATH] [--output-root PATH] [--python PYTHON] [--epochs N] [--stop-after N] [--dry-run]

RUN_NAME must match one row in configs/paper_runs.csv, for example:
  seq_cifar100_buffer200_seed0

Trains ProNC and saves a checkpoint and feature export after each task.
Use scripts/evaluate_artifacts.py to compute ETF and RCR results.

The optional --epochs and --stop-after flags are intended for smoke tests only.
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_NAME="$1"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/paper_runs.csv"
DATA_ROOT="./data"
OUTPUT_ROOT="outputs/paper_runs"
PYTHON_BIN="${PYTHON:-python}"
DEVICE="${CUDA_DEVICE:-0}"
DRY_RUN=0
EPOCHS_OVERRIDE=""
STOP_AFTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --stop-after)
      STOP_AFTER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 2
fi

ROW="$(
  "$PYTHON_BIN" - "$CONFIG" "$RUN_NAME" <<'PY'
import csv
import sys
path, run = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["run"] == run:
            print("\t".join(row[k] for k in [
                "dataset", "buffer", "seed", "epochs", "lr", "main_weight",
                "distill_weight", "ldcrad_start_task", "lr_scheduler",
                "lr_milestones", "ldcrad_score_mix", "ldcrad_mix_head",
                "ldcrad_mix_norm", "ldcrad_mix_rule", "ldcrad_gate_temp",
                "source_run",
            ]))
            break
    else:
        raise SystemExit(f"unknown run: {run}")
PY
)"

IFS=$'\t' read -r DATASET BUFFER SEED EPOCHS LR MAIN_WEIGHT DISTILL_WEIGHT \
  LDCRAD_START_TASK LR_SCHEDULER LR_MILESTONES LDCRAD_SCORE_MIX \
  LDCRAD_MIX_HEAD LDCRAD_MIX_NORM LDCRAD_MIX_RULE LDCRAD_GATE_TEMP SOURCE_RUN <<< "$ROW"

CONFIG_EPOCHS="$EPOCHS"
if [[ -n "$EPOCHS_OVERRIDE" ]]; then
  EPOCHS="$EPOCHS_OVERRIDE"
fi

DATA_ROOT="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$DATA_ROOT")"
OUTPUT_ROOT="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_ROOT")"
RUN_DIR="$OUTPUT_ROOT/${RUN_NAME}"
mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/results"

{
  echo "run=$RUN_NAME"
  echo "source_run=$SOURCE_RUN"
  echo "dataset=$DATASET"
  echo "epochs=$EPOCHS"
  echo "config_epochs=$CONFIG_EPOCHS"
  echo "lr=$LR"
  echo "buffer=$BUFFER"
  echo "main_weight=$MAIN_WEIGHT"
  echo "distill_weight=$DISTILL_WEIGHT"
  echo "seed=$SEED"
  echo "ldcrad_start_task=$LDCRAD_START_TASK"
  echo "ldcrad_score_mix=$LDCRAD_SCORE_MIX"
  echo "ldcrad_mode=task"
  echo "ldcrad_shrink=0.75"
  echo "ldcrad_radius_transform=additive"
  echo "ldcrad_power=0.25"
  echo "ldcrad_logterm=0.25"
  echo "ldcrad_mix_head=$LDCRAD_MIX_HEAD"
  echo "ldcrad_mix_norm=$LDCRAD_MIX_NORM"
  echo "ldcrad_mix_rule=$LDCRAD_MIX_RULE"
  echo "ldcrad_gate_conf=max"
  echo "ldcrad_gate_temp=$LDCRAD_GATE_TEMP"
  echo "lr_scheduler=$LR_SCHEDULER"
  echo "lr_milestones=$LR_MILESTONES"
  echo "data_root=$DATA_ROOT"
  echo "device=$DEVICE"
  if [[ -n "$STOP_AFTER" ]]; then
    echo "stop_after=$STOP_AFTER"
  fi
} > "$RUN_DIR/command.txt"

MAIN_CMD=(
  "$PYTHON_BIN" utils/main.py
  --backbone resnet18
  --model exorth_ce_svd_ldcrad
  --dataset "$DATASET"
  --n_epochs "$EPOCHS"
  --lr "$LR"
  --batch_size 32
  --buffer_size "$BUFFER"
  --device "$DEVICE"
  --num_workers 2
  --optim_mom 0.
  --optim_wd 0.0000
  --lr_scheduler "$LR_SCHEDULER"
  --lr_milestones $LR_MILESTONES
  --main_weight "$MAIN_WEIGHT"
  --distill_weight "$DISTILL_WEIGHT"
  --ce_weight 1
  --seed "$SEED"
  --base_path "$DATA_ROOT"
  --results_path "$RUN_DIR/results"
  --report_task_routing 1
  --ldc_mode linear
  --ldc_fit_source mixed
  --ldc_ridge 0.001
  --ldc_transport_shrink 1.0
  --ldc_score_mix 1.0
  --ldc_buffer_blend 0.0
  --ldcrad_mode task
  --ldcrad_start_task "$LDCRAD_START_TASK"
  --ldcrad_shrink 0.75
  --ldcrad_radius_transform additive
  --ldcrad_power 0.25
  --ldcrad_logterm 0.25
  --ldcrad_score_mix "$LDCRAD_SCORE_MIX"
  --ldcrad_mix_head "$LDCRAD_MIX_HEAD"
  --ldcrad_mix_norm "$LDCRAD_MIX_NORM"
  --ldcrad_mix_rule "$LDCRAD_MIX_RULE"
  --ldcrad_gate_conf max
  --ldcrad_gate_temp "$LDCRAD_GATE_TEMP"
  --ldcrad_artifact_path "$RUN_DIR/artifacts/artifact.pt"
  --savecheck task
  --ckpt_name "$RUN_NAME"
)

if [[ -n "$STOP_AFTER" ]]; then
  MAIN_CMD+=(--stop_after "$STOP_AFTER")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  cat "$RUN_DIR/command.txt"
  printf '\ncd %q\n' "$ROOT/rcr"
  printf '%q ' "${MAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$ROOT/rcr"
exec > "$RUN_DIR/log.txt" 2>&1

echo "[run] $(date -Iseconds)"
cat "$RUN_DIR/command.txt"

"${MAIN_CMD[@]}"

echo "[done] $(date -Iseconds)"
