#!/usr/bin/env bash
set -eo pipefail

# ===== Configuration =====
GPUS=(0 1 2 3 4 5)
BATCH_SIZE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${SCRIPT_DIR}/analysis.py"
BASE_DIR="${SCRIPT_DIR}"
CACHE_FOLDER="${HF_CACHE_DIR:-./.cache/huggingface}"

BASE_OUT="${SCRIPT_DIR}/ALL_ANALYSIS_OUTPUT"

# Bypass proxy
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

DIRS=(
  "arc_hetero"
  "arc_hetero_no_persona"
  "arc_homo_ablation"
  "arc_single_agent"
  "formal_logic_single_agent"
  "hh_rlhf_hetero_no_persona2"
  "truthfulqa_hetero_no_persona"
  "gsm8k_hetero"
  "truthfulqa_homo_ablation"
  "gsm8k_hetero_no_persona"
  "truthfulqa_single_agent"
  "gsm8k_homo_ablation"
  "winogrande_hetero"
  "gsm8k_single_agent"
  "winogrande_hetero2"
  "hellaswag_hetero"
  "pro_medicine_hetero"
  "winogrande_hetero_no_persona"
  "hellaswag_hetero_no_persona"
  "pro_medicine_hetero_no_persona"
  "winogrande_hetero_no_persona2"
  "formal_logic_hetero"
  "hellaswag_homo_ablation"
  "pro_medicine_homo_ablation"
  "winogrande_homo_ablation"
  "formal_logic_hetero_4models"
  "hellaswag_single_agent"
  "pro_medicine_single_agent"
  "winogrande_single_agent"
  "formal_logic_hetero_no_persona"
  "hh_rlhf_hetero"
  "formal_logic_hetero_no_persona_4models"
  "hh_rlhf_hetero2"
  "formal_logic_homo_ablation"
  "hh_rlhf_hetero_no_persona"
  "truthfulqa_hetero"
)

MODES=(
  "round_agent_avg"
  "per_question_agent"
  "round_cum_text"
)

SKIP_DONE=1
# ===== End configuration =====

if [[ ! -f "${ANALYSIS_PY}" ]]; then
  echo "ERROR: analysis.py not found at: ${ANALYSIS_PY}"
  exit 1
fi

mkdir -p "${BASE_OUT}/logs"


run_dir_on_gpu() {
  local gpu="$1"
  local dir="$2"
  local abs_dir="${BASE_DIR}/${dir}"

  if [[ ! -d "${abs_dir}" ]]; then
    echo "[GPU ${gpu}] WARN missing dir, skip: ${abs_dir}"
    return 0
  fi

  local log="${BASE_OUT}/logs/${dir}_gpu${gpu}.log"
  echo "[GPU ${gpu}] START ${dir}  log=${log}"

  {
    echo "===== START $(date) ====="
    echo "GPU=${gpu}"
    echo "DIR=${abs_dir}"
  } >> "${log}"

  export CUDA_VISIBLE_DEVICES="${gpu}"

  # ==========
  # Check if there are subdirectories (e.g., llama3.1-8b_persona)
  local has_subdirs=0
  for subdir in "${abs_dir}"/*/; do
    if [[ -d "${subdir}" && -d "${subdir}/history" ]]; then
      has_subdirs=1
      break
    fi
  done

  if [[ ${has_subdirs} -eq 1 ]]; then
    # Has config subdirectories, iterate over them
    for config_dir in "${abs_dir}"/*/; do
      if [[ ! -d "${config_dir}/history" ]]; then
        continue
      fi
      
      local config_name=$(basename "${config_dir}")
      local history_dir="${config_dir}/history"
      
      for mode in "${MODES[@]}"; do
        local out_dir="${BASE_OUT}/${dir}/${config_name}/${mode}"
        mkdir -p "${out_dir}"

        if [[ "${SKIP_DONE}" -eq 1 && -f "${out_dir}/file_summary.csv" ]]; then
          echo "[SKIP] ${dir}/${config_name} mode=${mode} already done" | tee -a "${log}"
          continue
        fi

        echo "----- [GPU ${gpu}] RUN ${dir}/${config_name} mode=${mode} -----" | tee -a "${log}"

        python "${ANALYSIS_PY}" "${history_dir}" \
          --mode "${mode}" \
          --batch-size "${BATCH_SIZE}" \
          --cache-folder "${CACHE_FOLDER}" \
          --out-dir "${out_dir}" \
          2>&1 | tee -a "${log}" || echo "[ERROR] ${dir}/${config_name} mode=${mode} failed" | tee -a "${log}"
      done
    done
  else
    # No subdirectories, process directly (if abs_dir itself has history)
    if [[ -d "${abs_dir}/history" ]]; then
      for mode in "${MODES[@]}"; do
        local out_dir="${BASE_OUT}/${dir}/${mode}"
        mkdir -p "${out_dir}"

        if [[ "${SKIP_DONE}" -eq 1 && -f "${out_dir}/file_summary.csv" ]]; then
          echo "[SKIP] ${dir} mode=${mode} already done" | tee -a "${log}"
          continue
        fi

        echo "----- [GPU ${gpu}] RUN ${dir} mode=${mode} -----" | tee -a "${log}"

        python "${ANALYSIS_PY}" "${abs_dir}/history" \
          --mode "${mode}" \
          --batch-size "${BATCH_SIZE}" \
          --cache-folder "${CACHE_FOLDER}" \
          --out-dir "${out_dir}" \
          2>&1 | tee -a "${log}" || echo "[ERROR] ${dir} mode=${mode} failed" | tee -a "${log}"
      done
    fi
  fi

  echo "===== DONE $(date) =====" | tee -a "${log}"
  echo "[GPU ${gpu}] DONE  ${dir}"
}


# ===== Round-robin task assignment to GPUs =====
NUM_GPUS=${#GPUS[@]}

declare -A GPU_TASKS
for i in "${!DIRS[@]}"; do
  gpu_idx=$((i % NUM_GPUS))
  gpu="${GPUS[$gpu_idx]}"
  GPU_TASKS[$gpu]+="${DIRS[$i]} "
done

pids=()
for gpu in "${GPUS[@]}"; do
  (
    for dir in ${GPU_TASKS[$gpu]}; do
      run_dir_on_gpu "${gpu}" "${dir}"
    done
  ) &
  pids+=($!)
  echo "Started GPU ${gpu}, tasks: ${GPU_TASKS[$gpu]}"
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo ""
echo "All experiments finished. Outputs under: ${BASE_OUT}" 
