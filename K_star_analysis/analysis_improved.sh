#!/usr/bin/env bash
set -eo pipefail

# ===== Configuration =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${SCRIPT_DIR}/analysis_improved.py"
BASE_DIR="${SCRIPT_DIR}"
BASE_OUT="${SCRIPT_DIR}/ALL_ANALYSIS_OUTPUT"

# Directory list
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

SKIP_DONE=1
# ===== End configuration =====

if [[ ! -f "${ANALYSIS_PY}" ]]; then
  echo "ERROR: analysis_improved.py not found at: ${ANALYSIS_PY}"
  exit 1
fi

mkdir -p "${BASE_OUT}/logs"

run_dir() {
  local dir="$1"
  local abs_dir="${BASE_DIR}/${dir}"

  if [[ ! -d "${abs_dir}" ]]; then
    echo "[WARN] missing dir, skip: ${abs_dir}"
    return 0
  fi

  local log="${BASE_OUT}/logs/${dir}_improved.log"
  echo ""
  echo "=========================================="
  echo "START ${dir}"
  echo "log=${log}"
  echo "=========================================="

  {
    echo "===== START $(date) ====="
    echo "DIR=${abs_dir}"
  } >> "${log}"

  # Check if there are subdirectories
  local has_subdirs=0
  for subdir in "${abs_dir}"/*/; do
    if [[ -d "${subdir}" && -d "${subdir}/history" ]]; then
      has_subdirs=1
      break
    fi
  done

  if [[ ${has_subdirs} -eq 1 ]]; then
    for config_dir in "${abs_dir}"/*/; do
      if [[ ! -d "${config_dir}/history" ]]; then
        continue
      fi
      
      local config_name=$(basename "${config_dir}")
      local history_dir="${config_dir}/history"
      local out_dir="${BASE_OUT}/${dir}/${config_name}/improved"
      mkdir -p "${out_dir}"

      if [[ "${SKIP_DONE}" -eq 1 && -f "${out_dir}/improved_file_summary.csv" ]]; then
        echo "[SKIP] ${dir}/${config_name} already done"
        continue
      fi

      echo ""
      echo "----- RUN ${dir}/${config_name} -----"

      python "${ANALYSIS_PY}" "${history_dir}" \
        --out-dir "${out_dir}" \
        2>&1 | tee -a "${log}"
      
      if [[ $? -ne 0 ]]; then
        echo "[ERROR] ${dir}/${config_name} failed" | tee -a "${log}"
      fi
    done
  else
    if [[ -d "${abs_dir}/history" ]]; then
      local out_dir="${BASE_OUT}/${dir}/improved"
      mkdir -p "${out_dir}"

      if [[ "${SKIP_DONE}" -eq 1 && -f "${out_dir}/improved_file_summary.csv" ]]; then
        echo "[SKIP] ${dir} already done"
      else
        echo ""
        echo "----- RUN ${dir} -----"

        python "${ANALYSIS_PY}" "${abs_dir}/history" \
          --out-dir "${out_dir}" \
          2>&1 | tee -a "${log}"
        
        if [[ $? -ne 0 ]]; then
          echo "[ERROR] ${dir} failed" | tee -a "${log}"
        fi
      fi
    fi
  fi

  echo "===== DONE $(date) =====" >> "${log}"
  echo "[DONE] ${dir}"
}


# ===== Sequential execution =====
echo "Starting improved N* analysis..."
echo "This uses cached embeddings - no GPU needed."
echo ""

for dir in "${DIRS[@]}"; do
  run_dir "${dir}"
done

echo ""
echo "=========================================="
echo "All improved analysis finished!"
echo "Outputs under: ${BASE_OUT}/*/improved/"
echo "=========================================="
