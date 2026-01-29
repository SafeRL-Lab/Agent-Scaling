\#!/usr/bin/env bash
# exp1_marginal_gain_full_hetero_only.sh
set -euo pipefail

cd .

DATA="truthfulqa"          # or formal_logic / gsm8k / hellaswag
MODEL="qwen2.5-7b"
DATA_SIZE=126
ROUNDS=3
DATA_DIR="./data"

HET_AGENT_MODELS="llama3.1-8b,qwen2.5-7b,mistral-7b"
SEED=42

# Output directory configuration
OUT_DIR="./${DATA}_hetero"
LOG_DIR="${OUT_DIR}/exp1_logs"

# bitsandbytes (only effective for local HF models; not used with vLLM)
USE_BNB_4BIT=0
USE_BNB_8BIT=0

# Create necessary directories
mkdir -p "${OUT_DIR}/history" "${LOG_DIR}"

AGENT_NUMS=(2 4 8 12 16)
SOLVERS=("debate" "vote")

# Available physical GPU IDs
GPU_LIST=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_LIST[@]}
MAX_JOBS_PER_GPU=12
MAX_PARALLEL_JOBS=$((GPU_COUNT * MAX_JOBS_PER_GPU))

# vLLM configuration
export USE_VLLM=1
export VLLM_BASE_URLS="http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1"

echo "=========================================="
echo "Exp1: Marginal gain analysis (heterogeneous only)"
echo "Dataset: ${DATA}"
echo "Output directory: ${OUT_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Start time: $(date)"
echo "GPU_LIST: ${GPU_LIST[*]}"
echo "=========================================="

run_experiment() {
    local SOLVER=$1
    local NUM_AGENTS=$2
    local GPU_ID=$3                # Logical GPU index: 0..GPU_COUNT-1
    local PHYS_GPU_ID=${GPU_LIST[$GPU_ID]}  # Map to physical GPU

    local START_TIME
    START_TIME=$(date +%s)

    # Store args in array to avoid quoting issues
    local EXTRA_ARGS=()
    EXTRA_ARGS+=(--multi_persona)
    EXTRA_ARGS+=(--agent_models "$HET_AGENT_MODELS")
    EXTRA_ARGS+=(--out_dir "$OUT_DIR")

    # Add vLLM parameters
    EXTRA_ARGS+=(--use_vllm)
    EXTRA_ARGS+=(--vllm_base_urls "$VLLM_BASE_URLS")

    if [[ "$USE_BNB_4BIT" == "1" ]]; then
        EXTRA_ARGS+=(--load_in_4bit)
    elif [[ "$USE_BNB_8BIT" == "1" ]]; then
        EXTRA_ARGS+=(--load_in_8bit)
    fi

    local LOG_FILE="${LOG_DIR}/${SOLVER}_N${NUM_AGENTS}_heterogeneous.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU ${PHYS_GPU_ID} (slot ${GPU_ID})] [${SOLVER}] N=${NUM_AGENTS} (heterogeneous) started"
    echo "  Log file: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${PHYS_GPU_ID} python src/main.py \
        --model "$MODEL" \
        --num_agents "$NUM_AGENTS" \
        --data "$DATA" \
        --data_size "$DATA_SIZE" \
        --debate_rounds "$ROUNDS" \
        --solver "$SOLVER" \
        --data_dir "$DATA_DIR" \
        --seed "$SEED" \
        --verbose \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"

    local END_TIME
    END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    echo "✓ [$(date '+%Y-%m-%d %H:%M:%S')] [GPU ${PHYS_GPU_ID} (slot ${GPU_ID})] [${SOLVER}] N=${NUM_AGENTS} (heterogeneous) done! Duration: ${DURATION}s"
}

export -f run_experiment
export MODEL DATA DATA_SIZE ROUNDS DATA_DIR SEED HET_AGENT_MODELS USE_BNB_4BIT USE_BNB_8BIT USE_VLLM VLLM_BASE_URLS GPU_COUNT MAX_JOBS_PER_GPU OUT_DIR LOG_DIR
export GPU_LIST

# Generate heterogeneous tasks only
TASKS=()
for SOLVER in "${SOLVERS[@]}"; do
    for NUM_AGENTS in "${AGENT_NUMS[@]}"; do
        TASKS+=("$SOLVER $NUM_AGENTS")
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "Total tasks (heterogeneous only): ${TOTAL_TASKS}"
echo ""

TASK_IDX=0
RUNNING_JOBS=0

# Launch initial tasks
for ((i=0; i<MAX_PARALLEL_JOBS && TASK_IDX<TOTAL_TASKS; i++)); do
    GPU_ID=$((i / MAX_JOBS_PER_GPU % GPU_COUNT))   # 0..GPU_COUNT-1
    IFS=' ' read -r SOLVER NUM_AGENTS <<< "${TASKS[$TASK_IDX]}"
    run_experiment "$SOLVER" "$NUM_AGENTS" "$GPU_ID" &
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    TASK_IDX=$((TASK_IDX + 1))
done

# Dynamic scheduling
while [[ $TASK_IDX -lt $TOTAL_TASKS ]] || [[ $RUNNING_JOBS -gt 0 ]]; do
    wait -n 2>/dev/null || true
    RUNNING_JOBS=$((RUNNING_JOBS - 1))

    if [[ $TASK_IDX -lt $TOTAL_TASKS ]]; then
        SLOT_ID=$((TASK_IDX % MAX_PARALLEL_JOBS))
        GPU_ID=$((SLOT_ID / MAX_JOBS_PER_GPU % GPU_COUNT))  # 0..GPU_COUNT-1

        IFS=' ' read -r SOLVER NUM_AGENTS <<< "${TASKS[$TASK_IDX]}"
        run_experiment "$SOLVER" "$NUM_AGENTS" "$GPU_ID" &
        RUNNING_JOBS=$((RUNNING_JOBS + 1))
        TASK_IDX=$((TASK_IDX + 1))

        echo ">>> Progress: ${TASK_IDX}/${TOTAL_TASKS} (remaining: $((TOTAL_TASKS - TASK_IDX)))"
    fi
done

wait

echo ""
echo "=========================================="
echo "Exp1 (heterogeneous only) all done!"
echo "End time: $(date)"
echo "Completed ${TOTAL_TASKS} experiments"
echo "Output directory: ${OUT_DIR}"
echo "Log files: ${LOG_DIR}"
echo "=========================================="