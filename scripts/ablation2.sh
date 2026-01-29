#!/usr/bin/env bash
# exp_homogeneous_all_models.sh - Homogeneous experiment: 4 models, each with/without persona
set -euo pipefail

cd .

DATA="gsm8k"    
DATA_SIZE=126
ROUNDS=3
DATA_DIR="./data"
SEED=42

# 4 models
MODELS=("llama3.1-8b" "qwen2.5-7b" "mistral-7b")

# Corresponding vLLM ports (assuming 4 services are running)
declare -A VLLM_PORTS
VLLM_PORTS["llama3.1-8b"]=8001
VLLM_PORTS["qwen2.5-7b"]=8002
VLLM_PORTS["mistral-7b"]=8003

AGENT_NUMS=(2 4 8 12 16)
SOLVERS=("debate" "vote")

GPU_LIST=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_LIST[@]}
MAX_JOBS_PER_GPU=10
MAX_PARALLEL_JOBS=$((GPU_COUNT * MAX_JOBS_PER_GPU))

BASE_OUT_DIR="./${DATA}_homo_ablation"

echo "=========================================="
echo "Homogeneous experiment: 4 models x 2 configs (with/without persona)"
echo "Models: ${MODELS[*]}"
echo "Dataset: ${DATA}"
echo "Start time: $(date)"
echo "=========================================="

run_experiment() {
    local MODEL=$1
    local SOLVER=$2
    local NUM_AGENTS=$3
    local USE_PERSONA=$4   # 1 or 0
    local GPU_ID=$5
    local PHYS_GPU_ID=${GPU_LIST[$GPU_ID]}
    
    local PORT=${VLLM_PORTS[$MODEL]}
    local VLLM_URL="http://127.0.0.1:${PORT}/v1"
    
    # Output directory differentiation
    if [[ "$USE_PERSONA" == "1" ]]; then
        local OUT_DIR="${BASE_OUT_DIR}/${MODEL}_persona"
        local LOG_SUFFIX="persona"
    else
        local OUT_DIR="${BASE_OUT_DIR}/${MODEL}_nopersona"
        local LOG_SUFFIX="nopersona"
    fi
    
    local LOG_DIR="${OUT_DIR}/logs"
    mkdir -p "${OUT_DIR}/history" "${LOG_DIR}"
    
    local LOG_FILE="${LOG_DIR}/${SOLVER}_N${NUM_AGENTS}_${LOG_SUFFIX}.log"
    
    echo "[$(date '+%H:%M:%S')] [GPU ${PHYS_GPU_ID}] ${MODEL} | ${SOLVER} | N=${NUM_AGENTS} | persona=${USE_PERSONA}"

    # Build arguments
    local EXTRA_ARGS=()
    EXTRA_ARGS+=(--use_vllm)
    EXTRA_ARGS+=(--vllm_base_url "$VLLM_URL")
    EXTRA_ARGS+=(--out_dir "$OUT_DIR")
    
    if [[ "$USE_PERSONA" == "1" ]]; then
        EXTRA_ARGS+=(--multi_persona)
    fi

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

    echo "✓ ${MODEL} | ${SOLVER} | N=${NUM_AGENTS} | persona=${USE_PERSONA} done"
}

export -f run_experiment
export DATA DATA_SIZE ROUNDS DATA_DIR SEED BASE_OUT_DIR GPU_COUNT
export GPU_LIST
export VLLM_PORTS
export -A VLLM_PORTS

# Generate all tasks: 4 models x 2 configs x 2 solvers x 5 agent counts = 80 tasks
TASKS=()
for MODEL in "${MODELS[@]}"; do
    for USE_PERSONA in 0 1; do
        for SOLVER in "${SOLVERS[@]}"; do
            for NUM_AGENTS in "${AGENT_NUMS[@]}"; do
                TASKS+=("$MODEL $SOLVER $NUM_AGENTS $USE_PERSONA")
            done
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "Total tasks: ${TOTAL_TASKS}"
echo ""

# Parallel execution
TASK_IDX=0
RUNNING_JOBS=0

# Launch initial tasks
for ((i=0; i<MAX_PARALLEL_JOBS && TASK_IDX<TOTAL_TASKS; i++)); do
    GPU_ID=$((i / MAX_JOBS_PER_GPU % GPU_COUNT))
    IFS=' ' read -r MODEL SOLVER NUM_AGENTS USE_PERSONA <<< "${TASKS[$TASK_IDX]}"
    run_experiment "$MODEL" "$SOLVER" "$NUM_AGENTS" "$USE_PERSONA" "$GPU_ID" &
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    TASK_IDX=$((TASK_IDX + 1))
done

# Dynamic scheduling
while [[ $TASK_IDX -lt $TOTAL_TASKS ]] || [[ $RUNNING_JOBS -gt 0 ]]; do
    wait -n 2>/dev/null || true
    RUNNING_JOBS=$((RUNNING_JOBS - 1))

    if [[ $TASK_IDX -lt $TOTAL_TASKS ]]; then
        GPU_ID=$((TASK_IDX % MAX_PARALLEL_JOBS / MAX_JOBS_PER_GPU % GPU_COUNT))
        IFS=' ' read -r MODEL SOLVER NUM_AGENTS USE_PERSONA <<< "${TASKS[$TASK_IDX]}"
        run_experiment "$MODEL" "$SOLVER" "$NUM_AGENTS" "$USE_PERSONA" "$GPU_ID" &
        RUNNING_JOBS=$((RUNNING_JOBS + 1))
        TASK_IDX=$((TASK_IDX + 1))
        echo ">>> Progress: ${TASK_IDX}/${TOTAL_TASKS}"
    fi
done

wait

echo ""
echo "=========================================="
echo "All done!"
echo "End time: $(date)"
echo "Completed ${TOTAL_TASKS} experiments"
echo ""
echo "Output directory structure:"
echo "  ${BASE_OUT_DIR}/"
echo "    ├── llama3.1-8b_nopersona/"
echo "    ├── llama3.1-8b_persona/"
echo "    ├── qwen2.5-7b_nopersona/"
echo "    ├── qwen2.5-7b_persona/"
echo "    ├── mistral-7b_nopersona/"
echo "    ├── mistral-7b_persona/"
echo "    ├── qwen3-8b_nopersona/"
echo "    └── qwen3-8b_persona/"
echo "=========================================="