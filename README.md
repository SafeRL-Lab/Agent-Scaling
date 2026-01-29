# Agent-Scaling: Multi-Agent Scaling with Heterogeneous LLM Ensembles

This repository provides the codebase for studying **how scaling the number of heterogeneous LLM agents with diverse reasoning personas improves collective performance** through debate and voting mechanisms. We introduce the **K\* metric** (effective diversity) based on embedding eigenvalue entropy to quantify semantic diversity among agents, and show that persona-guided multi-agent collaboration yields consistent gains across reasoning benchmarks.

## Project Structure

```
.
├── src/                          # Core source code
│   ├── main.py                   # Main orchestration: debate/voting loop
│   ├── evaluator.py              # Answer extraction & scoring (math, MCQ)
│   ├── data/                     # Dataset loaders
│   │   ├── data_utils.py         # Central data router
│   │   ├── gsm8k.py              # Grade School Math 8K
│   │   ├── arc.py                # ARC-Challenge / ARC-Easy
│   │   ├── hellaswag.py          # HellaSwag
│   │   ├── truthfulqa.py         # TruthfulQA
│   │   ├── piqa.py               # Physical Intuition QA
│   │   ├── winogrande.py         # WinoGrande
│   │   ├── mmlu_pro_medicine.py  # MMLU-Pro Medicine
│   │   └── mmlu_formal_logic.py  # MMLU Formal Logic
│   └── model/                    # Model wrappers
│       ├── model_utils.py        # Agent factory, persona definitions, unified engine
│       ├── llama.py              # LLaMA (v2/v3) wrapper via HuggingFace
│       ├── qwen.py               # Qwen wrapper via HuggingFace
│       ├── openai_compat.py      # OpenAI-compatible API client (vLLM, etc.)
│       └── azure_openai.py       # Azure OpenAI wrapper
├── scripts/                      # Experiment runner scripts
│   ├── add*.sh                   # Heterogeneous multi-agent experiments
│   ├── add*_noperspn.sh          # Same experiments without personas
│   └── ablation*.sh              # Ablation studies (persona impact, agent count)
├── K_star_analysis/              # K* diversity metric computation
│   ├── analysis.py               # Core N* (effective diversity) from embeddings
│   ├── analysis_improved.py      # Extended metrics: N*_conditioned, N*_weighted, Delta-N*
│   ├── exp2_embedding_robustness.py  # Cross-embedding-model robustness validation
│   ├── analysis.sh               # Runner for analysis.py
│   └── analysis_improved.sh      # Runner for analysis_improved.py
└── .gitignore
```

## Installation

### Requirements

- Python >= 3.9
- CUDA-compatible GPU(s)

### Dependencies

```bash
pip install torch transformers accelerate peft
pip install sentence-transformers datasets
pip install openai numpy pandas scipy tqdm
```

## Quick Start

### 1. Homogeneous Multi-Agent Voting

```bash
python src/main.py \
    --data gsm8k \
    --num_agents 5 \
    --model qwen2.5-7b \
    --solver vote \
    --debate_rounds 0
```

### 2. Heterogeneous Multi-Agent Debate with Personas

```bash
python src/main.py \
    --data formal_logic \
    --num_agents 4 \
    --agent_models "llama3.1-8b,qwen2.5-7b,mistral-7b,qwen3-8b" \
    --multi_persona \
    --solver debate \
    --debate_rounds 3 \
    --use_vllm \
    --vllm_base_urls "http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1"
```

### 3. Using Azure OpenAI Models

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY_ENV="your-api-key"

python src/main.py \
    --data gsm8k \
    --num_agents 4 \
    --agent_models "gpt-4o,gpt-4o-mini,o3-mini,gpt-4.1" \
    --multi_persona \
    --solver debate \
    --debate_rounds 3
```

### 4. Run K\* Analysis

```bash
python K_star_analysis/analysis.py \
    --jsonl_dir out/history/ \
    --mode round_agent_avg \
    --output_dir analysis/results/
```

## Key Concepts

### Multi-Agent Debate

Agents iteratively refine their answers by observing peer responses:

1. **Round 0**: Each agent generates an initial answer guided by its persona.
2. **Rounds 1–N**: Agents receive peer responses as context and produce refined answers.
3. **Final answer**: Determined by majority voting or debate consensus.

Three communication topologies are supported:
- **Decentralized** (default): Full mesh — every agent sees all peers.
- **Centralized**: Hub-and-spoke — one coordinator aggregates.
- **Sparse**: Each agent only sees neighboring agents.

### Reasoning Personas

Each agent is assigned a distinct reasoning persona that shapes its problem-solving style. For example, on math tasks:

| Persona | Strategy |
|---|---|
| Conservative Verifier | Step-by-step verification, double-checking |
| Creative Explorer | Pattern recognition, unconventional shortcuts |
| Rigorous Formalist | Precise notation, logical completeness |
| Intuitive Estimator | Rough estimates as sanity checks |
| Systematic Decomposer | Divide-and-conquer sub-problems |

### K\* Metric (Effective Diversity)

K\* measures the **effective number of distinct reasoning strategies** among agents using embedding-space analysis:

- Embed all agent responses with a sentence transformer (e.g., NV-Embed-v2).
- Compute eigenvalues of the embedding covariance matrix.
- **N\* = exp(H)** where H is the Shannon entropy of the normalized eigenvalue distribution.

Extended metrics include:
- **N\*_conditioned**: Diversity within correct vs. incorrect answer groups.
- **N\*_weighted**: Correctness-weighted diversity.
- **Delta-N\***: Marginal diversity contribution of each agent.

### Baselines

- **Baseline A**: Length-matched neutral padding replaces persona prompts (isolates persona effect).
- **Baseline B**: All agents share a single persona (tests heterogeneity benefit).

## Supported Models

| Category | Models |
|---|---|
| Local (HuggingFace) | LLaMA 3.1-8B, Qwen 2.5-7B/32B |
| vLLM Serving | Any model served via OpenAI-compatible API |
| Azure OpenAI | GPT-4o, GPT-4o-mini, o1, o3-mini, o3, o4-mini, GPT-4.1 series |
| OpenAI-Compatible | GPT-5-mini, Gemini-2.5-Flash, etc. |

## Supported Datasets

| Task Type | Datasets |
|---|---|
| Mathematical Reasoning | GSM8K |
| Multiple Choice QA | ARC, HellaSwag, TruthfulQA, PIQA, WinoGrande |
| Domain-Specific | MMLU-Pro Medicine, MMLU Formal Logic |

## Running Large-Scale Experiments

The `scripts/` directory contains bash scripts for running experiment sweeps:

```bash
# Heterogeneous agents across multiple agent counts
bash scripts/add.sh

# Same experiment without personas (control)
bash scripts/add_noperspn.sh

# Ablation: persona vs. no-persona across models
bash scripts/ablation.sh
```

These scripts manage multi-GPU scheduling and run experiments across varying agent counts (2, 4, 8, 12, 16) with both debate and voting solvers.

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset name | (required) |
| `--num_agents` | Number of agents | 5 |
| `--agent_models` | Comma-separated model list (heterogeneous) | `""` |
| `--multi_persona` | Enable diverse personas per agent | `False` |
| `--baseline_a` | Baseline A: neutral padding | `False` |
| `--baseline_b` | Baseline B: shared persona | `False` |
| `--solver` | Aggregation method: `vote` or `debate` | `vote` |
| `--debate_rounds` | Number of debate rounds | 5 |
| `--sparse` | Sparse communication topology | `False` |
| `--centralized` | Centralized communication topology | `False` |
| `--use_vllm` | Use vLLM backend | `False` |
| `--temperature` | Sampling temperature | 1.0 |
| `--top_p` | Nucleus sampling threshold | 0.9 |
| `--load_in_4bit` | 4-bit quantization (bitsandbytes) | `False` |
| `--load_in_8bit` | 8-bit quantization (bitsandbytes) | `False` |

## License

This project is for research purposes.
