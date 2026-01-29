import argparse, sys, os, copy, time, random, json, pickle, re, collections, gc
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
from model.model_utils import get_agents, engine, get_persona_config
from data.data_utils import load_data
from evaluator import (
    get_instruction_suffix,
    evaluate_gsm8k,
    evaluate_mcq,
    base_evaluate_gsm8k,
    base_evaluate_mcq,
)


def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default="out/")

    # data
    parser.add_argument('--data_dir', type=str, default="./data/")
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--sub_data', type=str, default='')
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--debug', action='store_true')

    # agent
    parser.add_argument('--num_agents', type=int, default=5)
    parser.add_argument('--agent_selection', type=str, default="none")
    parser.add_argument('--multi_persona', action='store_true')
    parser.add_argument('--baseline_a', action='store_true',
                        help='Baseline A: length-matched neutral padding without persona')
    parser.add_argument('--baseline_b', action='store_true',
                        help='Baseline B: single shared persona for all agents')

    # model
    parser.add_argument('--model', type=str, default="llama3.1")
    parser.add_argument('--agent_models', type=str, default="")

    # Azure OpenAI settings
    parser.add_argument('--azure_endpoint', type=str, default=os.getenv('AZURE_OPENAI_ENDPOINT', ''))
    parser.add_argument('--azure_api_version', type=str, default=os.getenv('AZURE_OPENAI_API_VERSION', '2025-04-01-preview'))
    parser.add_argument('--azure_api_key_env', type=str, default=os.getenv('AZURE_OPENAI_API_KEY_ENV', 'API_KEY'))

    # OpenAI-compatible API settings (for closed-source models like gpt-5-mini, gemini-2.5-flash)
    parser.add_argument('--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY', ''))
    parser.add_argument('--openai_base_url', type=str, default=os.getenv('OPENAI_BASE_URL', ''))

    # vLLM settings
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--vllm_base_urls', type=str, default=os.getenv('VLLM_BASE_URLS', ''))
    parser.add_argument('--vllm_base_url', type=str, default=os.getenv('VLLM_BASE_URL', 'http://127.0.0.1:8000/v1'))
    parser.add_argument('--vllm_api_key', type=str, default=os.getenv('VLLM_API_KEY', 'EMPTY'))
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)

    # local model cache dir
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')

    # debate
    parser.add_argument('--debate_rounds', type=int, default=5)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--centralized', action='store_true')

    parser.add_argument('--solver', type=str, default='vote', choices=['vote', 'debate'])
    parser.add_argument('--generate_first_round', action='store_true')
    parser.add_argument('--max_num_agents', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--bae', action='store_true')
    parser.add_argument('--cot', action='store_true')

    return parser.parse_args()


def _build_length_match_pad(personas: dict, target_chars: Optional[int] = None) -> str:
    """Build neutral, non-instructional padding text to match persona prompt length (roughly, in characters)."""
    # pick a representative persona prompt length if not provided
    if target_chars is None:
        for _, pdata in (personas or {}).items():
            if isinstance(pdata, dict) and pdata.get("prompt"):
                target_chars = len(pdata["prompt"])
                break
        if target_chars is None:
            target_chars = 0

    if target_chars <= 0:
        return ""

    # Use digits and separators to minimize semantic content while still consuming tokens.
    base = "0123456789-=+_*/.,;:|~"
    chunk = (base + " ") * ((target_chars // (len(base) + 1)) + 2)
    return chunk[:target_chars]


def get_new_message(args, sample, responses, personas=None, suffix=None, agent_persona_map=None):
    """
    Generate messages for the next debate round.
    Returns:
        new_message: dict, format {agent_name: {"role": "user", "content": "..."}}
        persona_configs: list, per-agent sampling configuration
    """
    new_message = {}
    persona_configs = []
    agents = list(responses.keys())

    if getattr(args, "verbose", False):
        print("\n[get_new_message] agent assignments:")
        for a in agents:
            parts = a.split("__")
            model = parts[-3] if len(parts) >= 3 else "UNKNOWN_MODEL"
            persona = parts[-2] if len(parts) >= 2 else "UNKNOWN_PERSONA"
            agent_id = parts[-1] if len(parts) >= 1 else "UNKNOWN_AGENT"
            print(f"  - {agent_id} | persona={persona} | model={model}")

    if len(agents) > 1:  # MULTI-AGENT DEBATE
        if not args.centralized:  # DECENTRALIZED MAD
            for i, agent in enumerate(agents):
                msg = "These are the recent opinions from other agents: "
                if args.sparse:
                    peers = [agents[(i - 1) % len(agents)], agents[(i + 1) % len(agents)]]
                else:
                    peers = agents[:i] + agents[i + 1:]

                for other_agent in peers:
                    msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}"

                if suffix is not None:
                    msg += suffix

                # Retrieve persona information
                persona_name = agent.split("__")[-2]
                if personas and persona_name in personas:
                    persona_data = personas[persona_name]
                    if isinstance(persona_data, dict):
                        # Enhanced persona with config
                        final_msg = f"{persona_data['prompt']}\n\n{msg}"
                        persona_configs.append(get_persona_config(persona_name, personas))
                    else:
                        # Legacy persona (plain string)
                        final_msg = f"{persona_data}\n\n{msg}"
                        persona_configs.append(None)
                else:
                    final_msg = msg
                    persona_configs.append(None)

                new_message[agent] = {"role": "user", "content": final_msg}

        else:  # CENTRALIZED MAD
            for i, agent in enumerate(agents):
                if i == 0:
                    msg = "These are the recent opinions from other agents: "
                    peers = agents[:i] + agents[i + 1:]
                    for other_agent in peers:
                        msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}"
                else:
                    msg = f"This is the recent opinion from another agent: \n{responses[agents[0]]}\n"
                    msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
                    msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}"

                if suffix is not None:
                    msg += suffix

                # Baseline A: length-matched neutral padding
                if getattr(args, 'baseline_a', False) and getattr(args, 'length_match_pad', ''):
                    msg = f"{args.length_match_pad}\n\n{msg}"

                persona_name = agent.split("__")[-2]
                if personas and persona_name in personas:
                    persona_data = personas[persona_name]
                    if isinstance(persona_data, dict):
                        final_msg = f"{persona_data['prompt']}\n\n{msg}"
                        persona_configs.append(get_persona_config(persona_name, personas))
                    else:
                        final_msg = f"{persona_data}\n\n{msg}"
                        persona_configs.append(None)
                else:
                    final_msg = msg
                    persona_configs.append(None)

                new_message[agent] = {"role": "user", "content": final_msg}

    else:  # SINGLE AGENT SELF REFINEMENT
        for i, agent in enumerate(agents):
            msg = f"This was your most recent opinion:\n{responses[agents[i]]}\n"
            msg += f"\n\nRevise your recent opinion to give your updated final answer to the question:\n{sample}"

            if suffix is not None:
                msg += suffix

            # Baseline A: length-matched neutral padding
            if getattr(args, 'baseline_a', False) and getattr(args, 'length_match_pad', ''):
                msg = f"{args.length_match_pad}\n\n{msg}"

            persona_name = agent.split("__")[-2]
            if personas and persona_name in personas:
                persona_data = personas[persona_name]
                if isinstance(persona_data, dict):
                    final_msg = f"{persona_data['prompt']}\n\n{msg}"
                    persona_configs.append(get_persona_config(persona_name, personas))
                else:
                    final_msg = f"{persona_data}\n\n{msg}"
                    persona_configs.append(None)
            else:
                final_msg = msg
                persona_configs.append(None)

            new_message[agent] = {"role": "user", "content": final_msg}

    return new_message, persona_configs


def main(args):
    # =========================
    # Load Agents
    # =========================
    agents, personas = get_agents(args)

    # -------------------------
    # Baseline mode validation / setup
    # -------------------------
    if getattr(args, "baseline_a", False) and getattr(args, "baseline_b", False):
        raise ValueError("baseline_a and baseline_b are mutually exclusive. Please enable only one.")
    if getattr(args, "multi_persona", False) and (getattr(args, "baseline_a", False) or getattr(args, "baseline_b", False)):
        raise ValueError("multi_persona cannot be combined with baseline_a/baseline_b. Choose one mode.")

    # Precompute neutral padding for Baseline A (length-matched prompt)
    if getattr(args, "baseline_a", False):
        args.length_match_pad = _build_length_match_pad(personas)
    else:
        args.length_match_pad = ""

    if not isinstance(agents, (list, tuple)):
        agents = [agents] * args.num_agents
    else:
        agents = list(agents)
        if len(agents) < args.num_agents:
            agents = [agents[i % len(agents)] for i in range(args.num_agents)]
        elif len(agents) > args.num_agents:
            agents = agents[:args.num_agents]

    agent_model_keys = getattr(args, "agent_model_keys", None)
    if (agent_model_keys is None or len(agent_model_keys) == 0) and args.agent_models:
        parsed = [mk.strip() for mk in args.agent_models.split(",") if mk.strip()]
        agent_model_keys = parsed if parsed else [args.model]

    if agent_model_keys is None:
        agent_model_keys = [args.model] * args.num_agents
    elif len(agent_model_keys) < args.num_agents:
        agent_model_keys = [agent_model_keys[i % len(agent_model_keys)] for i in range(args.num_agents)]
    else:
        agent_model_keys = list(agent_model_keys)[:args.num_agents]

    if args.verbose:
        print("[main] num_agents =", args.num_agents)
        print("[main] agent_model_keys =", agent_model_keys)
        print("[main] personas loaded:")
        for pname, pdata in personas.items():
            if isinstance(pdata, dict):
                print(f"  - {pname}: temp={pdata.get('temperature')}, top_p={pdata.get('top_p')}, style={pdata.get('style')}")
            else:
                print(f"  - {pname}: (legacy string persona)")

    # =========================
    # Load Data
    # =========================
    test_X, test_Y = load_data(args, split='test')

    # =========================
    # Setup Names / filenames
    # =========================
    model_tag = args.model
    if args.agent_models:
        uniq = list(dict.fromkeys(agent_model_keys))
        model_tag = f"MIXED{len(uniq)}"

    fname = f"{args.data}_{args.solver}_{args.data_size}__{model_tag}_N={args.num_agents}_R={args.debate_rounds}"
    if args.sparse:
        fname += '_SPARSE'
    elif args.centralized:
        fname += '_CENTRAL'
    if args.bae:
        fname += '_BAE'
    if args.multi_persona:
        fname += '_HETERO_ENHANCED'

    if getattr(args, "baseline_a", False):
        fname += '_BASEA_LENMATCH'
    if getattr(args, "baseline_b", False):
        fname += '_BASEB_SINGLEP'

    SUFFIX = get_instruction_suffix(args)

    if args.data in ['gsm8k']:
        evaluate = base_evaluate_gsm8k if args.bae else evaluate_gsm8k
    elif args.data in ['hellaswag', 'pro_medicine', 'formal_logic', 'arc', 'piqa', 'truthfulqa', 'winogrande']:
        evaluate = base_evaluate_mcq if args.bae else evaluate_mcq
    else:
        raise NotImplementedError

    # =========================
    # Debate Loop
    # =========================
    sample_responses = []
    iscorr_list = []

    for idx, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=len(test_X)):
        print('\n\nQuestion: ', x + SUFFIX, '\n')

        print("Gathering initial opinions...")
        round_iscorr = []

        # ============================================================
        # Build initial messages
        # ============================================================
        if args.multi_persona:
            messages = []
            persona_configs = []
            persona_list = list(personas.items())

            for i in range(args.num_agents):
                persona_name, persona_data = persona_list[i % len(persona_list)]

                if isinstance(persona_data, dict):
                    combined_content = f"{persona_data['prompt']}\n\n{x + SUFFIX}"
                    persona_configs.append({
                        "temperature": persona_data.get("temperature", 0),
                        "top_p": persona_data.get("top_p", 0.9),
                        "max_new_tokens": getattr(args, 'max_new_tokens', 512)
                    })
                else:
                    combined_content = f"{persona_data}\n\n{x + SUFFIX}"
                    persona_configs.append(None)

                messages.append({"role": "user", "content": combined_content})

            agent_names = []
            for i in range(args.num_agents):
                persona_name = persona_list[i % len(persona_list)][0]
                mk = agent_model_keys[i % len(agent_model_keys)]
                agent_names.append(f"{args.data}_{args.data_size}__{mk}__{persona_name}__Agent{i+1}")

        elif getattr(args, "baseline_b", False):
            # Single shared persona for all agents
            persona_list = list(personas.items())
            base_persona_name, base_persona_data = persona_list[0]
            if isinstance(base_persona_data, dict):
                base_prompt = base_persona_data.get("prompt", "")
                base_cfg = get_persona_config(base_persona_name, personas)
            else:
                base_prompt = str(base_persona_data)
                base_cfg = None

            combined_content = f"{base_prompt}\n\n{x + SUFFIX}" if base_prompt else (x + SUFFIX)
            messages = [{"role": "user", "content": combined_content}] * args.num_agents
            persona_configs = [base_cfg] * args.num_agents

            agent_names = [
                f"{args.data}_{args.data_size}__{agent_model_keys[i % len(agent_model_keys)]}__{base_persona_name}__Agent{i+1}"
                for i in range(args.num_agents)
            ]

        elif getattr(args, "baseline_a", False):
            # Length-matched neutral padding, no persona
            pad = getattr(args, "length_match_pad", "")
            combined_content = f"{pad}\n\n{x + SUFFIX}" if pad else (x + SUFFIX)
            messages = [{"role": "user", "content": combined_content}] * args.num_agents
            persona_configs = [None] * args.num_agents
            agent_names = [
                f"{args.data}_{args.data_size}__{agent_model_keys[i % len(agent_model_keys)]}__LenMatch__Agent{i+1}"
                for i in range(args.num_agents)
            ]

        else:
            # Default baseline (short prompt)
            messages = [{"role": "user", "content": x + SUFFIX}] * args.num_agents
            persona_configs = [None] * args.num_agents
            agent_names = [
                f"{args.data}_{args.data_size}__{agent_model_keys[i % len(agent_model_keys)]}__None__Agent{i+1}"
                for i in range(args.num_agents)
            ]

        # ============================================================
        # First round inference - pass persona_configs
        # ============================================================
        responses = engine(messages, agents, args.num_agents, persona_configs=persona_configs)
        agent_responses = dict(zip(agent_names, responses))

        # Verbose: print per-agent generation parameters
        if args.verbose:
            print("\n[Round 0] Agent generation configs:")
            for name, cfg in zip(agent_names, persona_configs):
                if cfg:
                    print(f"  {name}: temp={cfg.get('temperature')}, top_p={cfg.get('top_p')}")
                else:
                    print(f"  {name}: using default params")

        # evaluate
        if args.centralized:
            central_agent_response = {list(agent_responses.keys())[0]: list(agent_responses.values())[0]}
            final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
        else:
            final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

        print(f"ROUND 0 : {final_resps} (answer = {y})")

        if args.data in ['gsm8k']:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == np.round(y, 1) for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': np.round(y, 1),
                'persona_configs': [str(c) for c in persona_configs]
            }
        else:
            round_data = {
                'responses': agent_responses,
                'final_answers': final_resps,
                'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                'debate_answer': debate_resps,
                'debate_answer_iscorr': is_corr,
                'answer': y,
                'persona_configs': [str(c) for c in persona_configs]
            }

        rounds_data_dict = {'0': round_data}
        round_iscorr.append(is_corr)

        # ============================================================
        # Debate rounds
        # ============================================================
        for r in range(1, args.debate_rounds + 1):
            print(f"Debating round {r}...")

            if args.multi_persona or getattr(args, "baseline_b", False):
                new_agent_messages, persona_configs = get_new_message(
                    args, x, agent_responses, personas, suffix=SUFFIX
                )
            else:
                new_agent_messages, persona_configs = get_new_message(
                    args, x, agent_responses, suffix=SUFFIX
                )

            messages = list(new_agent_messages.values())
            responses = engine(messages, agents, args.num_agents, persona_configs=persona_configs)
            agent_responses = dict(zip(agent_names, responses))

            if args.verbose:
                print(f"\n[Round {r}] Agent generation configs:")
                for name, cfg in zip(agent_names, persona_configs):
                    if cfg:
                        print(f"  {name}: temp={cfg.get('temperature')}, top_p={cfg.get('top_p')}")

            if args.centralized:
                central_agent_response = {list(agent_responses.keys())[0]: list(agent_responses.values())[0]}
                final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
            else:
                final_resps, debate_resps, is_corr = evaluate(agent_responses, y)

            print("\n\n" + str(messages[0])[:200] + "...\n\n")
            print(f"ROUND {r} : {final_resps} (answer = {y})")

            if args.data in ['gsm8k']:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == np.round(y, 1) for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': np.round(y, 1),
                    'persona_configs': [str(c) for c in persona_configs]
                }
            else:
                round_data = {
                    'responses': agent_responses,
                    'final_answers': final_resps,
                    'final_answer_iscorr': [y_pred == y for y_pred in final_resps],
                    'debate_answer': debate_resps,
                    'debate_answer_iscorr': is_corr,
                    'answer': y,
                    'persona_configs': [str(c) for c in persona_configs]
                }

            rounds_data_dict[str(r)] = round_data
            round_iscorr.append(is_corr)

        sample_responses.append(rounds_data_dict)
        iscorr_list.append(round_iscorr)

        # Save
        history_dir = os.path.join(args.out_dir, "history")
        os.makedirs(history_dir, exist_ok=True)
        out_path = os.path.join(history_dir, f"{fname}.jsonl")
        with open(out_path, "w") as f:
            for record in sample_responses:
                f.write(json.dumps(record, default=convert_numpy) + "\n")

        round_accs = np.array(iscorr_list).mean(0)
        for i, acc in enumerate(round_accs):
            print(f'Round {i} Acc.: {acc:.4f}')

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, f"{fname}.tsv")
    with open(tsv_path, "a") as f:
        line = f"\n{args.timestamp}\t{fname}\t{round_accs}"
        f.writelines(line)


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open('token', 'r') as f:
        token = f.read().strip()
    args.token = token

    main(args)
