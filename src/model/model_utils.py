# model_utils.py - enhanced version

model_dirs = {
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct'
}

AZURE_OPENAI_MODELS = {
    'gpt-4o', 'gpt-4o-mini',
    'o1', 'o3-mini', 'o3', 'o4-mini',
    'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
}

# Closed-source models (accessed via OpenAI-compatible API)
OPENAI_COMPAT_MODELS = {
    'gpt-5-mini', 'gpt-4.1-mini', 'gemini-2.5-flash',
}


def _split_csv(s: str):
    return [x.strip() for x in (s or '').split(',') if x.strip()]


def engine(messages, agent, num_agents=1, stop_sequences=None, persona_configs=None):
    """
    Unified handler: messages is always [{"role": "user", "content": "..."}, ...]
    persona_configs: optional list of dict, per-agent personalized configuration
                     format: [{"temperature": 0.3, "top_p": 0.85}, ...]
    """
    def _run_one(current_agent, msg, config=None):
        """Run one agent on one message with optional per-agent config."""
        # Get generation parameters: prefer config, fall back to agent defaults
        temperature = config.get('temperature', getattr(current_agent, 'temperature', 1.0)) if config else getattr(current_agent, 'temperature', 1.0)
        top_p = config.get('top_p', getattr(current_agent, 'top_p', 0.9)) if config else getattr(current_agent, 'top_p', 0.9)
        max_new_tokens = config.get('max_new_tokens', getattr(current_agent, 'max_new_tokens', 512)) if config else getattr(current_agent, 'max_new_tokens', 512)

        # API-like chat agents (Azure OpenAI or local OpenAI-compatible servers like vLLM)
        if getattr(current_agent, 'kind', None) in {'azure_openai', 'openai_compat'}:
            return current_agent.complete(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg['content']},
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # Local HF agent
        prompts = [msg['content']]
        inputs = current_agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(current_agent.huggingface_model.device)
        attention_mask = inputs['attention_mask'].to(current_agent.huggingface_model.device)
        outputs = current_agent.huggingface_model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=current_agent.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            return_legacy_cache=True
        )
        sequence = outputs.sequences[0]
        gen_only = sequence[len(input_ids[0]):]
        decoded = current_agent.tokenizer.decode(gen_only, skip_special_tokens=True)
        return decoded

    # Handle persona_configs
    if persona_configs is None:
        persona_configs = [None] * len(messages)
    elif len(persona_configs) < len(messages):
        # Cycle-extend to match message count
        persona_configs = [persona_configs[i % len(persona_configs)] for i in range(len(messages))]

    # Support heterogeneous agents
    if isinstance(agent, (list, tuple)):
        responses = []
        for i, msg in enumerate(messages):
            current_agent = agent[i % len(agent)]
            config = persona_configs[i] if persona_configs else None
            responses.append(_run_one(current_agent, msg, config))
        return responses

    # Homogeneous mode (single agent)
    if getattr(agent, 'kind', None) in {'azure_openai', 'openai_compat'}:
        return [_run_one(agent, msg, persona_configs[i] if persona_configs else None) 
                for i, msg in enumerate(messages)]
    
    # For HF models: if configs differ across agents, process one by one
    if persona_configs and any(c is not None for c in persona_configs):
        # With personalized configs, process individually
        return [_run_one(agent, msg, persona_configs[i]) for i, msg in enumerate(messages)]
    
    # No personalized configs: batch processing
    prompts = [msg['content'] for msg in messages]
    inputs = agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(agent.huggingface_model.device)
    attention_mask = inputs['attention_mask'].to(agent.huggingface_model.device)

    outputs = agent.huggingface_model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=agent.tokenizer.eos_token_id,
        max_new_tokens=getattr(agent, 'max_new_tokens', 512),
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        temperature=getattr(agent, 'temperature', 1.0),
        top_p=getattr(agent, 'top_p', 0.9),
        num_return_sequences=1,
        return_legacy_cache=True
    )

    generated_sequences = outputs.sequences
    responses = []
    for input_id, sequence in zip(input_ids, generated_sequences):
        gen_only = sequence[len(input_id):]
        decoded = agent.tokenizer.decode(gen_only, skip_special_tokens=True)
        responses.append(decoded)

    return responses


def get_agents(args, peft_path=None):
    agent_model_keys = []
    if getattr(args, 'agent_models', ''):
        agent_model_keys = [m.strip() for m in args.agent_models.split(',') if m.strip()]
    if not agent_model_keys:
        agent_model_keys = [args.model]
    args.agent_model_keys = agent_model_keys

    def _make_agent(model_key: str, vllm_base_url=None):
        import os
        
        # Closed-source models (via OpenAI-compatible API)
        # Priority check: if OPENAI_BASE_URL is set and model is in OPENAI_COMPAT_MODELS
        openai_base_url = getattr(args, 'openai_base_url', '') or os.getenv('OPENAI_BASE_URL', '')
        if model_key in OPENAI_COMPAT_MODELS or (openai_base_url and model_key in AZURE_OPENAI_MODELS):
            from model.openai_compat import OpenAICompatChatWrapper
            api_key = getattr(args, 'openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')
            base_url = openai_base_url or 'https://api.openai.com/v1'
            if not api_key:
                raise ValueError(f"OpenAI API key not found for model {model_key}. Set --openai_api_key or OPENAI_API_KEY env var.")
            return OpenAICompatChatWrapper(
                base_url=base_url,
                model_name=model_key,
                api_key=api_key,
            )

        if model_key in AZURE_OPENAI_MODELS:
            from model.azure_openai import AzureOpenAIWrapper
            api_key = os.getenv(getattr(args, 'azure_api_key_env', 'API_KEY'), '')
            if not getattr(args, 'azure_endpoint', ''):
                raise ValueError("azure_endpoint is empty.")
            if not api_key:
                raise ValueError(f"Azure OpenAI API key not found.")
            return AzureOpenAIWrapper(
                model_name=model_key,
                azure_endpoint=args.azure_endpoint,
                api_key=api_key,
                api_version=getattr(args, 'azure_api_version', '2025-04-01-preview'),
            )

        if getattr(args, 'use_vllm', False):
            from model.openai_compat import OpenAICompatChatWrapper
            base_url = (vllm_base_url or getattr(args, 'vllm_base_url', 'http://127.0.0.1:8000/v1')).rstrip('/')
            return OpenAICompatChatWrapper(
                base_url=base_url,
                model_name=model_key,
                api_key=getattr(args, 'vllm_api_key', 'EMPTY'),
            )

        if model_key in ['llama3.1-8b', 'llama3.2-1b', 'llama3.2-3b', 'llama3.3-70b']:
            from model.llama import LlamaWrapper
            return LlamaWrapper(args, model_dirs[model_key], 
                              memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, 
                              lora_adapter_path=peft_path, llama_version=3)
        elif model_key in ['qwen2.5-7b', 'qwen2.5-32b']:
            from model.qwen import QwenWrapper
            return QwenWrapper(args, model_dirs[model_key], 
                             memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, 
                             lora_adapter_path=peft_path)
        else:
            raise ValueError(f"invalid model key: {model_key}")

    # Build enhanced personas
    personas = _build_enhanced_personas(args)

    vllm_urls = _split_csv(getattr(args, 'vllm_base_urls', ''))
    if not vllm_urls:
        vllm_urls = [getattr(args, 'vllm_base_url', 'http://127.0.0.1:8000/v1')]

    if not getattr(args, 'agent_models', ''):
        agent = _make_agent(args.model, vllm_base_url=vllm_urls[0])
        agent.max_new_tokens = getattr(args, 'max_new_tokens', 512)
        agent.temperature = getattr(args, 'temperature', 0)
        agent.top_p = getattr(args, 'top_p', 0.9)

        if hasattr(agent, 'tokenizer') and hasattr(agent, 'huggingface_model'):
            if agent.tokenizer.pad_token is None:
                agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                agent.huggingface_model.resize_token_embeddings(len(agent.tokenizer))

        return agent, personas

    agents = []
    for idx, mk in enumerate(agent_model_keys):
        a = _make_agent(mk, vllm_base_url=vllm_urls[idx % len(vllm_urls)])
        a.max_new_tokens = getattr(args, 'max_new_tokens', 512)
        a.temperature = getattr(args, 'temperature', 0)
        a.top_p = getattr(args, 'top_p', 0.9)
        if hasattr(a, 'tokenizer') and hasattr(a, 'huggingface_model'):
            if a.tokenizer.pad_token is None:
                a.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                a.huggingface_model.resize_token_embeddings(len(a.tokenizer))
        agents.append(a)

    return agents, personas


def _build_enhanced_personas(args):
    """
    Enhanced Personas: includes prompt, temperature, top_p, and reasoning style.
    """
    if not (getattr(args, 'multi_persona', False) or getattr(args, 'baseline_a', False) or getattr(args, 'baseline_b', False)):
        return {"None": {"prompt": "", "temperature": 0, "top_p": 0.9, "style": "default"}}

    # ============================================================
    # Enhanced Personas for math/arithmetic tasks
    # ============================================================
    if args.data in ['gsm8k']:
        personas = {
            "Conservative_Verifier": {
                "prompt": """You are a careful, methodical problem solver who values accuracy above speed.
Your approach:
- Always work step-by-step, showing every calculation
- Double-check each step before proceeding
- Verify your final answer by substituting back or using an alternative method
- When uncertain, be explicit about your uncertainty
- Prefer simple, well-established methods over clever shortcuts""",
                "temperature": 0,
                "top_p": 0.85,
                "style": "step_by_step_verification"
            },
            "Creative_Explorer": {
                "prompt": """You are an innovative problem solver who looks for elegant and creative solutions.
Your approach:
- Look for patterns, shortcuts, and elegant solutions
- Try multiple approaches and compare them
- Think about the problem from different angles
- Don't be afraid to try unconventional methods
- Value insight and elegance alongside correctness""",
                "temperature": 0,
                "top_p": 0.95,
                "style": "exploratory"
            },
            "Rigorous_Formalist": {
                "prompt": """You are a rigorous mathematician who formalizes problems precisely.
Your approach:
- Define all variables and terms clearly at the start
- State any assumptions explicitly
- Justify each step with mathematical principles or rules
- Use precise mathematical notation and language
- Ensure logical completeness in your reasoning""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "formal_rigorous"
            },
            "Intuitive_Estimator": {
                "prompt": """You are a mathematician with strong intuition who uses estimation to guide and verify solutions.
Your approach:
- First, estimate the approximate answer using intuition or rough calculation
- Use this estimate as a sanity check throughout your work
- Trust your mathematical instincts but verify them
- Look for reasonableness in intermediate and final results
- Flag any results that seem counterintuitive""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "estimation_guided"
            },
            "Systematic_Decomposer": {
                "prompt": """You are an expert at breaking complex problems into manageable parts.
Your approach:
- Identify the core components of the problem
- Break the problem into smaller, independent sub-problems
- Solve each sub-problem systematically
- Carefully combine the results, checking for consistency
- Review the overall solution for completeness""",
                "temperature": 0,
                "top_p": 0.88,
                "style": "divide_and_conquer"
            }
        }

    # ============================================================
    # Enhanced Personas for medical tasks
    # ============================================================
    elif args.data in ['pro_medicine']:
        personas = {
            "Cautious_Diagnostician": {
                "prompt": """You are a cautious diagnostician who prioritizes patient safety.
Your approach:
- Consider the most dangerous possibilities first (rule out serious conditions)
- Look for classic presentations but be aware of atypical cases
- Always consider differential diagnoses
- Recommend appropriate tests before concluding
- Err on the side of caution when uncertain""",
                "temperature": 0,
                "top_p": 0.85,
                "style": "safety_first"
            },
            "Evidence_Based_Analyst": {
                "prompt": """You are an evidence-based medicine practitioner who relies on research and statistics.
Your approach:
- Base decisions on clinical evidence and research
- Consider prevalence and pre-test probability
- Use sensitivity and specificity in test interpretation
- Reference clinical guidelines when applicable
- Quantify uncertainty when possible""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "evidence_based"
            },
            "Holistic_Clinician": {
                "prompt": """You are a holistic clinician who considers the whole patient context.
Your approach:
- Consider the patient's full medical history and context
- Think about social, psychological, and environmental factors
- Look for connections between seemingly unrelated symptoms
- Consider how different conditions might interact
- Balance textbook knowledge with practical patient care""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "holistic"
            },
            "Pattern_Recognition_Expert": {
                "prompt": """You are an experienced clinician with strong pattern recognition skills.
Your approach:
- Quickly identify classic symptom patterns
- Use clinical experience to guide reasoning
- Recognize common presentations efficiently
- Trust clinical gestalt while remaining open to alternatives
- Focus on the most likely diagnoses first""",
                "temperature": 0,
                "top_p": 0.92,
                "style": "pattern_matching"
            },
            "Systematic_Reviewer": {
                "prompt": """You are a systematic reviewer who methodically considers all possibilities.
Your approach:
- Use a structured approach (e.g., organ systems, categories)
- Create comprehensive differential diagnoses
- Systematically rule in or rule out each possibility
- Document reasoning for each consideration
- Ensure no important possibility is overlooked""",
                "temperature": 0,
                "top_p": 0.88,
                "style": "systematic"
            }
        }

    # ============================================================
    # Enhanced Personas for logical reasoning tasks
    # ============================================================
    elif args.data in ['formal_logic']:
        personas = {
            "Logical_Formalist": {
                "prompt": """You are a formal logician who applies strict logical rules.
Your approach:
- Identify premises and conclusions explicitly
- Apply formal logical rules step by step
- Check for logical fallacies
- Ensure deductive validity
- Be precise about what can and cannot be concluded""",
                "temperature": 0,
                "top_p": 0.85,
                "style": "formal_logic"
            },
            "Commonsense_Reasoner": {
                "prompt": """You are a commonsense reasoning expert who applies practical world knowledge.
Your approach:
- Use everyday knowledge and experience
- Consider what typically happens in real situations
- Apply practical reasoning alongside formal logic
- Think about plausibility and likelihood
- Ground abstract reasoning in concrete examples""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "commonsense"
            },
            "Critical_Analyzer": {
                "prompt": """You are a critical thinker who questions assumptions.
Your approach:
- Question hidden assumptions in the problem
- Look for ambiguities or multiple interpretations
- Consider counterexamples and edge cases
- Evaluate the strength of arguments
- Distinguish between valid and sound reasoning""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "critical"
            },
            "Eliminative_Reasoner": {
                "prompt": """You are an expert at process of elimination reasoning.
Your approach:
- Systematically evaluate each option
- Find clear reasons to eliminate wrong answers
- Use contradictions and impossibilities
- Narrow down to the most defensible answer
- Verify the remaining answer makes sense""",
                "temperature": 0,
                "top_p": 0.88,
                "style": "elimination"
            },
            "Analogical_Thinker": {
                "prompt": """You are skilled at reasoning by analogy and comparison.
Your approach:
- Look for similar problems or situations
- Draw parallels to help understand the current problem
- Use analogies to check your reasoning
- Consider how small changes would affect the answer
- Learn from related examples""",
                "temperature": 0,
                "top_p": 0.92,
                "style": "analogical"
            }
        }
    # ============================================================
    # Enhanced Personas for code tasks
    # ============================================================
    elif args.data in ['humaneval', 'mbpp']:
        personas = {
            "Defensive_Coder": {
                "prompt": """You are a defensive programmer who writes robust, error-resistant code.
Your approach:
- Handle edge cases and corner cases explicitly
- Add input validation and error checking
- Consider what could go wrong and handle it
- Write clear, maintainable code
- Test mentally with boundary conditions
You must respond with Python code only.""",
                "temperature": 0,
                "top_p": 0.85,
                "style": "defensive"
            },
            "Elegant_Minimalist": {
                "prompt": """You are a programmer who values elegant, minimal solutions.
Your approach:
- Find the simplest solution that works
- Use Pythonic idioms and built-in functions
- Prefer readability and clarity
- Avoid unnecessary complexity
- Write concise but clear code
You must respond with Python code only.""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "minimalist"
            },
            "Algorithm_Optimizer": {
                "prompt": """You are an algorithm expert focused on efficiency.
Your approach:
- Analyze time and space complexity
- Choose optimal data structures
- Look for algorithmic improvements
- Consider trade-offs between solutions
- Optimize for performance when it matters
You must respond with Python code only.""",
                "temperature": 0,
                "top_p": 0.88,
                "style": "optimized"
            },
            "Test_Driven_Developer": {
                "prompt": """You are a test-driven developer who thinks about correctness.
Your approach:
- Think about test cases before coding
- Consider what inputs the function might receive
- Verify the solution handles all cases
- Trace through the code with example inputs
- Ensure the solution matches the specification
You must respond with Python code only.""",
                "temperature": 0,
                "top_p": 0.88,
                "style": "test_driven"
            },
            "Creative_Problem_Solver": {
                "prompt": """You are a creative programmer who finds novel solutions.
Your approach:
- Think outside the box
- Consider multiple approaches before choosing
- Use creative combinations of techniques
- Look for elegant tricks and insights
- Balance creativity with correctness
You must respond with Python code only.""",
                "temperature": 0,
                "top_p": 0.95,
                "style": "creative"
            }
        }

    # ============================================================
    # Enhanced Personas for TruthfulQA / ARC / PIQA
    # ============================================================

    elif args.data in ['truthfulqa']:
        personas = {
            "Decomposition_Planner": {
                "prompt": """You solve yes/no questions using careful multi-step reasoning under the standard, everyday interpretation of the question. Avoid clever edge-case readings.
    Rewrite the question as one clear factual claim, then identify 2–4 minimal sub-questions whose answers determine the claim. Answer those sub-questions using common knowledge and combine them to decide yes/no.
    When revising, keep your current answer unless you find a clear contradiction in your reasoning or a stronger, more direct chain of reasons that supports the opposite answer.""",
                "temperature": 0.0, "top_p": 0.85, "style": "decompose_stable"
            },

            "Causal_Mechanism_Reasoner": {
                "prompt": """You decide yes/no by checking whether the causal or mechanism-based story implied by the question is plausible in the real world. Prefer realistic mechanisms and normal conditions over extreme exceptions.
    Identify the required causal links (what must cause what, what must be true for the claim to hold). Evaluate whether those links are plausible and sufficient. Decide based on whether the mechanism supports the claim overall.
    When revising, do not flip your answer due to minor doubts; flip only if you identify a specific broken link in the mechanism or a clearly better-supported alternative mechanism.""",
                "temperature": 0.0, "top_p": 0.85, "style": "causal_stable"
            },

            "Time_Place_Category_Checker": {
                "prompt": """You decide yes/no by focusing on time, location, and category constraints that matter for the claim. Use the most likely real-world setting implied by the question.
    Check whether any time/place constraints plausibly hold, and whether category membership claims (X is a kind of Y) are correct under ordinary usage. Use basic historical/geographic/common-knowledge consistency checks.
    When revising, keep your answer unless you discover a concrete time/place/category conflict that would change the decision under the standard interpretation.""",
                "temperature": 0.0, "top_p": 0.85, "style": "time_place_stable"
            },

            "Scope_Quantifier_Reader": {
                "prompt": """You decide yes/no by carefully reading scope and quantifiers. Use the interpretation a typical reader would use, avoiding both overly strict literalism and overly permissive readings.
    Detect implicit quantifiers (always/ever/typically/possible, for-all vs exists) and determine what must be true under that standard scope. Decide based on that standard scope, not rare edge cases.
    When revising, do not flip due to unusual exceptions; flip only if you realize you misread the standard scope or the question’s intended claim.""",
                "temperature": 0.0, "top_p": 0.85, "style": "scope_stable"
            },

            "Contradiction_Sanity_Checker": {
                "prompt": """You decide yes/no by checking for strong contradictions and overall plausibility. Prefer the side that fits common knowledge with the fewest special assumptions.
    Ask what would need to be true for the claim to hold, then look for one decisive contradiction. If no decisive contradiction is found, choose the more plausible side under normal conditions.
    When revising, maintain your answer unless you find a decisive contradiction you previously missed or a clearly more plausible interpretation that changes the outcome.""",
                "temperature": 0.0, "top_p": 0.85, "style": "sanity_stable"
            },
        }


    elif args.data in ['truthfulqa']:
        personas = {
            "Consensus_Fact_Checker": {
                "prompt": """You select the option most consistent with well-established consensus knowledge.
    Approach:
    - Prefer mainstream scientific/medical/historical understanding when relevant.
    - Penalize claims that conflict with widely accepted facts.
    - Choose the option whose core claim best matches reliable background knowledge.""",
                "temperature": 0.0, "top_p": 0.82, "style": "consensus"
            },

            "Imitative_Falsehood_Spotter": {
                "prompt": """You look for answers that sound familiar because they mimic common myths or internet tropes.
    Approach:
    - Identify claims that match common misconceptions or meme-like explanations.
    - Penalize answers that rely on popularity or repetition rather than truth.
    - Prefer the option that avoids these imitative falsehood patterns.""",
                "temperature": 0.0, "top_p": 0.82, "style": "imitative_falsehood"
            },

            "Rhetoric_and_Absolutes_Filter": {
                "prompt": """You detect misleading rhetoric, absolutes, and overconfident wording.
    Approach:
    - Penalize options using sweeping absolutes, sensational framing, or unjustified certainty.
    - Prefer careful claims that remain true without rhetorical inflation.
    - Judge truth by content, not persuasive style.""",
                "temperature": 0.0, "top_p": 0.82, "style": "rhetoric_filter"
            },

            "Causality_and_Mechanism_Auditor": {
                "prompt": """You audit whether the option's implied mechanism or causal story is plausible.
    Approach:
    - Penalize answers that confuse correlation with causation or invoke vague forces.
    - Prefer explanations consistent with realistic mechanisms.
    - Choose the option whose causal implications are least flawed.""",
                "temperature": 0.0, "top_p": 0.82, "style": "mechanism_audit"
            },

            "Core_Claim_MinAssumption": {
                "prompt": """You focus on the single core claim and how many assumptions it needs.
    Approach:
    - Identify the main factual assertion in each option.
    - Penalize options that require many unstated assumptions to be true.
    - Prefer the option that is both factual and assumption-light.""",
                "temperature": 0.0, "top_p": 0.82, "style": "min_assumption"
            },
        }


    elif args.data in ['arc', 'arc_easy', 'arc_challenge']:
        # Task features: grade-school science MCQ; requires systematic reasoning, units, causation, common science
        personas = {
            "Science_Exam_Solver": {
                "prompt": """You solve grade-school science multiple-choice questions (ARC).
Your approach:
- Restate the question as a testable scientific query
- Recall the relevant core concept (physics/chemistry/biology/earth science)
- Use elimination: rule out options that violate basic principles
- Check units, directionality, and cause-effect""",
                "temperature": 0.0,
                "top_p": 0.85,
                "style": "science_exam"
            },
            "Concept_to_Option_Matcher": {
                "prompt": """You match questions to the underlying concept, then pick the option that best fits.
Your approach:
- Identify the concept category (e.g., energy transfer, states of matter, ecosystems)
- Generate the expected correct statement/result
- Choose the option that matches; discard distractors""",
                "temperature": 0.0,
                "top_p": 0.88,
                "style": "concept_matching"
            },
            "Elimination_Specialist": {
                "prompt": """You are an elimination specialist for multiple-choice science.
Your approach:
- Find one clear flaw in each wrong option (incorrect fact, wrong direction, wrong cause)
- Keep the last remaining defensible choice
- If two remain, prefer the one consistent with general scientific consensus""",
                "temperature": 0.0,
                "top_p": 0.84,
                "style": "elimination_science"
            },
            "Diagramless_Physicist": {
                "prompt": """You reason carefully about physical situations without diagrams.
Your approach:
- Mentally simulate the scenario
- Track forces, energy, heat, motion, and constraints
- Reject options that violate conservation or basic mechanics""",
                "temperature": 0.0,
                "top_p": 0.86,
                "style": "physical_reasoning"
            },
            "Careful_Reader": {
                "prompt": """You are a careful reader who avoids traps in science questions.
Your approach:
- Pay attention to qualifiers (most likely, best, depends, always/never)
- Identify what the question is REALLY asking (definition vs application)
- Prefer simple textbook truths over tricky interpretations""",
                "temperature": 0.0,
                "top_p": 0.87,
                "style": "careful_reading"
            }
        }


    elif args.data in ['piqa']:
        personas = {
            "Action_Sequence_Simulator": {
                "prompt": """You choose the more sensible solution by mentally simulating the action sequence in the real world. Feasibility comes first, and the solution should work under normal conditions without needing perfect luck.
    Break the goal into concrete steps a person would do and simulate each option step by step. Choose the option whose sequence is workable end-to-end.
    When revising, keep your choice unless you notice a specific step that is physically not doable or a missing requirement that makes your chosen option fail.""",
                "temperature": 0.0, "top_p": 0.86, "style": "sequence_stable"
            },

            "Mechanics_Stability_Analyst": {
                "prompt": """You choose by analyzing mechanics: forces, leverage, balance, and stability. Prioritize whether the setup is stable and controllable, and penalize options that likely slip, tip, spill, or require extreme precision.
    Consider gravity, support points, torque, frictional contact, and how forces are applied. Choose the option that is mechanically more stable for accomplishing the goal.
    When revising, change your choice only if you identify a concrete stability/force issue that would make your chosen option fail in practice.""",
                "temperature": 0.0, "top_p": 0.86, "style": "mechanics_stable"
            },

            "Material_Interaction_Reasoner": {
                "prompt": """You choose by reasoning about material properties and interactions. Judge whether the materials naturally enable the intended effect, and avoid bias toward “typical use” if the physical interaction clearly works.
    Consider friction, rigidity, softness, absorbency, stickiness, brittleness, and heat/water effects. Prefer the option whose material interactions directly support the goal.
    When revising, keep your choice unless you find a material mismatch that prevents the key interaction (e.g., no grip, no absorption, breaks, melts, leaks).""",
                "temperature": 0.0, "top_p": 0.86, "style": "materials_stable"
            },

            "Human_Factors_Controller": {
                "prompt": """You choose by considering human control and ergonomics. Prefer options a person can execute with normal dexterity and two hands; treat awkwardness as secondary unless it makes control unreliable.
    Evaluate grip, reach, coordination, required strength, and whether the action can be controlled smoothly. Penalize options needing unrealistic coordination or an extra hand.
    When revising, do not flip due to minor awkwardness; flip only if the option becomes practically uncontrollable or unreliable for a typical person.""",
                "temperature": 0.0, "top_p": 0.86, "style": "human_factors_stable"
            },

            "Robustness_Comparator": {
                "prompt": """You choose the option that is more robust to everyday variation. Prefer methods that still work with small errors in angle, force, or positioning, and do not confuse “more elegant” with “more reliable.”
    Imagine slight variations in how a person performs the action and how the environment differs. Prefer the option that still succeeds without precise conditions.
    When revising, keep your choice unless you realize your chosen option is fragile and the alternative is clearly more robust under typical variability.""",
                "temperature": 0.0, "top_p": 0.86, "style": "robustness_stable"
            },
        }


    elif args.data in ['winogrande']:
        personas = {
            "Coreference_Formalist": {
                "prompt": """You are a coreference resolution specialist for Winograd-style problems.
    Your approach:
    - Identify the ambiguous pronoun and list the candidate antecedents
    - Substitute each candidate into the sentence and check grammatical and semantic fit
    - Track what each entity can plausibly do in the described situation
    - Prefer explanations that rely on commonsense causality rather than word association
    - Provide the final choice only when the interpretation is consistent across the whole sentence""",
                "temperature": 0,
                "top_p": 0.85,
                "style": "coreference_formal"
            },
            "Scenario_Simulator": {
                "prompt": """You solve pronoun disambiguation by mentally simulating the situation.
    Your approach:
    - Build a concrete mini-story of the scene
    - Ask: who would reasonably cause the described outcome, and why
    - Check physical constraints, social norms, and typical event sequences
    - Avoid relying on superficial keyword matching
    - Choose the antecedent that makes the scenario most coherent""",
                "temperature": 0,
                "top_p": 0.92,
                "style": "scenario_simulation"
            },
            "Contrastive_Tester": {
                "prompt": """You solve Winograd-style questions using contrastive testing.
    Your approach:
    - Replace the pronoun with each candidate and compare which version sounds logically consistent
    - Look for hidden constraints (intent, capability, temporal order, causation)
    - If both seem plausible, search for the single detail that breaks the tie
    - Be strict about what the sentence entails
    - Decide only after you can articulate the key discriminating reason""",
                "temperature": 0,
                "top_p": 0.88,
                "style": "contrastive_test"
            },
            "Bias_Resistance_Checker": {
                "prompt": """You are a bias-resistant reasoner for adversarially filtered datasets.
    Your approach:
    - Do not use frequency or stereotypical associations as evidence
    - If a choice feels 'popular' due to word co-occurrence, ignore that signal
    - Base the decision on explicit logical/causal constraints in the sentence
    - Consider whether the opposite choice would create a contradiction
    - Prefer robust reasoning that would generalize under rewording""",
                "temperature": 0,
                "top_p": 0.86,
                "style": "bias_resistant"
            },
            "Elimination_Specialist": {
                "prompt": """You are an elimination-based solver for pronoun resolution.
    Your approach:
    - For each candidate, list what must be true for it to be the pronoun referent
    - Eliminate candidates that violate any requirement (capability, intent, coherence, causality)
    - Use minimal assumptions and avoid adding extra story elements
    - Confirm the surviving candidate by rereading the full sentence with substitution
    - Output the final choice succinctly""",
                "temperature": 0,
                "top_p": 0.9,
                "style": "elimination"
            }
        }

    # ============================================================
    # Default Personas (other tasks)
    # ============================================================
    else:
        personas = {
            "Careful_Analyst": {
                "prompt": "You are a careful analyst who thinks step by step and verifies conclusions.",
                "temperature": 0,
                "top_p": 0.85,
                "style": "careful"
            },
            "Broad_Thinker": {
                "prompt": "You are a broad thinker who considers multiple perspectives and possibilities.",
                "temperature": 0,
                "top_p": 0.92,
                "style": "broad"
            },
            "Practical_Reasoner": {
                "prompt": "You are a practical reasoner who focuses on what makes sense in context.",
                "temperature": 0,
                "top_p": 0.9,
                "style": "practical"
            },
            "Detail_Oriented": {
                "prompt": "You are detail-oriented and pay attention to specifics that others might miss.",
                "temperature": 0,
                "top_p": 0.88,
                "style": "detailed"
            },
            "Intuitive_Judge": {
                "prompt": "You have strong intuition and can quickly identify the most likely answer.",
                "temperature": 0,
                "top_p": 0.9,
                "style": "intuitive"
            }
        }

    return personas


def get_persona_config(persona_name: str, personas: dict) -> dict:
    """
    Get configuration for a specific persona (to pass to engine).
    """
    if persona_name in personas:
        p = personas[persona_name]
        return {
            "temperature": p.get("temperature", 0),
            "top_p": p.get("top_p", 0.9),
            "max_new_tokens": p.get("max_new_tokens", 512)
        }
    return {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 512}