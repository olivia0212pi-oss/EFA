from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common.config import load_config
from common.reproducibility import set_seed
from generation.utils import SYSTEM_PROMPT
from sampling.utils import acceptance_probability, should_stop_mcmc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed or adaptive suffix Power Sampling.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--problem", default="What is 12 times 15?")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--output", help="Optional JSON result path.")
    return parser.parse_args()


def _encode_prompt(tokenizer: Any, problem: str, device: Any) -> Any:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this problem step by step:\n{problem}"},
    ]
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{SYSTEM_PROMPT}\n\nProblem: {problem}\n"
    return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)


def _sample_suffix(model: Any, prefix_ids: Any, max_new_tokens: int, temperature: float) -> Any:
    output = model.generate(
        prefix_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        pad_token_id=model.config.eos_token_id,
    )
    return output[:, prefix_ids.shape[1] :]


def _conditional_loglik(model: Any, prefix_ids: Any, suffix_ids: Any, temperature: float) -> float:
    import torch

    if suffix_ids.shape[1] == 0:
        return 0.0
    input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
        start = prefix_ids.shape[1] - 1
        suffix_logits = logits[:, start : start + suffix_ids.shape[1], :] / temperature
        logprobs = torch.log_softmax(suffix_logits, dim=-1)
        selected = logprobs.gather(2, suffix_ids.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    sampling_config = config["power_sampling"]
    model_name = model_config["sampling"]
    dtype = getattr(torch, model_config["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(model_config.get("trust_remote_code", False))
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": "cuda"},
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    ).eval()
    device = next(model.parameters()).device
    prompt_ids = _encode_prompt(tokenizer, args.problem, device)

    max_new_tokens = int(sampling_config["max_new_tokens"])
    temperature = float(sampling_config["temperature"])
    beta = float(sampling_config["beta"])
    proposals = int(sampling_config["proposals"])
    initial = _sample_suffix(model, prompt_ids, max_new_tokens, temperature)
    initial_tokens = int(initial.shape[1])
    anchor_fraction = float(sampling_config["anchor_fraction"])
    if not 0.0 <= anchor_fraction < 1.0:
        raise ValueError("power_sampling.anchor_fraction must be in [0, 1).")
    anchor_tokens = min(int(initial_tokens * anchor_fraction), max(0, initial_tokens - 1))
    fixed_reasoning = initial[:, :anchor_tokens]
    prefix_ids = torch.cat([prompt_ids, fixed_reasoning], dim=1)
    current = initial[:, anchor_tokens:]
    suffix_budget = max(1, max_new_tokens - anchor_tokens)
    current_base = _conditional_loglik(model, prefix_ids, current, 1.0)
    current_q = _conditional_loglik(model, prefix_ids, current, temperature)
    proposal_tokens = 0
    rejected_tokens = 0
    accepted = 0
    history: list[dict[str, Any]] = []
    loglik_history = [current_base]

    for step in range(proposals):
        proposal = _sample_suffix(model, prefix_ids, suffix_budget, temperature)
        generated = int(proposal.shape[1])
        proposal_tokens += generated
        proposal_base = _conditional_loglik(model, prefix_ids, proposal, 1.0)
        proposal_q = _conditional_loglik(model, prefix_ids, proposal, temperature)
        probability = acceptance_probability(
            current_base, proposal_base, current_q, proposal_q, beta
        )
        did_accept = random.random() < probability
        if did_accept:
            current = proposal
            current_base = proposal_base
            current_q = proposal_q
            accepted += 1
        else:
            rejected_tokens += generated
        loglik_history.append(current_base)
        history.append(
            {
                "step": step + 1,
                "proposal_tokens": generated,
                "proposal_base_loglik": proposal_base,
                "accept_probability": probability,
                "accepted": did_accept,
                "state_base_loglik": current_base,
            }
        )
        print(
            f"step={step + 1} accepted={did_accept} p={probability:.3f} "
            f"loglik={current_base:.2f}"
        )
        if args.adaptive and should_stop_mcmc(
            loglik_history,
            min_steps=int(sampling_config["min_steps"]),
            epsilon=float(sampling_config["gain_epsilon"]),
        ):
            break

    final_ids = torch.cat([fixed_reasoning, current], dim=1)
    final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    result = {
        "schema_version": 1,
        "model": model_name,
        "problem": args.problem,
        "method": "adaptive" if args.adaptive else "fixed",
        "final_text": final_text,
        "initial_tokens": initial_tokens,
        "anchor_tokens": anchor_tokens,
        "proposals_attempted": len(history),
        "proposal_tokens": proposal_tokens,
        "total_generated_tokens": initial_tokens + proposal_tokens,
        "rejected_tokens": rejected_tokens,
        "accepted": accepted,
        "acceptance_rate": accepted / len(history) if history else 0.0,
        "beta": beta,
        "temperature": temperature,
        "history": history,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote result to {output_path}")


if __name__ == "__main__":
    main()
