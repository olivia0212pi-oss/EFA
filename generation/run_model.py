from __future__ import annotations

import argparse

from common.config import load_config
from common.reproducibility import set_seed
from generation.utils import apply_chat_template, chosen_logprobs, make_math_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one small inference smoke test.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--problem", default="What is 12 times 15?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    from vllm import LLM, SamplingParams

    model_config = config["model"]
    generation_config = config["generation"]
    llm = LLM(
        model=model_config["generation"],
        dtype=model_config["dtype"],
        gpu_memory_utilization=float(model_config["gpu_memory_utilization"]),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        seed=int(config["seed"]),
    )
    prompt = apply_chat_template(llm.get_tokenizer(), make_math_prompt(args.problem))
    params = SamplingParams(
        temperature=float(generation_config["temperature"]),
        top_p=float(generation_config["top_p"]),
        max_tokens=int(generation_config["max_tokens"]),
        logprobs=int(generation_config["logprobs"]),
        seed=int(config["seed"]),
    )
    output = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
    logprobs = chosen_logprobs(output.token_ids, output.logprobs)
    print(output.text)
    print(f"\nGenerated tokens: {len(output.token_ids)}")
    if logprobs:
        print(f"Average chosen-token logprob: {sum(logprobs) / len(logprobs):.4f}")


if __name__ == "__main__":
    main()

