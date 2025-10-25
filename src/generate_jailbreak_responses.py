import argparse
import json
import random
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from joblib import Parallel, delayed

from strong_reject.evaluate import strongreject_rubric

from src.utils import expand_shortcut_model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-model",
        default="llama3-8b",
        help="Name of target model (see `utils.py` for all the available models).",
    )
    parser.add_argument(
        "--jailbreak-dataset",
        default="data/jailbreaks.json",
        help="JSON-formatted jailbreak dataset to use.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for generation."
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-K for generation.")
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p for generation."
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=10,
        help="Number of sequences to sample from the model.",
    )
    parser.add_argument(
        "--n-judge-jobs",
        type=int,
        default=1000,
        help="Number of concurrent judge threads.",
    )
    parser.add_argument("--output-dir", help="Path to output directory.", required=True)
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    # load the jailbreak dataset
    jailbreaks_dataset = load_dataset("json", data_files=args.jailbreak_dataset)[
        "train"
    ]
    jailbreaks_dataset = jailbreaks_dataset.map(
        lambda example: {
            "jailbreak_prompt_text": example["jailbroken_prompt"][0]["content"]
        }
    )

    # set the sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        n=min(args.num_return_sequences, 100),
    )

    # load the model
    model_name_or_path = expand_shortcut_model_name(args.target_model)
    if isinstance(model_name_or_path, tuple):
        llm = LLM(
            model=model_name_or_path[0],
            tokenizer=model_name_or_path[1],
            dtype=torch.bfloat16,
        )
    else:
        llm = LLM(model=model_name_or_path, dtype=torch.bfloat16)

    # generate responses
    convos = [
        [{"role": "user", "content": s}]
        for s in jailbreaks_dataset["jailbreak_prompt_text"]
    ]

    outputs = llm.chat(convos, sampling_params)
    responses = [[r.text for r in out.outputs] for out in outputs]

    # add responses to data set
    jailbreaks_dataset = jailbreaks_dataset.add_column("response", responses)

    # judge responses
    def judge_fn(example):
        prompt = example["forbidden_prompt"]
        responses = example["response"]
        return [strongreject_rubric(prompt, r)["score"] for r in responses]

    judge_scores = Parallel(n_jobs=args.n_judge_jobs, prefer="threads")(
        delayed(judge_fn)(example) for example in jailbreaks_dataset
    )

    jailbreaks_dataset = jailbreaks_dataset.add_column("score", judge_scores)

    # dump socred responses
    jailbreaks_dataset.to_json(
        f"{args.output_dir}/{args.target_model}-scored_responses.json"
    )
