import os
import asyncio
import datetime
import logging
import random
import chz
import datasets
import tinker
from typing import cast
from tinker_utils.qwen import Qwen3InstructRenderer
from tinker_utils.data import build_question
from tinker_utils.env import CodeEnv
from tinker_utils.log import setup_logging
from tinker_utils.checkpoint import save_checkpoint
from tinker_utils.lcb import normalize_tests


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = os.path.join(
        os.path.expanduser("~/code-rl-logs"),
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    )
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 128
    group_size: int = 8
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 10  # 0 = disabled
    eval_every: int = 10 # -1 = disabled
    max_tokens: int = 24576
    format_coef: float = 0.1
    reward_timeout: int = 6
    temperature: float = 1.0
    max_steps: int = -1  # -1 = unlimited


def main(config: Config):
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="train")
            ) for name in ("primeintellect", "taco", "lcbv5")
        ]
    )

    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="test")
            ) for name in ("codeforces", "lcbv5")
        ]
    )

    # Set up logging
    ml_logger = setup_logging(config.log_path, config=config)

    # Create Tinker clients
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    # Build renderer (Qwen3-Instruct-2507 → no thinking tags)
    tokenizer = training_client.get_tokenizer()
    renderer = Qwen3InstructRenderer(tokenizer)

    # Initial sampling client (named "init" to avoid collision with step 0 post-update weights)
    sampling_client = training_client.save_weights_and_get_sampling_client(name="init")

    # Run async training loop
    asyncio.run(_training_loop(
        config, training_client, sampling_client, renderer,
        train_dataset, test_dataset, ml_logger
    ))

    # Final checkpoint
    save_checkpoint(training_client, "final", config.log_path, {"step": "final"})
    ml_logger.close()


########################################################################
# Helper functions
########################################################################
def should_skip(advantages: list[float]) -> bool:
    return all(a == 0.0 for a in advantages)


def compute_advantages(rewards: list[float]) -> list[float]:
    mean_reward = sum(rewards) / len(rewards)
    return [r - mean_reward for r in rewards]


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float
) -> tinker.types.Datum:
    n_gen = len(tokens) - ob_len
    # Pad prompt positions with 0.0; generation positions get the actual values
    padded_logprobs = [0.0] * (ob_len - 1) + list(logprobs)        # length L-1
    padded_advantages = [0.0] * (ob_len - 1) + [advantage] * n_gen  # length L-1

    return tinker.types.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=dict(
            target_tokens=tokens[1:],
            logprobs=padded_logprobs,
            advantages=padded_advantages,
        ),
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> None:
    fwd_bwd_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
    optim_future = training_client.optim_step(adam_params)
    fwd_bwd_future.result()
    optim_future.result()


########################################################################
# Async helpers
########################################################################
async def _run_group(example, renderer, sampling_client, sampling_params, config):
    """Sample group_size completions for one problem, run env.step, return results."""
    question = build_question(example)
    if question is None:
        return None, None, None

    raw_tests = example.get("test_cases") or example.get("tests") or []
    metadata = example.get("metadata", {})
    tests = normalize_tests(raw_tests, metadata)
    if not tests:
        return None, None, None

    messages = [{"role": "user", "content": question}]
    observation = renderer.build_generation_prompt(messages)

    # Flatten observation tokens
    ob_tokens = []
    for chunk in observation.chunks:
        if hasattr(chunk, "tokens"):
            ob_tokens.extend(list(chunk.tokens))

    # Sample group_size completions
    sample_result = await sampling_client.sample_async(
        prompt=observation,
        num_samples=config.group_size,
        sampling_params=sampling_params,
    )

    env = CodeEnv(
        problem=question,
        tests=tests,
        renderer=renderer,
        format_coef=config.format_coef,
        reward_timeout=config.reward_timeout,
    )

    # Run env.step concurrently for all completions
    seqs_and_logprobs = [
        (list(seq.tokens), list(seq.logprobs))
        for seq in sample_result.sequences
    ]
    step_tasks = [env.step(gen_tokens) for gen_tokens, _ in seqs_and_logprobs]
    step_results = await asyncio.gather(*step_tasks)

    return seqs_and_logprobs, list(step_results), ob_tokens


async def _run_eval(test_dataset, renderer, sampling_client, sampling_params, config,
                    n_eval=64):
    eval_indices = random.sample(range(len(test_dataset)), min(n_eval, len(test_dataset)))
    tasks = [
        _run_group(test_dataset[i], renderer, sampling_client, sampling_params, config)
        for i in eval_indices
    ]
    results = await asyncio.gather(*tasks)

    correct_scores = []
    for seqs_and_logprobs, step_results, _ in results:
        if seqs_and_logprobs is None:
            continue
        for r in step_results:
            correct_scores.append(r.metrics.get("correct", 0))

    return {
        "eval/mean_correct": sum(correct_scores) / len(correct_scores) if correct_scores else 0.0,
        "eval/n_problems": len(correct_scores),
    }


async def _training_loop(config, training_client, sampling_client, renderer,
                         train_dataset, test_dataset, ml_logger):
    adam_params = tinker.types.AdamParams(learning_rate=config.learning_rate)
    problems_per_step = config.batch_size // config.group_size
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )

    # --- Initial eval (pre-training baseline, logged at step -1) ---
    if config.eval_every > 0:
        logger.info("Running initial eval (pre-training)...")
        eval_results = await _run_eval(
            test_dataset, renderer, sampling_client, sampling_params, config
        )
        ml_logger.log_metrics(eval_results, step=-1)

    step = 0
    dataset_indices = list(range(len(train_dataset)))
    random.shuffle(dataset_indices)
    idx_ptr = 0

    while config.max_steps < 0 or step < config.max_steps:
        # Refill shuffle if needed
        if idx_ptr + problems_per_step > len(dataset_indices):
            random.shuffle(dataset_indices)
            idx_ptr = 0

        batch_examples = [train_dataset[dataset_indices[idx_ptr + i]]
                          for i in range(problems_per_step)]
        idx_ptr += problems_per_step

        # --- Concurrent group rollouts ---
        group_tasks = [
            _run_group(ex, renderer, sampling_client, sampling_params, config)
            for ex in batch_examples
        ]
        group_results = await asyncio.gather(*group_tasks)

        # Collect datums across all groups
        all_datums = []
        total_rewards, total_correct, total_format = [], [], []

        for seqs_and_logprobs, step_results, ob_tokens in group_results:
            if seqs_and_logprobs is None:
                continue  # skipped (bad question or no tests)

            rewards = [r.reward for r in step_results]
            advantages = compute_advantages(rewards)

            if should_skip(advantages):
                continue

            ob_len = len(ob_tokens)
            for (gen_tokens, gen_logprobs), adv, result in zip(
                seqs_and_logprobs, advantages, step_results
            ):
                all_tokens = ob_tokens + gen_tokens
                datum = make_datum(all_tokens, gen_logprobs, ob_len, adv)
                all_datums.append(datum)

            total_rewards.extend(rewards)
            for r in step_results:
                total_correct.append(r.metrics.get("correct", 0))
                total_format.append(r.metrics.get("format", 0))

        if not all_datums:
            continue

        # --- Train ---
        train_step(training_client, all_datums, adam_params)

        # --- Update sampling client ---
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{step}"
        )

        # --- Log ---
        ml_logger.log_metrics({
            "train/mean_reward": sum(total_rewards) / len(total_rewards),
            "train/mean_correct": sum(total_correct) / len(total_correct),
            "train/mean_format": sum(total_format) / len(total_format),
            "train/n_datums": len(all_datums),
        }, step=step)

        # --- Checkpoint ---
        if config.save_every > 0 and step % config.save_every == 0:
            save_checkpoint(training_client, f"step_{step}", config.log_path,
                            {"step": step}, kind="both")

        # --- Eval ---
        if config.eval_every > 0 and step % config.eval_every == 0:
            eval_results = await _run_eval(
                test_dataset, renderer, sampling_client, sampling_params, config
            )
            ml_logger.log_metrics(eval_results, step=step)

        step += 1

    # --- Final eval (post-training, logged at step = total steps completed) ---
    if config.eval_every > 0:
        logger.info("Running final eval (post-training)...")
        eval_results = await _run_eval(
            test_dataset, renderer, sampling_client, sampling_params, config
        )
        ml_logger.log_metrics(eval_results, step=step)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
