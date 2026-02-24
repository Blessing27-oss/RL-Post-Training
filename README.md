# AI Agents @ Dartmouth College

## Problem Set 1, Part II: Code RL Post-Training

This codebase should implement an RL post-training loop via Tinker (see the problem set and this README for more details).

## Running the Training Script

Two prerequisites are required before running `train.py`:

**1. Set your Tinker API key:**

```bash
export TINKER_API_KEY=<your_key_here>
```

**2. Start the Sandbox Fusion Docker container:**

```bash
docker run -d -p 8080:8080 -v "$(pwd)/sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml" volcengine/sandbox-fusion:server-20250609
```

Then run training (example for a 20-step test run):

```bash
python train.py --max_steps=20 --max_tokens=4096 --batch_size=32 --group_size=8 --eval_every=10 --save_every=10
```

Logs and metrics are written to `~/code-rl-logs/<timestamp>/metrics.jsonl`. The script logs `eval/mean_correct` at step -1 (pre-training baseline) and at the final step, providing a before/after comparison on the test set.

## Sandbox Configuration

I suggest using [Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) as the code execution sandbox. A base configuration is provided in `sandbox_config/local.yaml`. The simplest way to run this sandbox is via `Docker`:

```bash
docker run -it -p 8080:8080 -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml volcengine/sandbox-fusion:server-20250609
```

For configuring its use:

```python
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
SANDBOX_MAX_CONCURRENCY = int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
```

I also provide a number of utilities in `tinker_utils` to get things moving quickly. These include:

- **Module:** `tinker_utils/checkpoint.py` Handles saving and loading training checkpoints with full optimizer state.
- **Module:** `tinker_utils/data.py` has utilities for preparing dataset examples into model prompts.
- **Module:** `tinker_utils/env.py` defines the code generation environment with sandbox execution.
- **Module:** `tinker_utils/log.py` provides a multi-backend logging system supporting JSON files, console output, and Weights & Biases.
- **Module:** `tinker_utils/lcb.py` provides utilities for working with LiveCodeBench dataset format and test execution.
- **Module:** `tinker_utils/renderers.py` has conversation formatting and token rendering for different model architectures.
- **Module:** `tinker_utils/qwen.py` has specialized renderers for Qwen3 models.

The starter code in `train.py` includes the following function signatures, which you should fill out and compose into a functional training loop. This is intended to help scaffold your implementation, and to make us easier to test things.

```python
def should_skip(advantages: list[float]) -> bool:
    # Should we skip this training step?

def compute_advantages(
    rewards: list[float]
) -> list[float]:
    # Compute advantages from rewards


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float
) -> tinker.types.Datum:
    # Make a training datapoint for Tinker


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> None:
    # Run one training step
```

I recommend making good use of `asyncio.gather()`, e.g. to sample multiple results simultaneously. Look at the documentation for Tinker's `async` APIs.
