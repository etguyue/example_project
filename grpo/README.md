# Qwen2.5-7B Reasoning Training with GRPO

This example fine-tunes Qwen2.5-7B-Instruct on GSM8K math questions using Group Relative Policy Optimization (GRPO), Unsloth, and LoRA. It rewards correct answers and structured reasoning output, then saves the LoRA adapter and runs a sample inference with vLLM.

## Files

| File or directory | Purpose |
| --- | --- |
| `Qwen2_5_(7B)_R1_GRPO.py` | Model loading, dataset preparation, rewards, training, and inference |
| `requirements.txt` | Dependency pins resolved for Linux x86_64 and Python 3.12 |
| `gsm8k/main/` | Local training and test Parquet files for the main dataset configuration |
| `gsm8k/socratic/` | Alternative dataset configuration, unused by this script |

## Environment and installation status

The script targets a Linux GPU server such as an AutoDL instance. It recommends an A100/A800 with at least 40 GB of GPU memory.

```bash
cd 'example_project/grpo_example'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Configure model and dataset paths

The script currently uses two absolute AutoDL paths. Edit them before running on another machine.

### Model

In `FastLanguageModel.from_pretrained`, set `model_name` to your local model directory:

```python
model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
```

The directory must contain the model weights, configuration, and tokenizer files expected by the loader. Check the exact directory spelling, including the three underscores in the current path.

### Dataset

`get_gsm8k_questions()` currently loads:

```python
data = load_dataset('/root/autodl-tmp/datasets/gsm8k', 'main')[split]
```

To use the bundled Parquet files independently of the working directory, replace that line with:

```python
from pathlib import Path

dataset_dir = Path(__file__).resolve().parent / "gsm8k" / "main"
data = load_dataset(
    "parquet",
    data_files={
        "train": str(dataset_dir / "train-00000-of-00001.parquet"),
        "test": str(dataset_dir / "test-00000-of-00001.parquet"),
    },
    split=split,
)
```

The script uses the training split. Each row must contain `question` and `answer`; the reference answer is extracted from the text following GSM8K's `####` delimiter. The test split is not evaluated automatically.

## Training workflow

1. Load Qwen2.5-7B-Instruct in 4-bit mode with vLLM fast inference enabled.
2. Attach rank-32 LoRA adapters to the attention and MLP projection layers.
3. Convert GSM8K questions into system/user chat messages and extract reference answers.
4. Generate groups of candidate completions and score them with five reward functions.
5. Optimize the adapters with `GRPOTrainer` for 250 steps.
6. Save the adapter, reload it for inference, and print a response to `Calculate pi.`

The system prompt requests this output structure:

```xml
<reasoning>
Reasoning steps go here.
</reasoning>
<answer>
42
</answer>
```

## Reward functions

| Function | Behavior |
| --- | --- |
| `correctness_reward_func` | Awards `2.0` when the extracted answer exactly matches the reference string; otherwise `0.0` |
| `int_reward_func` | Awards `0.5` when the extracted answer passes `str.isdigit()` |
| `strict_format_reward_func` | Awards `0.5` for matching the strict tag and newline pattern |
| `soft_format_reward_func` | Awards `0.5` for matching the more permissive reasoning/answer tag pattern |
| `xmlcount_reward_func` | Adds `0.125` for each of four expected tag/newline checks and applies trailing-text penalties |

Correctness is an exact string comparison, not mathematical equivalence. The integer reward does not accept negative signs, decimal points, or thousands separators. Format rewards encourage structure but do not verify the reasoning itself.

## Main settings

| Setting | Current value |
| --- | --- |
| Maximum sequence length | `1024` |
| Maximum prompt length | `256` |
| Maximum training completion length | `768` |
| LoRA rank / alpha | `32` / `32` |
| GPU memory utilization | `0.6` |
| Learning rate | `5e-6` |
| Optimizer | `paged_adamw_8bit` |
| Scheduler / warmup ratio | `cosine` / `0.1` |
| Per-device training batch size | `1` |
| Gradient accumulation steps | `1` |
| Generations per question | `6` |
| Training steps / checkpoint interval | `250` / `250` |
| Inference temperature / top-p | `0.8` / `0.95` |

## Run and outputs

After installing the dependencies and addressing the paths and configuration issues below, run from this directory:

```bash
source .venv/bin/activate
python 'Qwen2_5_(7B)_R1_GRPO.py'
```

Quote the filename because it contains parentheses. Running or importing the file immediately loads the model and starts training; it has no `if __name__ == "__main__"` guard.

Outputs are relative to the working directory:

| Output | Contents |
| --- | --- |
| `outputs/` | Trainer checkpoints and associated training artifacts |
| `grpo_saved_lora/` | Saved LoRA adapter; the base model is still needed for inference |
| Terminal | Training logs, sampled questions/reference answers/completions, and the final generated response |
