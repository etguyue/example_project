# -*- coding: utf-8 -*-
"""
Training an R1 Model with Qwen2.5-7B and GRPO Reinforcement Learning
Course: Hands-on LLM Distillation and Fine-tuning
Purpose: Train the reasoning capabilities of Qwen2.5-7B using GRPO (Group Relative Policy Optimization)
Environment: AutoDL GPU instance; A100/A800 with 40GB+ recommended
Dependencies: pip install unsloth vllm
"""

# ========================================
# Step 1: Load the model (enable fast inference with vLLM)
# ========================================

import unsloth
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024  # Increase for longer reasoning traces
lora_rank = 32  # A higher rank improves model capabilities but slows down training

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,  # Enable fast inference with vLLM
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if GPU memory is insufficient
)


# ========================================
# Step 2: Configure LoRA
# ========================================

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ========================================
# Step 3: Prepare the GSM8K dataset
# ========================================

import re
from datasets import load_dataset, Dataset

# System prompt: Define the reasoning output format
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract the answer from text marked with ####."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    """Load the GSM8K dataset."""
    data = load_dataset('/root/autodl-tmp/datasets/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data


dataset = get_gsm8k_questions()


# ========================================
# Step 4: Define reward functions
# ========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Correctness reward: Check whether the answer is correct (highest weight)."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Integer reward: Check whether the answer is an integer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Strict format reward: Require exact compliance with the XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Soft format reward: Require basic compliance with the XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """Calculate a score for XML tag completeness."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML tag count reward."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ========================================
# Step 5: Train with GRPOTrainer
# ========================================

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,  # Generate 6 candidate answers per question
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()


# ========================================
# Step 6: Test and save the model
# ========================================

# Save LoRA parameters
model.save_lora("grpo_saved_lora")

# Test model inference
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Calculate pi."},
], tokenize=False, add_generation_prompt=True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=2048,
)

output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)


# ========================================
# Model export options (uncomment as needed)
# ========================================

# Save in 16-bit floating-point format
# model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Save in 4-bit integer format
# model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")

# Save only the LoRA adapter
# model.save_pretrained_merged("model", tokenizer, save_method="lora")

# Save in GGUF q4_k_m format
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
