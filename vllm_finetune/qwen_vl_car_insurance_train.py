# -*- coding: utf-8 -*-
"""
Qwen2-VL 3B Vision Model Fine-tuning - Auto Insurance Underwriting Expert

Purpose: Fine-tune Qwen2.5-VL-3B for vehicle odometer recognition using Unsloth
Environment: AutoDL GPU instance
"""

# ========================================
# Step 1: Load the model
# ========================================

import json
import os
from PIL import Image
from unsloth import FastVisionModel
import torch

print("Loading the Qwen2.5-VL-3B model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct",
    use_gradient_checkpointing="unsloth",
)

print("Configuring LoRA parameters...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Fine-tune vision layers
    finetune_language_layers=True,    # Fine-tune language layers
    finetune_attention_modules=True,  # Fine-tune attention modules
    finetune_mlp_modules=True,        # Fine-tune MLP modules
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ========================================
# Step 2: Prepare data (Excel format)
# ========================================

import pandas as pd

print("Loading training data...")

def load_excel_dataset(file_path):
    """Load a dataset from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        print(f"Excel column names: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None


def convert_excel_to_training_format(df):
    """Convert Excel data into the training format."""
    converted_data = []
    
    for idx, row in df.iterrows():
        image_path = row["image"]
        prompt = row["prompt"]
        response = row["response"]
        
        if pd.notna(image_path) and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                conversation = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image", "image": image}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response}
                            ]
                        }
                    ]
                }
                converted_data.append(conversation)
                print(f"Successfully processed sample {idx + 1}: {image_path}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        else:
            print(f"Warning: Image file does not exist or the path is empty: {image_path}")
    
    return converted_data


train_df = load_excel_dataset("qwen-vl-train.xlsx")
if train_df is not None:
    converted_dataset = convert_excel_to_training_format(train_df)
else:
    print("Unable to load the dataset. Exiting.")
    exit()

print(f"Successfully loaded {len(converted_dataset)} training samples")


# ========================================
# Step 3: Test inference before training
# ========================================

print("\nTesting model inference before training...")
FastVisionModel.for_inference(model)

test_image = Image.open("images/1-vehicle-odometer-reading.jpg").convert('RGB')
test_instruction = "You are an auto insurance underwriting expert. Here is an image of a vehicle odometer. Please extract the key information from it."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": test_instruction}
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("Model output before training:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)


# ========================================
# Step 4: Train the model
# ========================================

print("\nStarting model training...")
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
    max_seq_length=2048,
)

# GPU memory information
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Total GPU memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of GPU memory in use.")

# Run training
trainer_stats = trainer.train()

# Training statistics
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)

print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds")
print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
print(f"Peak GPU memory usage: {used_memory} GB")
print(f"GPU memory used for LoRA training: {used_memory_for_lora} GB")
print(f"GPU memory utilization: {used_percentage}%")


# ========================================
# Step 5: Test inference after training
# ========================================

print("\nTesting model inference after training...")
FastVisionModel.for_inference(model)

inputs = tokenizer(
    test_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("Model output after training:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                   use_cache=True, temperature=1.5, min_p=0.1)


# ========================================
# Step 6: Save the model
# ========================================

print("\nSaving the LoRA adapter...")
model.save_pretrained("car_insurance_lora_model")
tokenizer.save_pretrained("car_insurance_lora_model")
print("Training complete! Model saved to the car_insurance_lora_model directory")
