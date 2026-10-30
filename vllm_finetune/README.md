# Qwen2.5-VL Fine-tuning for Vehicle Odometer Recognition

`qwen_vl_car_insurance_train.py` fine-tunes Qwen2.5-VL-3B-Instruct to extract information from vehicle odometer images. It uses Unsloth's `FastVisionModel`, LoRA adapters, and supervised fine-tuning with TRL.

## Project files

```text
vllm_finetune/
├── qwen_vl_car_insurance_train.py
├── requirements.txt
├── qwen-vl-train.xlsx
└── images/
    ├── 1-vehicle-odometer-reading.jpg
    └── 2-vehicle-odometer-reading.jpg
```

## Environment and setup

The requirements target **Linux x86_64, Python 3.12, and an NVIDIA CUDA GPU**, such as an AutoDL GPU instance.

```bash
cd 'example_project/vllm_finetune'
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install openpyxl
python -m pip check
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Configure the model path

The script loads the model from this absolute path:

```python
model, tokenizer = FastVisionModel.from_pretrained(
    "/root/autodl-tmp/models/Qwen/Qwen2.5-VL-3B-Instruct",
    use_gradient_checkpointing="unsloth",
)
```

Change the path to the location of your downloaded Qwen2.5-VL-3B-Instruct model.

## Excel training dataset

The script reads the first sheet of `qwen-vl-train.xlsx` using `pandas.read_excel`. The bundled workbook contains two examples with prompts and responses.

| Column | Required | Meaning |
| --- | --- | --- |
| `id` | No | Sample identifier; not used by the training code |
| `prompt` | Yes | User instruction as text |
| `image` | Yes | Path to the corresponding image |
| `response` | Yes | Reference assistant answer as text |

Example prompt:

> You are an auto insurance underwriting expert. Here is an image of a vehicle odometer. Please extract the key information from it.

Example image path:

```text
images/1-vehicle-odometer-reading.jpg
```

Example response structure:

```text
This vehicle odometer image shows the following key information:

1. **Total mileage**: 528,915 kilometers
2. **Current speed**: 0 kilometers per hour
3. **Current time**: 19:18
4. **Current temperature**: +4.5°C
5. **Current gear**: Park (P)
```

Keep the column names exactly as shown. Populate prompts and responses with text and ensure each image path exists.

`convert_excel_to_training_format()` opens each image as RGB and constructs a conversation containing a user text/image message and an assistant text response. Missing image paths and image-opening errors are logged, and those samples are skipped. All successfully loaded images are held in memory.

## Training workflow

1. **Load the model:** Initialize Qwen2.5-VL-3B-Instruct with Unsloth gradient checkpointing.
2. **Configure LoRA:** Enable adapters for vision layers, language layers, attention modules, and MLP modules.
3. **Prepare the dataset:** Load the Excel workbook and convert valid rows into multimodal conversations.
4. **Run baseline inference:** Generate a response for `images/1-vehicle-odometer-reading.jpg` and stream it to the terminal.
5. **Train:** Switch to training mode and run `SFTTrainer` with `UnslothVisionDataCollator`.
6. **Compare and save:** Repeat inference on the same image, then save the adapter and tokenizer/processor assets.

This is supervised fine-tuning on reference responses. It does not use GRPO or reward functions.

## Main configuration

| Setting | Current value |
| --- | --- |
| LoRA rank / alpha | `16` / `16` |
| LoRA dropout | `0` |
| Per-device training batch size | `2` |
| Gradient accumulation steps | `4` |
| Effective batch size on one GPU | `8` examples per optimizer step |
| Training steps | `30` |
| Warmup steps | `5` |
| Learning rate | `2e-4` |
| Optimizer | `adamw_8bit` |
| Weight decay | `0.01` |
| Learning-rate scheduler | `linear` |
| Seed | `3407` |
| Requested maximum sequence length | `2048` |
| Logging interval | Every step |
| External reporting | Disabled (`report_to="none"`) |
| Inference maximum new tokens | `128` |
| Inference temperature / min-p | `1.5` / `0.1` |

## Run

After installing dependencies, configuring the model path, and checking the compatibility notes below, run from this project directory:

```bash
source .venv/bin/activate
python qwen_vl_car_insurance_train.py
```

## Outputs and evaluation limits

| Output | Description |
| --- | --- |
| Terminal | Dataset diagnostics, before/after responses, training logs, runtime, and GPU memory statistics |
| `outputs/` | Trainer output directory; checkpoint creation depends on the trainer's save settings |
| `car_insurance_lora_model/` | LoRA adapter and assets saved by `tokenizer.save_pretrained()` |

The before/after test image is also present in the training workbook. With only two training examples and no validation split, this comparison demonstrates the workflow; it does not measure generalization or establish extraction accuracy. Use separate held-out images for evaluation.