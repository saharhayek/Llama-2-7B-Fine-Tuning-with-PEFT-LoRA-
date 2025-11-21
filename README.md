# Llama-2-7B-Fine-Tuning-with-PEFT-LoRA-
Fine-tune a 7B Llama-2 chat model on a small instruction-following dataset using PEFT + LoRA for lightweight, memory-efficient training.

Notebook: finalLLM.ipynb
Base model: NousResearch/Llama-2-7b-chat-hf
Dataset: mlabonne/guanaco-llama2-1k
Fine-tuned model name: llama-2-7b-miniguanaco

# What the notebook does:
1. Imports required libraries:
   transformers, peft, datasets, accelerate, bitsandbytes, torch, pandas, numpy.

2. Loads the dataset:
   dataset_name = "mlabonne/guanaco-llama2-1k"
   Dataset contains ~1000 rows of instruction → response pairs.
   Automatically splits into training and evaluation subsets.

3. Loads the base LLM:
   model_name = "NousResearch/Llama-2-7b-chat-hf"
   Loads in 4-bit or 8-bit quantization (QLoRA setup) to reduce GPU memory usage.
   Applies tokenizer settings (padding, EOS tokens, truncation).

4. Applies PEFT + LoRA configuration:
   Sets LoRA parameters such as:
     r = 16
     lora_alpha = 32
     lora_dropout = 0.05
     target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
   Wraps the base model into a PEFT model.

5. Prepares training:
   Defines training arguments:
     output_dir = "llama-2-7b-miniguanaco"
     per_device_train_batch_size = 1 or 2 (depending on GPU)
     gradient_accumulation_steps = 4 or more
     num_train_epochs = 2–3
     lr = 2e-4 (typical LoRA learning rate)
     warmup_steps = small number
     fp16 or bf16 enabled if GPU supports it

6. Starts training:
   Uses HuggingFace Trainer or custom training loop.
   Monitors loss and evaluation loss.
   Saves PEFT adapter weights (not the full 7B model).

7. Saves the fine-tuned model:
   Saved under:
     new_model = "llama-2-7b-miniguanaco"
   Only LoRA adapter weights are stored, making the result small and portable.

8. Inference:
   Loads base model + PEFT adapter for evaluation.
   Uses generate() to produce answers to new instructions.
   Compares outputs before vs after fine-tuning.

Dependencies:
pip install transformers datasets peft accelerate bitsandbytes sentencepiece jupyter

Hardware notes:
- 7B model with LoRA fine-tuning requires ~6–10 GB VRAM with 4-bit quantization.
- If running on CPU, training will be extremely slow.

Expected output:
- A functional fine-tuned model "llama-2-7b-miniguanaco"
- Lower training loss compared to base model
- Improved instruction-following quality on evaluation prompts
- Inference examples demonstrating improved responses

Usage notes:
- Only adapter weights are trained and saved; base model must be loaded together.
- For inference:
   load_peft_model(base_model, adapter_path)
- Dataset can be replaced with any instruction-formatted dataset (alpaca, oasst, etc.)
