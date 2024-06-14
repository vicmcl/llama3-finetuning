# llama3-finetuning

## Introduction

Automatically generating tags for code-related questions can greatly facilitate the process of searching and organizing questions on platforms like Stack Overflow. Traditional approaches to tag generation often rely on hand-crafted rules or machine learning models that require extensive feature engineering. However, with the advent of large language models (LLMs), we can leverage their capabilities to generate high-quality tags for code-related questions, thanks to their contextual understanding.

## Training Details

* **Dataset**: A curated set of code-related questions from Stack Overflow, along with their corresponding tags, available on [Hugging Face](https://huggingface.co/datasets/amaye15/Stack-Overflow-Zero-Shot-Classification)
* **Model**: Llama3-8b from Meta Ai.
* **Training**: Single NVIDIA L4 GPU training using the LoRA technique and quantization for efficient computation.
* **Evaluation**: the accuracy of the fine tuned model is measured by counting the number of tags correctly predicted by the model.

## Model Inference

To use the fine-tuned model for tag generation, you can import it from Hugging Face using the UnSloth library for faster inference.

### Usage

#### Install the required dependencies

```bash
pip install unsloth
```

#### Inference

```python
from unsloth import FastLanguageModel

prompt = """Give a list of tags for the input sentence.

### Input:
{}

### Output:
{}"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "vicmcl/llama-3-tagger",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

inputs = tokenizer([prompt.format(input_sentence, "")], return_tensors = "pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens = 128,
    use_cache = True,
    pad_token_id=tokenizer.eos_token_id
)

decoded_outputs = tokenizer.batch_decode(outputs)
```
