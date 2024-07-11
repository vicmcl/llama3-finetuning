# llama3-finetuning

## Introduction

In the fast-paced world of software development, efficient searching and organization of code-related questions are crucial for productivity and innovation. Traditional approaches to tag generation are time-consuming and often rely on manual effort, hindering the ability to quickly find and reuse code. Our solution leverages the power of large language models (LLMs) to automatically generate high-quality tags for code-related questions, revolutionizing the way developers work.

## Business Impact
* **Improved Developer Productivity**:
by automating tag generation, developers can focus on writing code rather than manually tagging questions, leading to increased productivity and faster time-to-market.

* **Enhanced Code Reusability**:
accurate tags enable developers to quickly find and reuse existing code, reducing duplication of effort and improving overall code quality.

* **Better Knowledge Management**:
this solution facilitates the organization of code-related knowledge, making it easier for developers to access and build upon existing expertise.

## Training Details

* **Dataset**: a curated set of code-related questions from Stack Overflow, along with their corresponding tags, available on [Hugging Face](https://huggingface.co/datasets/amaye15/Stack-Overflow-Zero-Shot-Classification)
* **Model**: Llama3-8b from Meta Ai.
* **Training**: single NVIDIA L4 GPU training using the LoRA technique and quantization for efficient computation.
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
