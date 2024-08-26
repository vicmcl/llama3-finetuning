# llama3-finetuning

## Introduction

In the fast-paced world of software development, efficient searching and organization of code-related questions are crucial for productivity and innovation. Traditional approaches to tag generation are time-consuming and often rely on manual effort, hindering the ability to quickly find and reuse code. Our solution leverages the power of large language models (LLMs) to automatically generate high-quality tags for code-related questions, revolutionizing the way developers work.

## Business Impact
* **Question Routing:** Topic modeling helps direct users to the most relevant and accurate answers to their questions by categorizing questions into specific topics or tags.
* **Trend Analysis:** By analyzing the topics and tags associated with questions, topic modeling can identify emerging trends and patterns in the community, allowing for more targeted and relevant content creation.
* **Content Organization:** Topic modeling enables the organization of content into logical categories, making it easier for users to find and navigate relevant information.
* **Improved Search:** By incorporating topic modeling into search algorithms, users can receive more accurate and relevant search results, reducing the time and effort required to find the information they need.
* **Enhanced User Experience:** Topic modeling can also be used to personalize the user experience, recommending relevant questions, answers, and resources to users based on their interests and preferences.

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
