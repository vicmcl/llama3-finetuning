from fastapi import FastAPI, Body
from unsloth import FastLanguageModel
import uvicorn


def extract_tags(resp):
    return resp[0].split('Output:\n')[-1].split('<')[0].split(', ')


def get_response(model, tokenizer, prompt, input_sentence):
    inputs = tokenizer(
        [prompt.format(input_sentence, "")],
        return_tensors = "pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 128,
        use_cache = True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded_outputs = tokenizer.batch_decode(outputs)
    return decoded_outputs


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

app = FastAPI()

@app.post("/")
def query(query: str = Body(..., media_type="text/plain")):
    response = get_response(model, tokenizer, prompt, query)
    tags = extract_tags(response)
    return {"response": tags}

if __name__ == "__main__":
    uvicorn.run(app)  