import os
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
import torch

# Suppress parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Clear CUDA cache
torch.cuda.empty_cache()

# Check if CUDA (GPU) is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please check your GPU setup.")

# Model and tokenizer setup
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Set pad and eos token IDs explicitly
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example usage
inputs = tokenizer("Hello, can you say a bit about how LLM's are cool, and how this project that I'm working on is also?", return_tensors="pt", padding=True, truncation=True, max_length=100)
outputs = pipe(
    tokenizer.decode(inputs["input_ids"][0]),
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=50
)

# Cast logits to desired precision (if needed)
logits = torch.tensor(outputs[0]['logits'], dtype=torch.float32) if 'logits' in outputs[0] else None

# Print output
print(outputs[0]["generated_text"])

