from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Step 1: Create a directory to save the model
save_directory = "./gpt2-xl-model"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Step 2: Download GPT-2 XL weights and tokenizer
model_name = "openai-community/gpt2-xl"  # Correct model identifier from Hugging Face

# Load GPT-2 XL model (1.5B parameters)
print(f"Downloading {model_name} model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the tokenizer
print(f"Downloading {model_name} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Save the model and tokenizer locally
print(f"Saving {model_name} model and tokenizer to '{save_directory}'...")
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"{model_name} model and tokenizer downloaded and saved successfully!")
