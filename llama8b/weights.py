from transformers import AutoModelForCausalLM

# Load the model without specifying the quantization config
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
)

# Do not use model.to('cuda'), the model is already set to the right device

# Retrieve all the weights (parameters) from the model
all_weights = {}

# Iterate over the model's named parameters and store their values in a dictionary
for name, param in model.named_parameters():
    all_weights[name] = param.data.cpu().numpy()  # Convert to CPU and numpy for easier manipulation

# Example: Print the shape of the weights
for name, weights in all_weights.items():
    print(f"Layer: {name} | Weights Shape: {weights.shape}")

# Now you have access to the model's weights stored in the `all_weights` dictionary.
