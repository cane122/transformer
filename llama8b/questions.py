from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

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

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define a list of questions about cats
cat_questions = [
    "What are the main differences between domestic cats and wildcats?",
    "How do cats communicate with each other and with humans?",
    "What are the most popular breeds of domestic cats and their characteristics?",
    "Why do cats purr, and what does it signify?",
    "What are some common health issues that cats face as they age?",
    "How do a cat's hunting instincts manifest in their behavior?",
    "What role do cats play in various cultures and mythologies around the world?",
    "What are the benefits of having a cat as a pet compared to other animals?",
    "How can I tell if my cat is happy or stressed?",
    "What are some fun and engaging activities to keep indoor cats entertained?"
]

# Generate responses for each question
cat_responses = []
for question in cat_questions:
    response = pipe(question, max_length=100, num_return_sequences=1)[0]['generated_text']
    cat_responses.append(response)

# Print the responses
for i, response in enumerate(cat_responses):
    print(f"Response to question {i + 1}: {response}")
