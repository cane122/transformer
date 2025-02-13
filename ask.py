import torch
from embeding.token_embeding import TokenEmbedding
from model.transformer import Transformer
import torch.nn.functional as F

# Assuming a simple tokenizer like in your previous code
def simple_tokenizer(text):
    return text.split()

# Function to create an inverse vocabulary
def create_inverse_vocabulary(vocabulary):
    return {value: key for key, value in vocabulary.items()}

# Function to load the model and vocabulary
def load_model_and_vocab(model_path, vocab_path, device):
    # Load the model
    transformer = torch.load(model_path)
    transformer.to(device)

    # Load vocabulary (assumes it's stored as a Python dict in a text file)
    with open(vocab_path, 'r') as f:
        vocabulary = eval(f.read())
    return transformer, vocabulary

# Function to generate text using the user's question
def generate_response(model, question, start_token, end_token, tokenizer, vocab, max_length, device):
    model.eval()  # Set the model to evaluation mode

    # Retrieve start and end token indices
    start_token_index = vocab[start_token]
    end_token_index = vocab[end_token]

    # Use the question as the starting phrase (optionally, you can prepend the start token)
    start_phrase = question  # or f"{start_token} {question}" if desired

    # Call generate_text with the starting phrase as a string
    output = model.generate_text(start_phrase, start_token_index, end_token_index, vocab, create_inverse_vocabulary(vocab), max_length)
    
    return output

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")

    # Load the trained model and vocabulary
    model_path = 'transformer_model.pth'  # Path to the saved transformer model
    vocab_path = 'vocabulary.txt'           # Path to the saved vocabulary (as a dictionary)
    
    transformer, vocabulary = load_model_and_vocab(model_path, vocab_path, device)
    
    # Set your tokenizer, special tokens, and max_length
    tokenizer = simple_tokenizer
    start_token = '<s>'
    end_token = '</s>'
    max_length = 50  # Adjust as needed

    print("Model loaded successfully.")
    
    while True:
        # Ask user for input
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Generate the response from the model using the user's question
        response = generate_response(transformer, question, start_token, end_token, tokenizer, vocabulary, max_length, device)
        
        # Print the generated response
        print(f"Response: {response}")
    
if __name__ == "__main__":
    main()
