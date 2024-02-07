import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from model.transformer import Transformer
from embeding.token_embeding import TokenEmbedding

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU.")

# Assuming you are using a simple whitespace-based tokenizer for illustration purposes
def simple_tokenizer(text):
    return text.split()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming each line in the text file is a separate data point
        data_point = self.data[index]

        # Tokenize the sequence of words
        tokens = self.tokenizer(data_point)

        # Truncate or pad the tokens to the max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        else:
            tokens = tokens + ['<pad>'] * (self.max_seq_length - len(tokens))

        return tokens

def tokenize_and_pad(tokens, embedding_layer):
    # Convert tokens to embeddings using your embedding layer
    embeddings = embedding_layer(tokens)

    return embeddings

def main():
    # Define hyperparameters and model parameters
    num_layers = 2
    d_model = 128
    num_heads = 4
    d_ff = 256
    input_vocab_size = 273992
    max_seq_length = 50
    drop_prob = 0.4

    # Initialize your Transformer model and move it to GPU if available
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, input_vocab_size, max_seq_length, drop_prob)
    transformer.to(device)

    num_workers = 4  # You can adjust this based on your system's capabilities
    # Load your dataset using the custom DataLoader with multiple workers
    tokenizer = simple_tokenizer  # Replace with your actual tokenizer
    with open("training_set/cats.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    dataset = CustomDataset(data, tokenizer, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    # Initialize your TokenEmbedding with the appropriate vocab_size and d_model
    token_embedding = TokenEmbedding(input_vocab_size, d_model)

    # Define the Adam optimizer
    optimizer = Adam(transformer.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            source_tokens = batch  # No target required for language modeling

            # Convert tokens to embeddings using your TokenEmbedding
            source_embeddings = tokenize_and_pad(source_tokens, token_embedding)

            # Move data to GPU
            source_embeddings = source_embeddings.to(device)

            # Forward pass
            output = transformer.forward(source_embeddings, source_embeddings)  # Target is same as input for language modeling

            # Reshape the target to match the shape expected by CrossEntropyLoss
            target = source_embeddings.view(-1, source_embeddings.size(-1))

            # Calculate Cross-Entropy Loss
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.argmax(dim=-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}")

    # Generate some text
    start_token = '<s>'
    end_token = '</s>'
    generated_text = transformer.generate_text(start_token, end_token, max_length=50)
    print("Generated Text:", generated_text)

    # Print the shape of the output
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
