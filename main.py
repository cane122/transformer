import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import Transformer
from model.encoder import Encoder
from model.decoder import Decoder
from layers.positionwise_feed_forward import PositionwiseFeedForward

class TokenEmbedding:
    def __init__(self, vocab_size, d_model, embedding_file='5/model.txt'):
        self.embedding_matrix = self.load_embeddings(embedding_file)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def load_embeddings(self, embedding_file):
        embedding_matrix = {}
        with open(embedding_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                word = parts[0]
                vector = torch.tensor([float(val) for val in parts[1:]])
                embedding_matrix[word] = vector
        return embedding_matrix

    def __call__(self, input_tokens):
        # Initialize the array with zeros
        embeddings = torch.zeros((len(input_tokens), self.d_model))

        for i, token in enumerate(input_tokens):
            token = str(token)
            if token in self.embedding_matrix:
                embeddings[i] = self.embedding_matrix[token]
            # No need to handle out-of-vocabulary tokens explicitly, as they are already set to zeros.

        return embeddings

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
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.build_vocab()

    def build_vocab(self):
        # Build vocabulary from the entire dataset
        all_tokens = [token for sequence in self.data for token in self.tokenizer(sequence)]
        self.vocab = set(all_tokens)

        # Add special tokens for padding, start, and end
        special_tokens = ['<pad>', '<s>', '</s>']
        self.vocab.update(special_tokens)

        # Create token-to-index and index-to-token mappings
        self.token_to_index = {token: idx for idx, token in enumerate(self.vocab)}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Assuming each line in the text file is a separate data point
        data_point = self.data[index]

        # Tokenize the sequence of words
        tokens = self.tokenizer(data_point)

        # Create input and target sequences for language modeling
        source_sequence = tokens[:-1]  # Input sequence (excluding the last word)
        target_sequence = tokens[1:]   # Target sequence (excluding the first word)

        return {"source": source_sequence, "target": target_sequence}

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
    target_vocab_size = 273992
    max_seq_length = 50
    drop_prob = 0.4

    # Initialize your Transformer model and move it to GPU if available
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length, drop_prob)
    transformer.to(device)

    num_workers = 4  # You can adjust this based on your system's capabilities
    # Load your dataset using the custom DataLoader with multiple workers
    tokenizer = simple_tokenizer  # Replace with your actual tokenizer
    with open("training_set/cats.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    dataset = CustomDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    # Vocabulary size
    input_vocab_size = len(dataset.token_to_index)
    target_vocab_size = input_vocab_size

    # Initialize your TokenEmbedding with the appropriate vocab_size and d_model
    token_embedding = TokenEmbedding(input_vocab_size, d_model)

    # Define the Adam optimizer
    optimizer = Adam(transformer.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            source_tokens, target_tokens = batch["source"], batch["target"]

            # Convert tokens to embeddings using your TokenEmbedding
            source_embeddings = tokenize_and_pad(source_tokens, token_embedding)
            target_embeddings = tokenize_and_pad(target_tokens, token_embedding)

            # Move data to GPU
            source_embeddings, target_embeddings = source_embeddings.to(device), target_embeddings.to(device)

            # Forward pass
            output = transformer.forward(source, target)

            # Reshape the target to match the shape expected by CrossEntropyLoss
            target = target.view(-1)

            # Calculate Cross-Entropy Loss
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}")



    # Create an instance of the Transformer model
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length, drop_prob)

    # Generate some example input data
    source = torch.randn(64, max_seq_length, d_model)  # Example source data
    target = torch.randn(64, max_seq_length, d_model)  # Example target data

    # Perform a forward pass through the Transformer
    output = transformer.forward(source, target)

    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_length, drop_prob)
    start_token = '<s>'  # Replace with your actual start token index
    end_token = '</s>'      # Replace with your actual end token index
    generated_text = transformer.generate_text(start_token, end_token, max_length=50)
    print("Generated Text:", generated_text)


    # Print the shape of the output (should be of shape [batch_size, max_seq_length, target_vocab_size])
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
