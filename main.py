import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from collections import Counter
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
    def __init__(self, data, tokenizer, max_seq_length, start_token, end_token, vocabulary):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_token = start_token
        self.end_token = end_token
        self.vocab = vocabulary  # Assign the vocabulary to the dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_point = self.data[index]
        tokens = self.tokenizer(data_point)
        
        # Convert tokens to indices using vocabulary
        indexed_tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # Pad or truncate to max_seq_length
        indexed_tokens = indexed_tokens[:self.max_seq_length] + [self.vocab['<pad>']] * (self.max_seq_length - len(indexed_tokens))
        
        return torch.tensor(indexed_tokens, dtype=torch.long)  # Convert tokens to integer tensor



from collections import Counter

# Function to create vocabulary from dataset
def create_vocabulary(data, tokenizer, start_token='<s>', end_token='</s>', unk_token='<unk>', pad_token='<pad>'):
    # Initialize a Counter to count token frequencies
    token_counter = Counter()
    
    # Iterate over the data to tokenize and count tokens
    for sample in data:
        tokens = tokenizer(sample)
        token_counter.update(tokens)
    
    # Ensure that special tokens are in the counter
    special_tokens = [unk_token, pad_token]
    for token in special_tokens:
        if token not in token_counter:
            token_counter[token] = 0
    
    # Create a vocabulary dictionary with indices assigned to tokens
    vocabulary = {start_token: 0, end_token: 1}  # Assign indices to start and end tokens
    index = 2  # Start index for tokens
    for token, count in token_counter.items():
        vocabulary[token] = index
        index += 1
    
    return vocabulary


def tokenize_and_pad(tokens, embedding_layer):
    # Convert tokens to embeddings using your embedding layer
    embeddings = embedding_layer(tokens)

    return embeddings

def main():
    # Define hyperparameters and model parameters
    num_layers = 6
    d_model = 128
    num_heads = 8
    d_ff = 256
    drop_prob = 0.01
    num_workers = 64  # You can adjust this based on your system's capabilities
    # Load your dataset using the custom DataLoader with multiple workers
    tokenizer = simple_tokenizer  # Replace with your actual tokenizer
    start_token = '<s>'
    end_token = '</s>'
    max_seq_length = 50
    with open("training_set/cats.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    vocabulary = create_vocabulary(data, tokenizer)
    input_vocab_size = len(vocabulary)
    dataset = CustomDataset(data, tokenizer, max_seq_length, start_token, end_token, vocabulary)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    # Initialize your Transformer model and move it to GPU if available
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, input_vocab_size, max_seq_length, drop_prob, device)
    transformer.to(device)

    # Initialize your TokenEmbedding with the appropriate vocab_size and d_model
    token_embedding = TokenEmbedding(input_vocab_size, d_model)
    # Define the Adam optimizer
    optimizer = Adam(transformer.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50000  # Adjust as needed
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            source_tokens = batch  # No target required for language modeling

            # Convert tokens to embeddings using your TokenEmbedding
            source_embeddings = tokenize_and_pad(source_tokens, token_embedding)

            # Move data to GPU
            source_embeddings = source_embeddings.to(device)

            # Forward pass
            output = transformer.forward(source_tokens, source_tokens)  # Target is same as input for language modeling

            # Reshape the target to match the shape expected by CrossEntropyLoss
            # Reshape the target to match the shape expected by CrossEntropyLoss
            target = source_embeddings.view(-1, source_embeddings.size(-1))

            # Calculate Cross-Entropy Loss
            output_flattened = output.view(-1, output.size(-1))
            target_flattened = target.argmax(dim=-1)
            
            loss = F.cross_entropy(output_flattened, target_flattened)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss}")

    # Convert start and end tokens to their corresponding indices using vocabulary mapping
    start_token_index = vocabulary['<s>']  # Replace '<s>' with your actual start token
    end_token_index = vocabulary['</s>']  # Replace '</s>' with your actual end token
    def create_inverse_vocabulary(vocabulary):
        inverse_vocabulary = {value: key for key, value in vocabulary.items()}
        return inverse_vocabulary   
    vocab_invers = create_inverse_vocabulary(vocabulary)
    # Generate text
    generated_text = transformer.generate_text("</s> Cat",start_token_index, end_token_index, vocabulary, vocab_invers, max_length=50)
    print("Generated Text:", generated_text)
    # Save the model state dict
    torch.save(transformer.state_dict(), 'transformer_weights.pth')
    # Save the entire model
    torch.save(transformer, 'transformer_model.pth')



if __name__ == "__main__":
    main()
