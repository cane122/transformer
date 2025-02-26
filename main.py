import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from collections import Counter
from model.transformer import Transformer
from embeding.token_embeding import TokenEmbedding

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
        tokens = [self.start_token] + self.tokenizer(data_point) + [self.end_token]
        
        indexed_tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # Truncate first if too long
        if len(indexed_tokens) > self.max_seq_length:
            indexed_tokens = indexed_tokens[:self.max_seq_length-1] + [self.vocab[self.end_token]]
        
        # Pad with actual pad tokens
        padding = [self.vocab['<pad>']] * (self.max_seq_length - len(indexed_tokens))
        indexed_tokens += padding
        
        return torch.tensor(indexed_tokens, dtype=torch.long)

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

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")

    # Define hyperparameters and model parameters
    num_layers = 2
    d_model = 128
    num_heads = 8
    d_ff = 256
    drop_prob = 0
    lr = 0.001
    num_workers = 8  # You can adjust this based on your system's capabilities
    # Load your dataset using the custom DataLoader with multiple workers
    tokenizer = simple_tokenizer  # Replace with your actual tokenizer
    start_token = '<s>'
    end_token = '</s>'
    max_seq_length = 50
    with open("training_set/dummy.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    vocabulary = create_vocabulary(data, tokenizer)
    # After creating the vocabulary in the training script
    with open('vocabulary.txt', 'w') as f:
        f.write(str(vocabulary))  # Save the vocabulary dictionary as a string
    def create_inverse_vocabulary(vocabulary):
        inverse_vocabulary = {value: key for key, value in vocabulary.items()}
        return inverse_vocabulary   
    vocab_invers = create_inverse_vocabulary(vocabulary)
    input_vocab_size = len(vocabulary)
    dataset = CustomDataset(data, tokenizer, max_seq_length, start_token, end_token, vocabulary)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    # Initialize your Transformer model and move it to GPU if available
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, input_vocab_size, max_seq_length, drop_prob, device)
    transformer.to(device)
    # Define the Adam optimizer
    optimizer = Adam(transformer.parameters(), lr=lr)

    # Training loop
    num_epochs = 10000  # Adjust as needed
    for epoch in range(num_epochs):
        transformer.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)  # Move token indices to device

            # Prepare inputs for teacher forcing
            decoder_input = batch[:, :-1]  # Exclude last token
            target = batch[:, 1:].contiguous().view(-1)  # Shift right and flatten

            # Forward pass - modified for encoder-decoder structure
            output = transformer.forward(batch, decoder_input)

            # Reshape output and calculate loss
            output_flattened = output.view(-1, output.size(-1))
            loss = F.cross_entropy(output_flattened, target, ignore_index=vocabulary['<pad>'] )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        transformer.eval()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
        with torch.no_grad():
            test_output = transformer.generate_text("Cat", vocabulary['<s>'], 
                                                   vocabulary['</s>'], vocabulary,
                                                   vocab_invers, 20)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Sample: {test_output}")
        if average_loss < 0.01:
            break

    # Convert start and end tokens to their corresponding indices using vocabulary mapping
    start_token_index = vocabulary['<s>']  # Replace '<s>' with your actual start token
    end_token_index = vocabulary['</s>']  # Replace '</s>' with your actual end token
    # Generate text
    generated_text = transformer.generate_text("</s> Cat",start_token_index, end_token_index, vocabulary, vocab_invers, max_length=50)
    print("Generated Text:", generated_text)
    # Save the model state dict
    torch.save(transformer.state_dict(), 'transformer_weights.pth')
    # Save the entire model
    torch.save(transformer, 'transformer_model.pth')



if __name__ == "__main__":
    main()
