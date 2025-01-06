import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from collections import Counter
from model.transformer import Transformer
from embedding.token_embedding import TokenEmbedding
from llama8b.pipeline import pipe
from torch.amp import autocast, GradScaler

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 5,
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True,
    'learning_rate': 0.004,
    'max_seq_length': 50,
    'temperature': 5,
    'alpha': 0.5,
}

# Model Configuration
MODEL_CONFIG = {
    'num_layers': 32,
    'd_model': 128,
    'num_heads': 8,
    'd_ff': 256,
    'drop_prob': 0.1,
    'embed_size': 128,
}

# Special Tokens Configuration
SPECIAL_TOKENS = {
    'start_token': '<s>',
    'end_token': '</s>',
    'unk_token': '<unk>',
    'pad_token': '<pad>',
}

def simple_tokenizer(text):
    return text.split()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_seq_length, start_token, end_token, vocabulary):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.start_token = start_token
        self.end_token = end_token
        self.vocab = vocabulary
        self.pad_token_index = self.vocab[SPECIAL_TOKENS['pad_token']]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_point = self.data[index]
        tokens = self.tokenizer(data_point)
        
        # Add start and end tokens if they don't exceed max length
        tokens = tokens[:self.max_seq_length-2]  # Reserve space for special tokens
        indexed_tokens = [self.vocab.get(token, self.vocab[SPECIAL_TOKENS['unk_token']]) for token in tokens]
        
        # Add start and end token indices
        indexed_tokens = ([self.vocab[self.start_token]] + 
                         indexed_tokens + 
                         [self.vocab[self.end_token]])
        
        # Pad to max length
        padding_length = self.max_seq_length - len(indexed_tokens)
        if padding_length > 0:
            indexed_tokens.extend([self.pad_token_index] * padding_length)
        else:
            indexed_tokens = indexed_tokens[:self.max_seq_length]
        
        return torch.tensor(indexed_tokens, dtype=torch.long)

def create_vocabulary(data, tokenizer):
    token_counter = Counter()
    
    for sample in data:
        tokens = tokenizer(sample)
        token_counter.update(tokens)
    
    # Create vocabulary with special tokens first
    vocabulary = {}
    for idx, (token_name, token) in enumerate(SPECIAL_TOKENS.items()):
        vocabulary[token] = idx
    
    # Add remaining tokens
    current_idx = len(SPECIAL_TOKENS)
    for token, count in token_counter.most_common():
        if token not in vocabulary:
            vocabulary[token] = current_idx
            current_idx += 1
    
    return vocabulary

def detokenize(indices, inverse_vocab):
    """
    Convert token indices back to text, skipping special tokens.
    
    Args:
        indices: Tensor of token indices
        inverse_vocab: Dictionary mapping indices to tokens
    
    Returns:
        String of detokenized text
    """
    # Convert special tokens to set for faster lookup
    special_tokens = set(SPECIAL_TOKENS.values())
    
    # Convert indices to tokens, skipping special tokens and invalid indices
    valid_tokens = []
    for idx in indices:
        token_idx = idx.item()
        if token_idx in inverse_vocab:
            token = inverse_vocab[token_idx]
            if token not in special_tokens:
                valid_tokens.append(token)
    
    return " ".join(valid_tokens)

def tokenize_and_embed(tokens, embedding_layer, device):
    """
    Convert tokens to embeddings and create attention mask for padding.
    
    Args:
        tokens: Tensor of token indices [batch_size, seq_length]
        embedding_layer: Token embedding layer
        device: Device to place tensors on
    
    Returns:
        embeddings: Tensor of token embeddings [batch_size, seq_length, embed_dim]
        attention_mask: Tensor indicating valid tokens (1) vs padding (0)
    """
    tokens = tokens.to(device)
    # Get the numerical index for pad token
    pad_idx = vocabulary[SPECIAL_TOKENS['pad_token']]
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (tokens != pad_idx).to(torch.float32).unsqueeze(-1)
    
    # Get embeddings
    embeddings = embedding_layer(tokens)
    
    # Apply attention mask
    masked_embeddings = embeddings * attention_mask
    
    return masked_embeddings, attention_mask

def distillation_loss(student_logits, teacher_logits, embeddings, temperature=5.0, alpha=0.5):
    """
    Compute the knowledge distillation loss between student and teacher models.
    
    Args:
        student_logits: Output logits from the student model [batch_size * seq_length, vocab_size]
        teacher_logits: Output logits from the teacher model [batch_size * seq_length, vocab_size]
        embeddings: Token embeddings [batch_size * seq_length, embedding_dim]
        temperature: Temperature parameter for softening probability distributions
        alpha: Weight for balancing between distillation and embedding loss
    
    Returns:
        Combined loss value combining distillation and embedding similarity
    """
    # Compute softmax with temperature scaling
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Calculate the KL divergence loss
    distillation = F.kl_div(
        student_probs,
        teacher_probs,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Calculate embedding similarity loss using cosine similarity
    embedding_sim = 1 - F.cosine_similarity(
        student_logits,
        embeddings,
        dim=-1
    ).mean()
    
    # Combine the losses
    combined_loss = (alpha * distillation) + ((1 - alpha) * embedding_sim)
    
    return combined_loss

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")
    
    # Load and prepare data
    with open("training_set/cats.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    
    # Create vocabulary and inverse vocabulary
    global vocabulary  # Make vocabulary accessible to tokenize_and_embed
    vocabulary = create_vocabulary(data, simple_tokenizer)
    vocab_inverse = {idx: token for token, idx in vocabulary.items()}
    
    # Verify vocabularies
    print("Vocabulary size:", len(vocabulary))
    print("Special tokens in vocabulary:", {token: vocabulary[token] for token in SPECIAL_TOKENS.values()})
    print("Special tokens in inverse vocabulary:", {idx: token for idx, token in vocab_inverse.items() if token in SPECIAL_TOKENS.values()})
    
    # Verify all special tokens are present
    for token in SPECIAL_TOKENS.values():
        if token not in vocabulary:
            raise ValueError(f"Special token {token} missing from vocabulary")
        idx = vocabulary[token]
        if idx not in vocab_inverse:
            raise ValueError(f"Index {idx} for special token {token} missing from inverse vocabulary")
        if vocab_inverse[idx] != token:
            raise ValueError(f"Vocabulary mismatch for {token}: {vocab_inverse[idx]}")
    
    input_vocab_size = len(vocabulary)
    pad_idx = vocabulary[SPECIAL_TOKENS['pad_token']]  # Get pad token index
    
    dataset = CustomDataset(
        data, 
        simple_tokenizer, 
        TRAINING_CONFIG['max_seq_length'],
        SPECIAL_TOKENS['start_token'],
        SPECIAL_TOKENS['end_token'],
        vocabulary
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory']
    )

    # Initialize models
    transformer = Transformer(
        num_layers=MODEL_CONFIG['num_layers'],
        d_model=MODEL_CONFIG['d_model'],
        num_heads=MODEL_CONFIG['num_heads'],
        d_ff=MODEL_CONFIG['d_ff'],
        input_vocab_size=input_vocab_size,
        target_vocab_size=input_vocab_size,
        max_seq_length=TRAINING_CONFIG['max_seq_length'],
        drop_prob=MODEL_CONFIG['drop_prob'],
        device=device
    ).to(device)

    token_embedding = TokenEmbedding(input_vocab_size, MODEL_CONFIG['embed_size']).to(device)

    # Initialize optimizer and scaler
    optimizer = Adam(list(transformer.parameters()) + list(token_embedding.parameters()), 
                    lr=TRAINING_CONFIG['learning_rate'])
    scaler = GradScaler()

    # Training loop
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        total_loss = 0.0
        transformer.train()
        
        for batch_idx, batch in enumerate(dataloader):
            source_tokens = batch.to(device)
            
            # Get embeddings and attention mask
            source_embeddings, attention_mask = tokenize_and_embed(source_tokens, token_embedding, device)

            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                student_output = transformer(source_tokens, source_tokens)
                tokenized_texts = [detokenize(tokens, vocab_inverse) for tokens in source_tokens]

                try:
                    with torch.no_grad():
                        teacher_output = pipe(
                            tokenized_texts,
                            max_new_tokens=TRAINING_CONFIG['max_seq_length'],
                            num_return_sequences=1,
                            return_tensors='pt',
                            truncation=True
                        )
                        
                        if 'logits' in teacher_output[0]:
                            teacher_logits = teacher_output[0]['logits'].to(device)
                        else:
                            teacher_logits = torch.zeros_like(student_output).to(device)
                except Exception as e:
                    print(f"Error during teacher output generation: {e}. Inputs: {tokenized_texts}")
                    continue


                # Ensure all tensors have the same sequence length
                min_length = min(student_output.size(1), teacher_logits.size(1), source_embeddings.size(1))
                student_output = student_output[:, :min_length, :]
                teacher_logits = teacher_logits[:, :min_length, :]
                source_embeddings = source_embeddings[:, :min_length, :]
                attention_mask = attention_mask[:, :min_length]

                # Apply attention mask to outputs
                student_output = student_output * attention_mask
                teacher_logits = teacher_logits * attention_mask

                # Calculate loss
                loss = distillation_loss(
                    student_output.reshape(-1, student_output.size(-1)),
                    teacher_logits.reshape(-1, teacher_logits.size(-1)),
                    source_embeddings.reshape(-1, source_embeddings.size(-1))
                )

                print(f"Epoch [{epoch + 1}/{TRAINING_CONFIG['num_epochs']}] "
                      f"Batch [{batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{TRAINING_CONFIG['num_epochs']}], Average Loss: {average_loss:.4f}")

    # Save the models
    torch.save({
        'transformer_state_dict': transformer.state_dict(),
        'embedding_state_dict': token_embedding.state_dict(),
        'vocabulary': vocabulary,
        'config': {
            'training': TRAINING_CONFIG,
            'model': MODEL_CONFIG,
            'special_tokens': SPECIAL_TOKENS
        }
    }, 'transformer_checkpoint.pth')

if __name__ == "__main__":
    main()
