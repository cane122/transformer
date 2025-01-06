import torch
from llama8b.pipeline import pipe
from torch import nn
from model.transformer import Transformer
from embedding.token_embedding import TokenEmbedding
from torch.nn.functional import softmax

# Special tokens configuration (same as training time)
SPECIAL_TOKENS = {
    'start_token': '<s>',
    'end_token': '</s>',
    'unk_token': '<unk>',
    'pad_token': '<pad>',
}

# Load the model checkpoint
def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=True)
    transformer = Transformer(
        num_layers=checkpoint['config']['model']['num_layers'],
        d_model=checkpoint['config']['model']['d_model'],
        num_heads=checkpoint['config']['model']['num_heads'],
        d_ff=checkpoint['config']['model']['d_ff'],
        input_vocab_size=len(checkpoint['vocabulary']),
        target_vocab_size=len(checkpoint['vocabulary']),
        max_seq_length=checkpoint['config']['training']['max_seq_length'],
        drop_prob=checkpoint['config']['model']['drop_prob'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    
    token_embedding = TokenEmbedding(len(checkpoint['vocabulary']), checkpoint['config']['model']['embed_size'])
    token_embedding.load_state_dict(checkpoint['embedding_state_dict'])
    
    vocabulary = checkpoint['vocabulary']
    inverse_vocab = {idx: token for token, idx in vocabulary.items()}
    
    return transformer, token_embedding, vocabulary, inverse_vocab

# Tokenization and detokenization
def simple_tokenizer(text, vocab):
    return text.split()

def tokenize_and_embed(tokens, embedding_layer, device, vocabulary):
    """
    Convert tokens to embeddings and create attention mask for padding.
    
    Args:
        tokens: Tensor of token indices [batch_size, seq_length]
        embedding_layer: Token embedding layer
        device: Device to place tensors on
        vocabulary: Dictionary of token-to-index mappings
    
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


def detokenize(indices, inverse_vocab):
    special_tokens = set(SPECIAL_TOKENS.values())
    valid_tokens = [inverse_vocab[idx.item()] for idx in indices if inverse_vocab[idx.item()] not in special_tokens]
    return " ".join(valid_tokens)

# Tokenize question and generate output
def answer_question(question, model, embedding_layer, vocab, inverse_vocab, device):
    tokens = torch.tensor([vocab.get(t, vocab[SPECIAL_TOKENS['unk_token']]) for t in simple_tokenizer(question, vocab)],
                          dtype=torch.long).unsqueeze(0).to(device)
    
    # Add special start and end tokens
    tokens = torch.cat([torch.tensor([[vocab[SPECIAL_TOKENS['start_token']]]], device=device), tokens], dim=1)
    tokens = torch.cat([tokens, torch.tensor([[vocab[SPECIAL_TOKENS['end_token']]]], device=device)], dim=1)

    # Get embeddings and attention mask
    embeddings, attention_mask = tokenize_and_embed(tokens, embedding_layer, device, vocab)
    
    # Generate the response (passing through transformer model)
    model.eval()
    with torch.no_grad():
        output = model(tokens, tokens)
    
    # Detokenize the output to text
    output_indices = output.argmax(dim=-1).squeeze(0)
    response = detokenize(output_indices, inverse_vocab)
    
    return response

# Example to ask questions
def ask_questions():
    # Load model
    model, token_embedding, vocabulary, vocab_inverse = load_model('transformer_checkpoint.pth')
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    model_params = count_parameters(model)
    embedding_params = count_parameters(token_embedding)
    total_params = model_params + embedding_params
    print(f"Model has {total_params / 1e6} million parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    token_embedding.to(device)

    questions = [
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
    
    for question in questions:
        print(f"Question: {question}")
        answer = answer_question(question, model, token_embedding, vocabulary, vocab_inverse, device)
        print(f"Answer: {answer}")
        print("-" * 50)
    

if __name__ == "__main__":
    ask_questions()
