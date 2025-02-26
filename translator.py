import torch
import argparse
import json
import os
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model.transformer import Transformer

# Define a simple dataset class for translation data
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, source_vocab, target_vocab):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        # Use the empty string token as the default for unknown tokens.
        source = [self.source_vocab.get(token, self.source_vocab.get('', 0)) 
                  for token in self.source_texts[idx].split()]
        target = [self.target_vocab.get(token, self.target_vocab.get('', 0)) 
                  for token in self.target_texts[idx].split()]
        return torch.tensor(source), torch.tensor(target)

# Collate function to handle batching with padding
def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    
    source_lengths = [len(seq) for seq in source_batch]
    target_lengths = [len(seq) for seq in target_batch]
    
    max_source_len = max(source_lengths)
    max_target_len = max(target_lengths)
    
    padded_source = torch.zeros(len(batch), max_source_len, dtype=torch.long)
    padded_target = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    
    for i, (source, target) in enumerate(zip(source_batch, target_batch)):
        padded_source[i, :len(source)] = source
        padded_target[i, :len(target)] = target
    
    return padded_source, padded_target

def main():
    parser = argparse.ArgumentParser(description='Train or use a translation transformer')
    parser.add_argument('--mode', choices=['train', 'translate'], default='train', 
                        help='Whether to train the model or use it for translation')
    parser.add_argument('--data_path', type=str, required=False, 
                    help='Path to the training/test data JSON file (only required for training)')
    parser.add_argument('--source_vocab_path', type=str, required=True, 
                        help='Path to source language vocabulary JSON')
    parser.add_argument('--target_vocab_path', type=str, required=True, 
                        help='Path to target language vocabulary JSON')
    parser.add_argument('--model_path', type=str, default='transformer_model.pt', 
                        help='Path to save/load the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--text_to_translate', type=str, default='', 
                        help='Text to translate in translate mode')
    # Model architecture parameters
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--max_seq_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='Dropout probability')
    # New argument to choose target language (e.g., "spanish" or "serbian")
    parser.add_argument('--target_language', choices=['spanish', 'serbian'], default='spanish', 
                        help='Target language for translation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies from provided JSON files
    with open(args.source_vocab_path, 'r') as f:
        source_vocab = json.load(f)
    
    with open(args.target_vocab_path, 'r') as f:
        target_vocab = json.load(f)
    
    # Create inverse vocabulary for the target language (for converting indices back to tokens)
    target_vocab_inverse = {v: k for k, v in target_vocab.items()}
    
    # Initialize your Transformer model
    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        input_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        max_seq_length=args.max_seq_length,
        drop_prob=args.drop_prob,
        device=device
    ).to(device)
    
    if args.mode == 'train':
        # Load training data (assumed JSON structure with "source" and "target" keys)
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        
        source_texts = data['source']
        target_texts = data['target']
        
        train_dataset = TranslationDataset(source_texts, target_texts, source_vocab, target_vocab)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        # Use the target language’s padding token index (default to 0 if not found)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=target_vocab.get('<pad>', 0))
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch_idx, (source, target) in enumerate(train_loader):
                source = source.to(device)
                target = target.to(device)
                
                # For teacher forcing, shift target input and output by one token.
                target_input = target[:, :-1]
                target_output = target[:, 1:]
                
                output = model(source, target_input)
                output = output.view(-1, len(target_vocab))
                target_output = target_output.contiguous().view(-1)
                
                loss = criterion(output, target_output)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs} completed, Average Loss: {avg_loss:.4f}")
            torch.save(model.state_dict(), args.model_path)
            print(f"Model saved to {args.model_path}")
            
    elif args.mode == 'translate':
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Model loaded from {args.model_path}")
        else:
            print(f"Error: Model file {args.model_path} not found")
            return
        
        # For generation we use the target vocabulary’s start and end tokens.
        start_token_idx = target_vocab.get('<start>', target_vocab.get('', 0))
        end_token_idx = target_vocab.get('<end>', target_vocab.get('', 0))
        
        text_to_translate = args.text_to_translate if args.text_to_translate else input("Enter text to translate: ")
        
        translation = model.generate_text(
            text_to_translate,
            start_token_idx,
            end_token_idx,
            target_vocab,          # Use the target vocabulary for token lookup
            target_vocab_inverse,  # and its inverse for converting indices back to tokens
            max_length=args.max_seq_length
        )
        
        print(f"Input: {text_to_translate}")
        print(f"Translation: {translation}")

if __name__ == "__main__":
    main()
