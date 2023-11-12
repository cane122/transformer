import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size, d_model, embedding_file = '5/model.txt'):
        self.embedding_matrix = self.load_embeddings(embedding_file)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def load_embeddings(self, embedding_file):
        embedding_matrix = {}
        with open(embedding_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(val) for val in parts[1:]])
                embedding_matrix[word] = vector
        return embedding_matrix

    def __call__(self, input_tokens):
        embeddings = []
        for token in input_tokens:
            token = str(token)  # Convert token to string, assuming token indices are integers
            if token in self.embedding_matrix:
                embeddings.append(self.embedding_matrix[token])
            else:
                # Handle out-of-vocabulary tokens by using a vector of zeros
                embeddings.append(np.zeros(self.d_model))
        
        # Pad to match the specified vocab size (if needed)
        num_padding = self.vocab_size - len(embeddings)
        if num_padding > 0:
            embeddings.extend([np.zeros(self.d_model)] * num_padding)
        
        return np.array(embeddings)[:self.vocab_size]

# Example usage:
# Create a TokenEmbedding instance with an embedding file and specify vocab size and d_model.
# embedding_file = 'path/to/embedding_file.txt'
# vocab_size = 10000
# d_model = 300  # Adjust the dimensionality as needed.
# token_embedding = TokenEmbedding(embedding_file, vocab_size, d_model)
