import torch
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
                vector = torch.tensor([float(val) for val in parts[1:]])
                embedding_matrix[word] = vector
        return embedding_matrix

    def __call__(self, input_tokens):
         # Initialize the array with zeros
        embeddings = torch.zeros((self.vocab_size, self.d_model))

        for i, token in enumerate(input_tokens):
            token = str(token)
            if token in self.embedding_matrix:
                embeddings[i] = self.embedding_matrix[token]
            # No need to handle out-of-vocabulary tokens explicitly, as they are already set to zeros.

        return embeddings
