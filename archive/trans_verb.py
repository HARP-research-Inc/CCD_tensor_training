import torch

class Verb:
    def __init__(self, verb : str, N=50, model_dimension = 768):
        self.name = verb
        self.tensor = torch.randn(N, N, model_dimension)
        self.N = N
        self.model_dimension = model_dimension
    
    def __str__(self):
        return str(self.tensor)

    def update_tensor_at(self, i, j, embedding: torch.tensor):
        if i >= self.N or j >= self.model_dimension:
            raise IndexError("Index out of bounds")
        if embedding.size(1) != self.model_dimension:
            raise ValueError(f"Sentence embedding dimension mismatch\nExpected: {self.model_dimension}\nGot: {embedding.size(0)}")
        self.tensor[i][j] = embedding