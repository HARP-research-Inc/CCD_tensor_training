import numpy as np
import torch
from torch import nn
from tensorly.decomposition import parafac
import tensorly as tl
import torch.optim as optim
import requests

class FullRankTensorRegression(nn.Module):
    def __init__(self, noun_dim, sent_dim):
        super().__init__()
        self.sent_dim = sent_dim
        self.noun_dim = noun_dim
        self.V = nn.Parameter(torch.randn(sent_dim, noun_dim, noun_dim) * np.sqrt(2.0 / (noun_dim + sent_dim)))

    def forward(self, s, o):
        Vs_o = torch.einsum('ljk,bj,bk->bl', self.V, s, o)
        return Vs_o

def cp_decompose(tensor, rank):
    cp_tensor = parafac(tensor=tensor, rank=rank)
    
    # Extract factor matrices
    P = cp_tensor.factors[0]  
    Q = cp_tensor.factors[1]  
    R = cp_tensor.factors[2]   

    return P, Q, R

def get_embedding_in_parallel(embedding):
    response = requests.get("http://127.0.0.1:8000/embedding/"+embedding)
    if response.status_code == 200:
        return torch.tensor(response.json()["embedding"][0]).unsqueeze(0)
    else:
        print(f"Error: {response.status_code}, {response.json()}")
        return None

def dummy_function():
    return True

def cosine_sim(A, B):
    return (np.trace(A.T @ B) / (np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')))