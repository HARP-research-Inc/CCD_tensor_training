import torch
import tensorly
from tensorly.decomposition import parafac
import numpy as np


def decompose(tensor, rank):
    cp_tensor = parafac(tensor=tensor, rank=rank)
    
    # Extract factor matrices
    P = cp_tensor.factors[0]  
    Q = cp_tensor.factors[1]  
    R = cp_tensor.factors[2]   

    return P, Q, R


if __name__ == "__main__":
    tensorly.set_backend("pytorch")
    embeddings = torch.load("data/empirical_embeddings.pt", weights_only=False)

    for embedding in embeddings:
        #print(embedding)
        P, Q, R = decompose(embedding, 100)
        print(P)
    
    