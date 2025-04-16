import numpy as np
import torch
from tensorly.decomposition import parafac
import requests
import torch.nn.functional as F

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def get_embedding_in_parallel(word):
    """
    Queries word embeddings from local server with fastText preloaded. 

    """
    response = requests.get("http://127.0.0.1:8000/embedding/"+word)
    if response.status_code == 200:
        return torch.tensor(response.json()["embedding"][0]).unsqueeze(0)
    else:
        print(f"Error: {response.status_code}, {response.json()}")
        return None

def generate_embedding(line, pca, model, ft_model, tensor_function):
    """
    Queries fastText embeddings from FT model loaded in scope.

    Args: 
        


    Returns
    """
    sentence = line.strip("\n").strip(".").lower()
    words = sentence.split()

    subject = words[0]
    object = words[2]

    expected_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    expected_sentence_embedding = expected_sentence_embedding.cpu().numpy().reshape(1, -1)
    expected_sentence_embedding = torch.tensor(pca.transform(expected_sentence_embedding))

    subject_embedding = torch.tensor(ft_model[subject]).unsqueeze(0)  # Add batch dimension
    object_embedding = torch.tensor(ft_model[object]).unsqueeze(0)  # Add batch dimension


    actual_sentence_embedding = tensor_function(subject_embedding, object_embedding)

    # Ensure both embeddings are on the same device and have the same shape
    expected_sentence_embedding = expected_sentence_embedding.to(actual_sentence_embedding.device)
    expected_sentence_embedding = expected_sentence_embedding.view(1, -1)
    actual_sentence_embedding = actual_sentence_embedding.view(1, -1)

    # Normalize embeddings
    expected_sentence_embedding = F.normalize(expected_sentence_embedding, p=2, dim=1)
    actual_sentence_embedding = F.normalize(actual_sentence_embedding, p=2, dim=1)

    return expected_sentence_embedding, actual_sentence_embedding

def API_query_embedding(line, pca, model, tensor_function, pos = "transitive verb"):
    """
    Generates sentence embeddings using get_embedding_in_parallel queries.

    Args:
        line: string sentence
        pca: PCA reduction model
        model: preloaded BERT model
        tensor_function: trained model weights
        pos: part of speech. Defaults to transitive verb

    Returns:
        expected_sentence_embedding: BERT embedding
        actual_sentence_embedding: actual sentence embedding
    """
    sentence = line.strip("\n").strip(".").lower()
    words = sentence.split()

    word1 = words[0]
    if pos == "transitive verb":
        word2 = words[2]
    else:
        word2 = words[1]

    expected_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    expected_sentence_embedding = expected_sentence_embedding.cpu().numpy().reshape(1, -1)
    expected_sentence_embedding = torch.tensor(pca.transform(expected_sentence_embedding))

    #print("BERT embedding shape:", expected_sentence_embedding.shape)
    
    subject_embedding = get_embedding_in_parallel(word1)  # Add batch dimension
    object_embedding = get_embedding_in_parallel(word2)  # Add batch dimension

    actual_sentence_embedding = tensor_function(subject_embedding, object_embedding)

    # Ensure both embeddings are on the same device and have the same shape
    expected_sentence_embedding = expected_sentence_embedding.to(actual_sentence_embedding.device)
    expected_sentence_embedding = expected_sentence_embedding.view(1, -1)
    actual_sentence_embedding = actual_sentence_embedding.view(1, -1)

    # Normalize embeddings
    expected_sentence_embedding = F.normalize(expected_sentence_embedding, p=2, dim=1)
    actual_sentence_embedding = F.normalize(actual_sentence_embedding, p=2, dim=1)

    return expected_sentence_embedding, actual_sentence_embedding

def cp_decompose(tensor, rank):
    cp_tensor = parafac(tensor=tensor, rank=rank)
    
    # Extract factor matrices
    P = cp_tensor.factors[0]  
    Q = cp_tensor.factors[1]  
    R = cp_tensor.factors[2]   

    return P, Q, R

def cosine_sim(A, B):
    return (np.trace(A.T @ B) / (np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')))