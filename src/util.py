import numpy as np
import torch
from tensorly.decomposition import parafac
import requests
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from transformers import BertTokenizer, BertModel

def get_embedding_in_parallel(word, model, pca=None):
    """"""

    word_embedding = model.encode(word, convert_to_tensor=True)
    word_embedding = word_embedding.cpu().numpy().reshape(1, -1)

    if pca:
        word_embedding = pca.transform(word_embedding)

    return torch.from_numpy(word_embedding)

    """response = requests.get("http://127.0.0.1:8000/embedding/"+word)
    if response.status_code == 200:
        return torch.tensor(response.json()["embedding"][0]).unsqueeze(0)
    else:
        print(f"Error: {response.status_code}, {response.json()}")
        return None"""
        

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
    # split the sentence into words
    words = sentence.split()

    # take the first word: [jack] loves [diane]
    word1 = words[0]
    # if the pos is transitive verb, take the third word, otherwise take the second word
    if pos == "transitive verb":
        # jack loves [diane]
        word2 = words[2]
    else:
        # big [jack]
        word2 = words[1]

    # encode the sentence using BERT
    expected_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    # reshape the embedding to be a 1D tensor
    expected_sentence_embedding = expected_sentence_embedding.cpu().numpy().reshape(1, -1)
    # reduce the dimensionality of the embedding to 300 with PCA
    expected_sentence_embedding = torch.tensor(pca.transform(expected_sentence_embedding))

    #print("BERT embedding shape:", expected_sentence_embedding.shape)
    
    # get the embeddings for the subject and object from the fastText model
    subject_embedding = get_embedding_in_parallel(word1)  # Add batch dimension
    object_embedding = get_embedding_in_parallel(word2)  # Add batch dimension

    # pass the subject and object embeddings to the tensor function, evaluate the tensor function for those embeddings
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