import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import joblib

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from regression import FullRankTensorRegression
from util import cosine_sim, get_embedding_in_parallel, API_query_embedding, generate_embedding


if __name__ == "__main__":
    file = open("../data/test_sentences.txt", 'r')

    #loading embedding models
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #ft_model = api.load('fasttext-wiki-news-subwords-300')

    tensor_function = FullRankTensorRegression(300, 300)
    tensor_function.load_state_dict(torch.load("../adj_weights_on_the_fly.pt"))

    tensor_function.eval()
    
    #loading transform model
    pca = joblib.load("../data/adj_pca_model.pkl")

    expected1 , actual_sentence_embedding1 = API_query_embedding("big guy", pca, model, tensor_function, pos="adjective")    
    expected2 , actual_sentence_embedding2 = API_query_embedding("large man", pca, model, tensor_function, pos="adjective")
    expected3 , actual_sentence_embedding3 = API_query_embedding("large woman", pca, model, tensor_function, pos="adjective")
    expected4 , actual_sentence_embedding4 = API_query_embedding("guy big", pca, model, tensor_function, pos="adjective")
    expected5 , actual_sentence_embedding5 = API_query_embedding("fat man", pca, model, tensor_function, pos="adjective")

    print("\nall vs \'big guy\'")
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), actual_sentence_embedding2.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), actual_sentence_embedding3.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), actual_sentence_embedding4.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), actual_sentence_embedding5.detach().numpy()))

    print("\nCONTROL: all vs \'big guy\'")
    print("cosine similarity: ", cosine_sim(expected1.detach().numpy(), expected2.detach().numpy()))
    print("cosine similarity: ", cosine_sim(expected1.detach().numpy(), expected3.detach().numpy()))
    print("cosine similarity: ", cosine_sim(expected1.detach().numpy(), expected4.detach().numpy()))
    print("cosine similarity: ", cosine_sim(expected1.detach().numpy(), expected5.detach().numpy()))  

    print("\nactual vs expected")
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), expected1.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding2.detach().numpy(), expected2.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding3.detach().numpy(), expected3.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding4.detach().numpy(), expected4.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), expected4.detach().numpy()))
