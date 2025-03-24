import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import joblib
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from util import FullRankTensorRegression, cosine_sim, get_embedding_in_parallel
from similarity_scratch import API_query_embedding


def synonyms_test(sentences, verb, pca, model, tensor_function):
    sum_sim = 0
    num_samples = len(sentences)

    index = 0
    while(index < num_samples):
        line1 = sentences[index]
        line2 = sentences[index + 1]
        
        sub1 = line1.split(',')[0]
        obj1 = line1.split(',')[1]
        

        sub2 = line2.split(',')[0]
        obj2 = line2.split(',')[1]
        sentence1 = sub1 + " " + verb + " " + obj2
        sentence2 = sub2 + " " + verb + " " + obj1

        expected_sentence_embedding1, _ = API_query_embedding(sentence1, pca, model, tensor_function)
        expected_sentence_embedding2, _ = API_query_embedding(sentence2, pca, model, tensor_function)

        sum_sim += cosine_sim(expected_sentence_embedding1.detach().numpy(), expected_sentence_embedding2.detach().numpy())

        index += 2
    
    return sum_sim / (num_samples / 2)


if __name__ == "__main__":
    
    file = open("data/near_synonyms.csv", 'r')

    #loading embedding models

    print("*****loading BERT model*****")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("*****done loading BERT model*****")

    # print("*****loading FastText model*****")
    # ft_model = api.load('fasttext-wiki-news-subwords-300')
    # print("*****done loading FastText model*****")

    tensor_function = FullRankTensorRegression(300, 300)
    tensor_function.load_state_dict(torch.load("../data/hybrid_weights.pt"))

    tensor_function.eval()
    
    #loading transform model
    pca = joblib.load("../data/pca_model.pkl")

    data = file.readlines()
    print("far synonyms: ", synonyms_test(data, "strike", pca, model, tensor_function))
    
    file.close()



