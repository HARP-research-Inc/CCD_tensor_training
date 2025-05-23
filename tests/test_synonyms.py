import torch
from sentence_transformers import SentenceTransformer
import joblib

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from util import cosine_sim, get_embedding_in_parallel
from regression import TwoWordTensorRegression
from similarity_demo import API_query_embedding


def synonyms_test(sentences, verb, pca, model, tensor_function, type = "transitive verb"):
    """
    Synonyms test function. Requires 

    Args:
        Sentences: list of sentence vectors
        Verb: 
    """
    sum_sim = 0
    num_samples = len(sentences)

    index = 0
    while(index < num_samples):
        line1 = sentences[index]
        line2 = sentences[index + 1]
        
        obj1 = line1.split(',')[1]
        sub1 = line1.split(',')[0]
        

        sub2 = line2.split(',')[0]
        obj2 = line2.split(',')[1]
        sentence1 = sub1 + " " + verb + " " + obj2
        sentence2 = obj1 + " " + verb + " " + sub2

        print(sentence1, sentence2)

        #compare actual
        _, embedding1 = API_query_embedding(sentence1, pca, model, tensor_function, pos = type)
        _, embedding2 = API_query_embedding(sentence2, pca, model, tensor_function, pos = type)

        #compare expected
        # embedding1, _= API_query_embedding(sentence1, pca, model, tensor_function, pos = type)
        # embedding2, _ = API_query_embedding(sentence2, pca, model, tensor_function, pos = type)

        sum_sim += cosine_sim(embedding1.detach().numpy(), embedding2.detach().numpy())

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

    tensor_function = TwoWordTensorRegression(300, 300)
    tensor_function.load_state_dict(torch.load("../models/adj_weights.pt"))

    tensor_function.eval()
    
    #loading transform model
    pca = joblib.load("../data/adj_pca_model.pkl")

    data = file.readlines()
    print("near synonyms: ", synonyms_test(data, "", pca, model, tensor_function, type="adjective"))
    
    file.close()



