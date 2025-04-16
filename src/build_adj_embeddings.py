import torch
import gensim.downloader as api
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import json
import joblib  
from util import get_embedding_in_parallel

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

if __name__ == "__main__":

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    file_in = open("data/top_adjective.json")
    data = json.load(file_in)

    num_nouns = 50
    num_adjectives = len(data)
    pca = PCA(n_components=300)

    print(num_adjectives)

    print("Number of adjectives: ", num_adjectives)

    #empirical_embeddings = torch.zeros((num_adjectives*num_nouns, 384))
    buffer = list()
    noun_embeddings = list()

    for i, adj in enumerate(data):
        #print("v_i: ", v_i)
        nouns = data[adj]


        for noun in nouns:
            sentence = adj + " " + noun
            #print(sentence)
            noun_embedding = get_embedding_in_parallel(noun)
            adjective_embedding = get_embedding_in_parallel(adj)

            if noun_embedding is None or adjective_embedding is None:
                continue 
            sentence_embedding = model.encode(sentence)

            buffer.append(torch.from_numpy(sentence_embedding))
            noun_embeddings.append((noun_embedding, adjective_embedding))

    empirical_embeddings = torch.stack(buffer)
    print(empirical_embeddings.shape)

    os.makedirs("data", exist_ok=True)

    empirical_embeddings = pca.fit_transform(empirical_embeddings)

    # Save the PCA model
    joblib.dump(pca, "data/adj_pca_model.pkl")

    torch.save(empirical_embeddings, "data/adj_empirical_embeddings.pt")
    torch.save(noun_embeddings, "data/adj_dependent_data.pt")