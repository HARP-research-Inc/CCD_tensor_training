import torch
import gensim.downloader as api
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import json
import joblib  
from util import get_embedding_in_parallel
from torch.multiprocessing import Pool, cpu_count, set_start_method
import time

try:
     set_start_method('spawn', force=True)
except RuntimeError:
    pass

import numpy as np

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def worker(index, data):
    t = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{3*(index%2)}")

    buffer = list()
    noun_embeddings = list()

    i = 0

    for adj, nouns in data.items():
        for noun in nouns:
            sentence = adj + " " + noun
            #print(sentence)
            noun_embedding = get_embedding_in_parallel(noun, model)
            adjective_embedding = get_embedding_in_parallel(adj, model)

            if noun_embedding is None or adjective_embedding is None:
                continue 
            sentence_embedding = model.encode(sentence)

            buffer.append(torch.from_numpy(sentence_embedding))
            noun_embeddings.append((noun_embedding, adjective_embedding))
        
        if i % 5 == 0:
            print(f'Thread {index}, {i}/{len(data)}', "adjectives parsed,", int(time.time() - t), "seconds elapsed")

        i += 1
    
    print(f'Thread {index} completed')
    
    return buffer, noun_embeddings

if __name__ == "__main__":
    t = time.time()

    file_in = open("data/top_adjective.json")
    data = json.load(file_in)

    num_nouns = 50
    num_adjectives = len(data)

    print(num_adjectives)

    print("Number of adjectives: ", num_adjectives)

    #empirical_embeddings = torch.zeros((num_adjectives*num_nouns, 384))
    buffer = list()
    noun_embeddings = list()

    with Pool(processes=32) as p:
        for buf, ne in p.starmap(worker, enumerate([{ k: data[k] for k in keys } for keys in np.array_split(list(data.keys()), 32)])):
            buffer += buf
            noun_embeddings += ne

    empirical_embeddings = torch.stack(buffer)
    print("Shape:", empirical_embeddings.shape)

    os.makedirs("data", exist_ok=True)

    pca = PCA(n_components=300)
    empirical_embeddings = pca.fit_transform(empirical_embeddings)

    # Save the PCA model
    joblib.dump(pca, "data/adj_pca_model.pkl")

    torch.save(empirical_embeddings, "data/adj_empirical_embeddings.pt")
    torch.save(noun_embeddings, "data/adj_dependent_data.pt")

    print(f"Took {time.time() - t} seconds")