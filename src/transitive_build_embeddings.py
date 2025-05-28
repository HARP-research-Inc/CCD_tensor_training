import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import json
import joblib  # For saving the PCA model
from util import get_embedding_in_parallel
import time
import numpy as np

from torch.multiprocessing import Pool, cpu_count, set_start_method

try:
     set_start_method('spawn', force=True)
except RuntimeError:
    pass

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def build_trans(data, ft_model, BERT_model, FT):
    
    num_nouns = 50
    num_verbs = len(data)
    pca = PCA(n_components=300)

    print("Number of verbs: ", num_verbs)

    empirical_embeddings = torch.zeros((num_verbs*num_nouns*num_nouns, 384))
    
    s_o_embeddings = list()

    for v_i, verb in enumerate(data):
        #print("v_i: ", v_i)
        nouns = data[verb]

        subjects = nouns[0]
        objects = nouns[1]        
        
        for s_i, subject in enumerate(subjects, start=0):
            for o_i, object in enumerate(objects, start=0):
                #print(s_i, o_i)
                sentence = subject + " " + verb + " " + object

                #tokenize

                #get sentence embedding
                sentence_embedding = model.encode(sentence, convert_to_tensor=True)
                #get fasttext s/o embeddings
                subject_embedding = torch.Tensor(ft_model[subject])
                object_embedding = torch.Tensor(ft_model[object])
            
                s_o_embeddings.append((subject_embedding, object_embedding))

                print(sentence_embedding.shape)
                empirical_embeddings[v_i*num_nouns*num_nouns + s_i*num_nouns + o_i] = sentence_embedding
                
    os.makedirs("data", exist_ok=True)

    empirical_embeddings = pca.fit_transform(empirical_embeddings)

    return pca, empirical_embeddings, s_o_embeddings

"""def bov_worker(index, subjects, objects):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{3*(index%2)}")
    
    s_o_embeddings = list()

    for s_i, subject in enumerate(subjects):
        for o_i, object in enumerate(objects):
            subject_embedding = get_embedding_in_parallel(subject, model)
            object_embedding = get_embedding_in_parallel(object, model)

            s_o_embeddings.append((subject_embedding, object_embedding))
    
    return s_o_embeddings"""
    

def build_one_verb(data, verb, model, cache={}):
    t = time.time()
    num_nouns = 50
    num_verbs = len(data)
    pca = PCA(n_components=300)

    print("Number of verbs: ", num_verbs)

    empirical_embeddings = torch.zeros((num_nouns*num_nouns, 384))
    
    s_o_embeddings = list()

    nouns = data[verb]

    subjects = nouns[0]
    objects = nouns[1]        
    
    for s_i, subject in enumerate(subjects):
        for o_i, object in enumerate(objects):
            sentence = subject + " " + verb + " " + object

            if subject in cache:
                subject_embedding = cache[subject]
            else:
                subject_embedding = get_embedding_in_parallel(subject, model)
                cache[subject] = subject_embedding

            if object in cache:
                object_embedding = cache[object]
            else:
                object_embedding = get_embedding_in_parallel(object, model)
                cache[object] = object_embedding

            s_o_embeddings.append((subject_embedding, object_embedding))

            #get sentence embedding
            sentence_embedding = get_embedding_in_parallel(sentence, model)

            empirical_embeddings[s_i*num_nouns + o_i] = sentence_embedding
                
    os.makedirs("data", exist_ok=True)

    #empirical_embeddings = pca.fit_transform(empirical_embeddings)

    """with Pool(processes=32) as p:
        for embeddings in p.starmap(bov_worker, [pair + (objects,) for pair in enumerate(np.array_split(subjects, 32))]):
            s_o_embeddings += embeddings"""
    
    print("build_one_verb took", time.time() - t, "seconds")

    return pca, empirical_embeddings, s_o_embeddings

if __name__ == "__main__":

    file_in = open("data/one_verb.json")
    data = json.load(file_in)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    build_one_verb(data, "strike", model)

    # #FastText
    # ft_model = api.load('fasttext-wiki-news-subwords-300')
    
    # pca, empirical_embeddings, s_o_embeddings = build_trans(data, model, ft_model)

    # # Save the PCA model
    # joblib.dump(pca, "data/pca_model.pkl")

    # torch.save(empirical_embeddings, "data/hybrid_empirical_embeddings.pt")
    # torch.save(s_o_embeddings, "data/hybrid_dependent_data.pt")