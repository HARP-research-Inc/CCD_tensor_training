import torch
import gensim.downloader as api
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import json
import joblib  # For saving the PCA model



if __name__ == "__main__":

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


    #FastText
    ft_model = api.load('fasttext-wiki-news-subwords-300')

    file_in = open("data/one_verb.json")
    data = json.load(file_in)

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

    # Save the PCA model
    joblib.dump(pca, "data/pca_model.pkl")

    torch.save(empirical_embeddings, "data/hybrid_empirical_embeddings.pt")
    torch.save(s_o_embeddings, "data/hybrid_dependent_data.pt")