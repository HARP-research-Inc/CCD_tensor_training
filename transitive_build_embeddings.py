import torch
import gensim.downloader as api
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import json
import joblib  # For saving the PCA model
from util import get_embedding_in_parallel

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


def build_one_verb(data, verb, BERT_model):
    num_nouns = 50
    num_verbs = len(data)
    pca = PCA(n_components=300)

    print("Number of verbs: ", num_verbs)

    empirical_embeddings = torch.zeros((num_nouns*num_nouns, 384))
    
    s_o_embeddings = list()

    
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
            sentence_embedding = BERT_model.encode(sentence, convert_to_tensor=True)
            #get fasttext s/o embeddings
            subject_embedding = get_embedding_in_parallel(subject)
            object_embedding = get_embedding_in_parallel(object)

            flag = False

            if subject_embedding is None:
                subject_embedding = torch.rand((1, 300))
                flag = True

            if object_embedding is None:
                object_embedding = torch.rand((1, 300))
                flag = True
        
            s_o_embeddings.append((subject_embedding, object_embedding))

            if not flag:
                empirical_embeddings[s_i*num_nouns + o_i] = sentence_embedding
            else: 
                #randomized to prevent mapping nonsense to real data
                empirical_embeddings[s_i*num_nouns + o_i] = torch.rand((1,384))
                
    os.makedirs("data", exist_ok=True)

    empirical_embeddings = pca.fit_transform(empirical_embeddings)

    return pca, empirical_embeddings, s_o_embeddings

def BERT_only_no_PCA(data, verb, BERT_model):
    num_nouns = 50
    num_verbs = len(data)

    print("Number of verbs: ", num_verbs)

    empirical_embeddings = torch.zeros((num_nouns*num_nouns, 384))
    
    s_o_embeddings = list()

    
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
            sentence_embedding = BERT_model.encode(sentence, convert_to_tensor=True)

            subject_embedding =  BERT_model.encode(subject, convert_to_tensor=True)
            object_embedding = BERT_model.encode(object, convert_to_tensor=True)
        
            s_o_embeddings.append((subject_embedding, object_embedding))

            empirical_embeddings[s_i*num_nouns + o_i] = sentence_embedding

    return empirical_embeddings, s_o_embeddings

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