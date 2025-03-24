import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import os
import json
import joblib  # For loading the PCA model

def prepare_out_sample_embeddings(sentences, subjects, objects):
    # Load models
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ft_model = api.load('fasttext-wiki-news-subwords-300')

    # Load PCA model
    pca = joblib.load("data/pca_model.pkl")

    num_sentences = len(sentences)
    empirical_embeddings = torch.zeros((num_sentences, 384))
    s_o_embeddings = list()

    for i, (sentence, subject, object) in enumerate(zip(sentences, subjects, objects)):
        # Get sentence embedding
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)
        # Get FastText subject/object embeddings
        subject_embedding = torch.Tensor(ft_model[subject])
        object_embedding = torch.Tensor(ft_model[object])

        s_o_embeddings.append((subject_embedding, object_embedding))
        empirical_embeddings[i] = sentence_embedding

    # Apply PCA transformation
    empirical_embeddings = pca.transform(empirical_embeddings)

    os.makedirs("data", exist_ok=True)
    torch.save(empirical_embeddings, "data/out_sample_empirical_embeddings.pt")
    torch.save(s_o_embeddings, "data/out_sample_dependent_data.pt")

if __name__ == "__main__":
    # Example sentences, subjects, and objects
    sentences = ["cat strike fish", "dog strike ball"]
    subjects = ["cat", "dog"]
    objects = ["fish", "ball"]

    prepare_out_sample_embeddings(sentences, subjects, objects)
