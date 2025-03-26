import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import joblib
from util import FullRankTensorRegression, cosine_sim, get_embedding_in_parallel
import torch.nn.functional as F

def generate_embedding(line, pca, model, ft_model, tensor_function):
    sentence = line.strip("\n").strip(".").lower()
    words = sentence.split()

    subject = words[0]
    object = words[2]

    expected_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    expected_sentence_embedding = expected_sentence_embedding.cpu().numpy().reshape(1, -1)
    expected_sentence_embedding = torch.tensor(pca.transform(expected_sentence_embedding))

    #print("BERT embedding shape:", expected_sentence_embedding.shape)
    
    subject_embedding = torch.tensor(ft_model[subject]).unsqueeze(0)  # Add batch dimension
    object_embedding = torch.tensor(ft_model[object]).unsqueeze(0)  # Add batch dimension
    # print(subject_embedding.shape)
    # print(object_embedding.shape)

    # print("FastText embeddings (s/o):", subject_embedding.shape, object_embedding.shape)
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
    sentence = line.strip("\n").strip(".").lower()
    words = sentence.split()

    word1 = words[0]
    if pos == "transitive verb":
        word2 = words[2]
    else:
        word2 = words[1]

    expected_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    expected_sentence_embedding = expected_sentence_embedding.cpu().numpy().reshape(1, -1)
    expected_sentence_embedding = torch.tensor(pca.transform(expected_sentence_embedding))

    #print("BERT embedding shape:", expected_sentence_embedding.shape)
    
    subject_embedding = get_embedding_in_parallel(word1)  # Add batch dimension
    object_embedding = get_embedding_in_parallel(word2)  # Add batch dimension
    # print(subject_embedding.shape)
    # print(object_embedding.shape)

    # print("FastText embeddings (s/o):", subject_embedding.shape, object_embedding.shape)
    actual_sentence_embedding = tensor_function(subject_embedding, object_embedding)

    # Ensure both embeddings are on the same device and have the same shape
    expected_sentence_embedding = expected_sentence_embedding.to(actual_sentence_embedding.device)
    expected_sentence_embedding = expected_sentence_embedding.view(1, -1)
    actual_sentence_embedding = actual_sentence_embedding.view(1, -1)

    # Normalize embeddings
    expected_sentence_embedding = F.normalize(expected_sentence_embedding, p=2, dim=1)
    actual_sentence_embedding = F.normalize(actual_sentence_embedding, p=2, dim=1)

    return expected_sentence_embedding, actual_sentence_embedding

if __name__ == "__main__":
    file = open("data/test_sentences.txt", 'r')

    #loading embedding models
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #ft_model = api.load('fasttext-wiki-news-subwords-300')

    tensor_function = FullRankTensorRegression(300, 300)
    tensor_function.load_state_dict(torch.load("adj_weights_on_the_fly.pt"))

    tensor_function.eval()
    
    #loading transform model
    pca = joblib.load("data/adj_pca_model.pkl")

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
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), expected2.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), expected3.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), expected4.detach().numpy()))
    print("cosine similarity: ", cosine_sim(actual_sentence_embedding1.detach().numpy(), expected4.detach().numpy()))
