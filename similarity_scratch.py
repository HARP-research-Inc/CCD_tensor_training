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

    # Loading embedding models
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load the first model weights
    tensor_function1 = FullRankTensorRegression(300, 300)
    model_path1 = "adj_weights_on_the_fly.pt"
    tensor_function1.load_state_dict(torch.load(model_path1))
    tensor_function1.eval()

    # Load the second model weights
    tensor_function2 = FullRankTensorRegression(300, 300)
    model_path2 = "dummy_model.pt"
    tensor_function2.load_state_dict(torch.load(model_path2))
    tensor_function2.eval()

    # Debugging: Check if weights are loaded correctly
    print(f"Sample weights from {model_path1}: {list(tensor_function1.parameters())[0][0][:5]}")
    print(f"Sample weights from {model_path2}: {list(tensor_function2.parameters())[0][0][:5]}")

    # Loading transform model
    pca = joblib.load("data/adj_pca_model.pkl")

    # Compare outputs of the two models
    actual_sentence_embedding1_model1, _ = API_query_embedding("big guy", pca, model, tensor_function1, pos="adjective")
    actual_sentence_embedding2_model1, _ = API_query_embedding("fat man", pca, model, tensor_function1, pos="adjective")

    actual_sentence_embedding1_model2, _ = API_query_embedding("big guy", pca, model, tensor_function2, pos="adjective")
    actual_sentence_embedding2_model2, _ = API_query_embedding("fat man", pca, model, tensor_function2, pos="adjective")

    # Print cosine similarities for both models
    print(f"Model 1 ({model_path1}) Cosine similarity: ",
          cosine_sim(actual_sentence_embedding1_model1.detach().numpy(), actual_sentence_embedding2_model1.detach().numpy()))
    print(f"Model 2 ({model_path2}) Cosine similarity: ",
          cosine_sim(actual_sentence_embedding1_model2.detach().numpy(), actual_sentence_embedding2_model2.detach().numpy()))

    # Compare the outputs of the two models
    diff1 = torch.norm(actual_sentence_embedding1_model1 - actual_sentence_embedding1_model2).item()
    diff2 = torch.norm(actual_sentence_embedding2_model1 - actual_sentence_embedding2_model2).item()

    print(f"Difference between model outputs for 'boring evil': {diff1:.6f}")
    print(f"Difference between model outputs for 'boring badness': {diff2:.6f}")



