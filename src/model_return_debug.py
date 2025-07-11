import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import joblib
from util import cosine_sim
from regression import TwoWordTensorRegression
from util import API_query_embedding

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

if __name__ == "__main__":
    file = open("data/test_sentences.txt", 'r')

    # Loading embedding models
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load the first model weights
    tensor_function1 = TwoWordTensorRegression(300, 300)
    model_path1 = "adj_weights_on_the_fly.pt"
    tensor_function1.load_state_dict(torch.load(model_path1))
    tensor_function1.eval()

    # Load the second model weights
    tensor_function2 = TwoWordTensorRegression(300, 300)
    model_path2 = "dummy_model.pt"
    tensor_function2.load_state_dict(torch.load(model_path2))
    tensor_function2.eval()

    # Debugging: Check if weights are loaded correctly
    print(f"Sample weights from {model_path1}: {list(tensor_function1.parameters())[0][0][:5]}")
    print(f"Sample weights from {model_path2}: {list(tensor_function2.parameters())[0][0][:5]}")

    # Loading transform model
    pca = joblib.load("data/adj_pca_model.pkl")

    # Compare outputs of the two models
    _ , actual_sentence_embedding1_model1 = API_query_embedding("big guy", pca, model, tensor_function1, pos="adjective")
    _ , actual_sentence_embedding2_model1 = API_query_embedding("large man", pca, model, tensor_function1, pos="adjective")

    _ , actual_sentence_embedding1_model2 = API_query_embedding("big guy", pca, model, tensor_function2, pos="adjective")
    _ , actual_sentence_embedding2_model2 = API_query_embedding("large man", pca, model, tensor_function2, pos="adjective")

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


