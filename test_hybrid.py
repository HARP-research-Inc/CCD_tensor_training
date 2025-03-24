import torch
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import joblib
from util import cosine_sim, FullRankTensorRegression
import torch.nn.functional as F

if __name__ == "__main__":
    
    file = open("data/test_sentences.txt", 'r')

    #loading embedding models
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ft_model = api.load('fasttext-wiki-news-subwords-300')

    tensor_function = FullRankTensorRegression(300, 300)
    tensor_function.load_state_dict(torch.load("data/hybrid_weights.pt"))

    tensor_function.eval()
    
    #loading transform model
    pca = joblib.load("data/pca_model.pkl")

    data = file.readlines()
    total_loss = 0.0
    num_samples = 0

    for line in data:
        #print(line)
        sentence = line.strip("\n").strip(".").lower()
        words = sentence.split()

        subject = words[0]
        object = words[2]

        expected_sentence_embedding = model.encode(sentence, convert_to_tensor=True)
        expected_sentence_embedding = expected_sentence_embedding.cpu().numpy().reshape(1, -1)
        expected_sentence_embedding = torch.tensor(pca.transform(expected_sentence_embedding))

        print("BERT embedding shape:", expected_sentence_embedding.shape)
        
        subject_embedding = torch.tensor(ft_model[subject]).unsqueeze(0)  # Add batch dimension
        object_embedding = torch.tensor(ft_model[object]).unsqueeze(0)  # Add batch dimension
        print(subject_embedding.shape)
        print(object_embedding.shape)

        print("FastText embeddings (s/o):", subject_embedding.shape, object_embedding.shape)
        actual_sentence_embedding = tensor_function(subject_embedding, object_embedding)

        # Ensure both embeddings are on the same device and have the same shape
        expected_sentence_embedding = expected_sentence_embedding.to(actual_sentence_embedding.device)
        expected_sentence_embedding = expected_sentence_embedding.view(1, -1)
        actual_sentence_embedding = actual_sentence_embedding.view(1, -1)

        # Normalize embeddings
        expected_sentence_embedding = F.normalize(expected_sentence_embedding, p=2, dim=1)
        actual_sentence_embedding = F.normalize(actual_sentence_embedding, p=2, dim=1)

        # Calculate loss
        loss = F.mse_loss(actual_sentence_embedding, expected_sentence_embedding)
        total_loss += loss.item()
        num_samples += 1

        print("cosine similarity: ", cosine_sim(actual_sentence_embedding.detach().numpy(), expected_sentence_embedding.detach().numpy()))
        print("MSE Loss: ", loss.item())

    # Calculate and print aggregate loss
    aggregate_loss = total_loss / num_samples
    print("Aggregate MSE Loss: ", aggregate_loss)


