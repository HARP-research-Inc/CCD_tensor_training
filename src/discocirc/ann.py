import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

BERT_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ann(target: str, candidates: list, context = ""):
    max_score = float('-inf')
    best_candidate = None
    target_embedding = BERT_model.encode(context + target, convert_to_tensor=True)

    for candidate in candidates:
        candidate_embedding = BERT_model.encode(context + candidate, convert_to_tensor=True)
        score = F.cosine_similarity(target_embedding, candidate_embedding, dim=0).item()
        
        print(f"Comparing '{target}' with '{candidate}': score = {score}")
        
        if score > max_score:
            max_score = score
            best_candidate = candidate
    return best_candidate, max_score if best_candidate else "No suitable candidate found"

def load_model(directory: str, model_name):
    """
    load regression model from the given path.
    """
    model_path = f"{directory}/{model_name}"
    model = torch.load(model_path, weights_only=True)
    model.eval()
    return model

def load_ann(directory: str, candidates: str, target: str):
    """
    load ANN model from the given path.
    """
    model_name = ann(target, candidates)[0]

    model_path = f"{directory}/{model_name}"
    model = torch.load(model_path, weights_only=True)
    model.eval()
    return model


if __name__ == "__main__":
    with open("ref/adv_adj.txt", "r") as f:
        adv_adj = f.read().splitlines()
    
    target = "&"

    print(ann(target, adv_adj))