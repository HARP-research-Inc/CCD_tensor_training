from util import get_embedding_in_parallel, cosine_sim
from regression import FullRankTensorRegression
import torch

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

def build_cache(file_path):
    with open(file_path,'r') as reference:
        data = reference.readlines()
    
    cache = list() #stores as tuple pair (word, embedding)

    #building cache
    for word in data:
        word = word.strip()
        embedding = get_embedding_in_parallel(word)
        if(embedding is None):
            print("error:", word.strip())
            continue
        cache.append((word.strip(),embedding))
    
    reference.close()
    
    return cache

def linear_search(cache, word):
    """
    Searches for the nearest neighbor in the cache.
    
    Args:
        cache: list of tuples (word, embedding)
        word: string to search for
    
    Returns:
        embedding: tensor embedding of the word closest
    """
    closest_word = None
    closest_distance = 0

    word_embedding = get_embedding_in_parallel(word)
    if word_embedding is None:
        print("error:", word.strip())
        return None


    print(word)

    for i in range(len(cache)):
        candidate_word, candidate_embedding = cache[i]
        distance = cosine_sim(word_embedding, candidate_embedding)
        
        if distance > closest_distance:
            
            closest_distance = distance
            closest_word = candidate_word
            closest_embedding = candidate_embedding 
    
    return closest_word, closest_distance, closest_embedding

def load_model_in(cache, file_path, target_word):
    for pair in cache:
        if pair[0] == target_word:
            model = FullRankTensorRegression(300, 300)
            model.load_state_dict(torch.load(file_path+"/"+target_word, weights_only=False))
            model.eval()
            return model
            
    print("error: model not found")
    return None



if __name__ == "__main__":
    cache = build_cache("transitive_verb_model/table.reference")
    #print(cache)

    data = linear_search(cache, "break")

    print(data[0], data[1])

    model = FullRankTensorRegression(300, 300)
    model = load_model_in(cache, "transitive_verb_model", data[0])
    print(model(get_embedding_in_parallel("ball"),get_embedding_in_parallel("bat")))