from util import get_embedding_in_parallel, cosine_sim
import torch

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
    
    
    return cache


if __name__ == "__main__":
    cache = build_cache("transitive_verb_model/table.reference")
    print(cache)