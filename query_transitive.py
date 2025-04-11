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

        


if __name__ == "__main__":
    cache = build_cache("transitive_verb_model/table.reference")
    #print(cache)

    data = linear_search(cache, "diet")

    print(data[0], data[1])