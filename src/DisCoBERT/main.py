import torch
from sentence_transformers import SentenceTransformer
import spacy
import stanza

#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def __helper():
    pass
    

if __name__ == "__main__":
    spacy_module = spacy.load("training/coref")
    
    test_sentence = "Thomas plays Paradox games and he likes them."