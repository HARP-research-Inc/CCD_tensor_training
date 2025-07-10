from .categories import *
from . import discocat as DCC
import torch

from sentence_transformers import SentenceTransformer

class DisCoBERT(object):
    def __init__(self, spacy_model: str):
        """
        DisCoBERT wrapper.
        
        Args:
            model_path (str): spaCy model.
        """
        self.nlp = spacy.load(spacy_model)
        dummy = Category("blank")
        dummy.set_nlp(self.nlp)

    def encode_confident(self, text: str):
        _, discourse = DCC.driver(text, self.nlp)
        embedding = discourse.forward()[1]

        return embedding

    def encode(self, text: str):
        embedding = None
        try:
            _, discourse = DCC.driver(text, self.nlp)
            embedding = discourse.forward()[1]
        except ValueError:
            #returns zero vector
            embedding = torch.zeros(1, 384)
            embedding = embedding.cpu().numpy().reshape(1, -1)
        return embedding

    def driver(self, text: str):
        ref, discourse = DCC.driver(text, self.nlp)
        embedding = discourse.forward()[1]

        return ref, embedding

        

if __name__ == "__main__":
    # example usage:
    module = DisCoBERT("en_core_web_lg")

    embedding = module.encode_confident("The question is rather more complicated than it has been represented here.")
    print(embedding)

        
    
    