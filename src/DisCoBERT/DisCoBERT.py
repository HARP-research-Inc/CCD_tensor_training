from .categories import *
from . import discocat as DCC

from sentence_transformers import SentenceTransformer

class DisCoBERT(object):
    def __init__(self, spacy_model: str):
        """
        DisCoBERT wrapper.
        
        Args:
            model_path (str): Path to the BERT model.
        """
        self.nlp = spacy.load(spacy_model)
        dummy = Category("blank")
        dummy.set_nlp(self.nlp)

    def encode(self, text: str):
        _, discourse = DCC.driver(text, self.nlp)
        embedding = discourse.forward()[1]

        return embedding

        

if __name__ == "__main__":
    module = DisCoBERT("en_core_web_lg")

    embedding = module.encode("the quick brown fox jumped over the lazy dog.")
    print(embedding)

        
    
    