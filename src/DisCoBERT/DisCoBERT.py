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
    
    def get_failures(self):
        return Circuit.breaking_POS

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
    module = DisCoBERT("en_core_web_trf")

    #embedding = module.encode_confident("This topic is more complex than it seems.")
    #embedding = module.encode_confident("Bernstein was also exposed to the Fabians while in England, and their example encouraged him to question aspects of Marx’s theory.")

    #embedding = module.encode_confident("He tried spotted dick while in England.")
    #embedding = module.encode_confident("Bernstein was also exposed to the Fabians while in England")
    #embedding = module.encode_confident("Fascist oppression, in fact, was a major problem for communists and socialists alike, not only in Italy but subsequently in Spain under Francisco Franco and in Germany under Adolf Hitler. Socialist parties had drawn enough votes in Germany, Britain, and France to participate in or even to lead coalition governments in the 1920s and ’30s, and in Sweden the Swedish Social Democratic Workers’ Party won control of the government in 1932 with a promise to make their country into a “people’s home” based on “equality, concern, cooperation, and helpfulness.” Wherever fascists took power, however, communists and socialists were among the first to be suppressed.")
    
    #embedding = module.encode_confident("Socialist parties had drawn enough votes in Germany, Britain, and France to participate in or even to lead coalition governments in the 1920s and ’30s, and in Sweden they made a block game in 2009")

    #embedding = module.encode_confident("In Sweden the Swedish Social Democratic Workers’ Party won control of the government in 1932 with a promise to make their country into a people’s home” based on equality, concern, cooperation, and helpfulness.")
    #embedding = module.encode_confident("They denied him readmission to the university.")
    embedding = module.encode_confident("They allowed reentry but denied him readmission to the university.")

    print(embedding)

        
    
    