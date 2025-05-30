from .categories import *
from src.regression import CPTensorRegression, TwoWordTensorRegression

import torch
from sentence_transformers import SentenceTransformer
import spacy

"""
Models trained: 
- adv
- aux
- cconj adj
- cconj noun
- cconj verb
- determinants
- interjection
- prep aux
- prep verb
- pronoun
- sconj
- transitive verb

"""

class Spider(Box):
    """
    DisCoCat Spider. Defined as a Box utilzing a composition
    function equivalent to the adposition "and", and equivalent in the 
    graphical calculus to a logical "and".
    """
    def __init__(self, label, model_path):
        super().__init__(label, model_path)
        self.model_path = self.model_path + "cconj_adj_model/and"

class Noun(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['n']
    
    def forward(self):
        #apply inward adjectives, 
        pass

class Adjective(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['nl', 'n']

class Transitive_Verb(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['nl', 's', 'nr']

class Box_Factory(object):
    """
    Factory for creating boxes.
    """
    def __init__(self, NLP: spacy.load, model_path):
        self.NLP = NLP
        self.model_path = model_path

    def create_box(self, label: str, feature: str) -> Box:
        if feature == "spider":
            return Spider(label, self.model_path)
        elif feature == "bureaucrat":
            return Bureaucrat(label)
        elif feature == "NOUN" or feature == "PROPN" or feature == "PRON":
            return Noun(label, self.model_path)
        elif feature == "ADJ":
            return Adjective(label, self.model_path)
        elif feature == "VERB":
            return Transitive_Verb(label, self.model_path)
        else:
            return Box(label, self.model_path)

    # def create_box(self, label: str, model_path: str = None) -> Box:
    #     pass

        
if __name__ == "__main__":
    def get_embedding_in_parallel(word, model):
        word_embedding = model.encode(word, convert_to_tensor=True)
        word_embedding = word_embedding.cpu().numpy().reshape(1, -1)

        return torch.from_numpy(word_embedding)

    path_to_model = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training"

    model = TwoWordTensorRegression(384, 384)
    model.load_state_dict(torch.load(f"{path_to_model}/transitive_verb_model/abandon", weights_only=True))
    model.eval()

    bert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    a = get_embedding_in_parallel("I", bert)
    b = get_embedding_in_parallel("you", bert)

    model_output = model(a, b)
    print("Model output for 'I' and 'you':", model_output.shape)


