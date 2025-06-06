from .categories import *
from src.regression import CPTensorRegression, TwoWordTensorRegression
#from .ann import ann

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
        self.type = "NOUN"
        self.grammar = ['ADJ','SELF']

        self.embedding_state = Box.model_cache.retrieve_BERT(label)

        self.inward_requirements: dict = {("ADJ", "0:inf")}

    def forward_helper(self):
        for packet in self.packets:
            if packet[0] == "ADJ":
                pass


class Adjective(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['SELF', 'NOUN']
        self.type = "ADJ"
        self.model = Box.model_cache.load_ann(label, "src/DisCoBERT/ref/adj_model.txt", 1)
        self.inward_requirements: dict = {("ADV", "0:inf")}

    def forward_helper(self):
        pass


class Transitive_Verb(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['NOUN', 'SELF', 'NOUN']
        self.type = "VERB"

        self.inward_requirements: dict = {("ADV", "0:inf"), 
                                         ("NOUN", "2:2")} 
        
        self.ncomposed = 2
        
        

class Box_Factory(object):
    """
    Factory for creating boxes.
    """
    def __init__(self, NLP: spacy.load, model_path, lenient = False):
        self.NLP = NLP
        self.model_path = model_path
        self.lenient = lenient

    def create_box(self, label: str, feature: str):
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
            if self.lenient:
                return Box(label, self.model_path)
            else:
                raise ValueError(f"Unknown feature: {feature}")
    
    def set_lenient(self, value: bool):
        """
        Sets the lenient mode of the factory.
        If lenient is True, unknown features will return a generic Box.
        If False, an error will be raised for unknown features.
        """
        self.lenient = value
        return self
 
if __name__ == "__main__":
    factory = Box_Factory(spacy.load("en_core_web_trf"), "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training/")

    # tiny_discourse = Circuit("Tiny Discourse")

    for i in range(1000):
        test = factory.create_box(("Abbasid", "adj_model"), "ADJ")

        print(i, type(test.model))
    

    # tiny_discourse.forward()

    # print(tiny_discourse)


