from .categories import *
from src.regression import CPTensorRegression, TwoWordTensorRegression
#from .ann import ann

import torch
from sentence_transformers import SentenceTransformer
import spacy

"""
Models trained: 
- adv *
- aux 
- cconj adj 
- cconj noun
- cconj verb
- determiners *
- interjection
- prep aux
- prep verb
- pronoun
- sconj
- transitive verb *

"""

class Spider(Box):
    """
    DisCoCat Spider. Defined as a Box utilzing a composition
    function equivalent to the adposition "and", and equivalent in the 
    graphical calculus to a logical "and".
    """
    def __init__(self, label, model_path):
        super().__init__(label, model_path)
        self.model_path = "cconj_adj_model"
        self.type = "spider"
        self.model = Box.model_cache.load_model(self.model_path, "and", n=2)
        self.embedding_state = None
    
    def forward_helper(self):

        for packet in self.packets:
            if type(packet[1]) is not torch.Tensor:
                raise ValueError(f"Expected a torch.Tensor for packet, got {type(packet[1])}")
            if self.embedding_state is None:
                self.embedding_state = packet[1]
            else:
                self.embedding_state = self.model(self.embedding_state, packet[1])
        
        if self.embedding_state is None:
            raise ValueError("No packets were processed, embedding state is None.") 
        return self.embedding_state

class Determiner(Box):
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['SELF', 'NOUN']
        self.type = "DET"
        self.model = Box.model_cache.load_ann((label, "det_model"), n=1)

    def forward_helper(self):
        """
        returns the model
        """
        return self.model

class Adverb(Box):
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.type = "ADV"
        self.grammar = ['SELF', 'VERB', '|', "SELF", "ADJ"]
        self.model = Box.model_cache.load_ann((label, "adv_model"), n=1)
    
    def forward_helper(self):
        """
        returns the model
        """
        return self.model

class Auxilliary(Box):
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.type = "AUX"
        self.grammar = ['SELF', 'VERB']
        self.model = Box.model_cache.load_ann((label, "aux_model"), n=1)
    
    def forward_helper(self):
        """
        returns the model
        """
        return self.model

class Interjection(Box):
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.type = "INTJ"
        self.grammar = ['SELF', 'SENTENCE']
        self.model = Box.model_cache.load_ann((label, "intj_model"), n=1)
    
    def forward_helper(self):
        """
        returns the model
        """
        return self.model

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
                adjective_model : torch.nn.Module = packet[1]
                if not isinstance(adjective_model, torch.nn.Module):
                    raise ValueError(f"Expected a torch.nn.Module for adjective model, got {type(adjective_model)}")
                
                self.embedding_state = adjective_model(self.embedding_state)
        
        return self.embedding_state


class Adjective(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['SELF', 'NOUN']
        self.type = "ADJ"
        self.model = Box.model_cache.load_ann((label, "adj_model"), n=1)
        self.inward_requirements: dict = {("ADV", "0:inf")}

    def forward_helper(self):
        """
        returns the model
        """

        #adv handling will be implemented when adv class is implemented
        return self.model


class Transitive_Verb(Box):
    """

    """
    def __init__(self, label: str, model_path: str):
        super().__init__(label, model_path)
        self.grammar = ['NOUN', 'SELF', 'NOUN']
        self.type = "VERB"

        self.inward_requirements: dict = {("ADV", "0:inf"),
                                          ("INTJ", "0:inf"), 
                                         ("NOUN", "2:2")} 
        
        self.model = Box.model_cache.load_ann((label, "transitive_model"), n=2)

    def forward_helper(self):
        """
        returns an embedding state after processing the NOUN packets.
        """
        noun_packets = [packet[1] for packet in self.packets if packet[0] == "NOUN"]

        print("packet length", len(self.packets))
        if len(noun_packets) != 2:
            raise ValueError(f"Transitive verb {self.label} requires exactly two NOUN packets, got {len(noun_packets)}.")  
        
        #noun packets at index 1 should be pytorch tensors
        output = self.model(noun_packets[0], noun_packets[1])

        ####adverb stuff####
        for packet in self.packets:
            if packet[0] == "ADV" or packet[0] == "INTJ":
                print("test")
                model:torch.nn.Module = packet[1]
                output = model(output)

        return output

            
    """
    "PART OF SPEECH" DEFINITION PROBLEM...

    NEW PARTS OF SPEEC WILL BE DEFINED HERE



    VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    """

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
        elif feature == "DET":
            return Determiner(label, self.model_path)
        elif feature == "ADV":
            return Adverb(label, self.model_path)
        elif feature == "INTJ":
            return Interjection(label, self.model_path)
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

        #print(i, type(test.model))
    

    # tiny_discourse.forward()

    # print(tiny_discourse)


