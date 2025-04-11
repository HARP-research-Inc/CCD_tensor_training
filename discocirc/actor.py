import torch
from transformers import BertModel, BertTokenizer


"""
OLD: modify to use trained models, fastext, etc
"""
class Actor:

    def __init__(self, name: str, model: BertModel, tokenizer: BertTokenizer, debug = False):
        self.__name = name
        self.__state = None
        self.__model = model
        self.__tokenizer = tokenizer

        self.debug_mode = debug

        with torch.no_grad():
            tokens = self.__tokenizer(name, return_tensors="pt", padding=True, truncation=True)
            outputs = self.__model(**tokens)
            self.__init_embedding = outputs.last_hidden_state[:, 0, :]  
    """
    @hi
    """
    def get_name(self):
        return self.__name

    def get_state(self):
        return self.__state

    def get_vector_embedding(self):
        return self.__init_embedding

    def modify_state(self, state_embedding: torch.Tensor):
        if self.__state is None:
            self.__state = state_embedding
        else:
            self.__state = torch.mm(self.__state, state_embedding)  

    def tensor(self, verb: str, other: "Actor" = None):
        verb_input = self.__tokenizer(verb, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            verb_embedding = self.__model(**verb_input).last_hidden_state[:, 0, :]  

        subject_embedding = self.get_vector_embedding()
        object_embedding = other.get_vector_embedding() if other else subject_embedding  

        actor_matrix = torch.ger(subject_embedding.squeeze(), object_embedding.squeeze())  
        verb_matrix = torch.ger(verb_embedding.squeeze(), verb_embedding.squeeze())  

        sentence_embedding = torch.tensordot(verb_matrix, actor_matrix, dims=([1], [0]))

        self.modify_state(sentence_embedding)
        other.modify_state(sentence_embedding)

        return sentence_embedding
