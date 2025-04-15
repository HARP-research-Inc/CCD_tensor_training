import torch
import actor as ac
from transformers import BertModel, BertTokenizer

"""
OLD: modify to use trained models, fastext, etc
"""

class Discourse:
    def __init__(self, model: BertModel, tokenizer: BertTokenizer, debug = False):
        self.__actors : dict[str, ac.Actor] = {}
        self.__embedding = None
        self.__model = model
        self.__tokenizer = tokenizer
        self.debug_mode = debug

    def build_embedding(self):
        if self.debug_mode:
            for name, actor in self.__actors.items():
                if actor.get_state() is None:
                    print(f"Warning: Actor '{name}' has no state!")
                    print(type(actor_embbeddings[0]))
        
        actor_embbeddings = [actor.get_state() for actor in list(self.__actors.values())]

        self.__embedding = torch.logsumexp(torch.stack(actor_embbeddings), dim=0)
        return self.__embedding
    
    """
    Returns cached embedding.
    """
    def get_embedding(self):
        if self.__embedding is None:
            return self.build_embedding()
        
        return self.__embedding
    
    """
    Appends a new actor
    """
    def add_actor(self, new_actor: ac.Actor):
        self.__actors.update({new_actor.get_name(): new_actor})
    
    def add_actor(self, new_actor_name: str):
        if new_actor_name in self.__actors:
            return
        new_actor = ac.Actor(new_actor_name, self.__model, self.__tokenizer)
        self.__actors.update({new_actor_name : new_actor})
    
    def get_actor(self, actor_name: str):
        if not (actor_name in self.__actors):
            self.add_actor(actor_name)

        return self.__actors[actor_name]

    def get_actor_status(self, actor_name: str):
        if not (actor_name in self.__actors):
            raise RuntimeError(f"Unkown actor \'{actor_name}\'")

        target_actor = self.__actors[actor_name]

        return target_actor.get_state()

    def tensor(self, subject_name, object_name, verb):
        subject = self.get_actor(subject_name)
        object = self.get_actor(object_name)

        return subject.tensor(verb, object)
    
    def set_actor_status(self, actor_name: str):
        pass