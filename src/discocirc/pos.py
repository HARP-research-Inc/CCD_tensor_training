from .categories import Box, Wire
from src.regression import CPTensorRegression, TwoWordTensorRegression

import torch
from sentence_transformers import SentenceTransformer

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
class Noun(Box):
    """

    """
    def __init__(self, label: str, dimension=384):
        super.__init__(label)
        self.grammar = ['n']

class Adjective(Box):
    """

    """
    def __init__(self, label: str, dimension=384):
        super.__init__(label)
        self.grammar = ['nl', 'n']

class Transitive_Verb(Box):
    """

    """
    def __init__(self, label: str, dimension=384):
        self.grammar = ['nl', 's', 'nr']

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


