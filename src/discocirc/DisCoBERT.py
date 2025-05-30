from .categories import *
from .discocat import *
from .pos import *

from sentence_transformers import SentenceTransformer

class DisCoBERT(object):
    def __init__(self, model_path: str):
        """
        DisCoBERT class for handling BERT-based models in DisCoCat.
        
        Args:
            model_path (str): Path to the BERT model.
        """
        self.model_path = model_path
        self.boxFactory = Box_Factory()
        self.model = SentenceTransformer(model_path)