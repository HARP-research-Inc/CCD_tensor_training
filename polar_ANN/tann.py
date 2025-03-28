import torch

import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from util import get_embedding_in_parallel

class TANN:
    """
        Triangulation-Approximate-Nearest-Neighbor algorithm?
        Support-Point_ANN?

        Idk tbh the name is TBA.
    """
    def __init__(self, K: int, C: int):
        self.__top_array: list = None
        self.__N = -1
        self.__pointer_lists = list()
        self.__K = K
        self.__C = C
        self.__built = False

    def build(self, file: str):
        
        with open(file, 'r') as f:
            contents = f.readlines()
        
        for line in contents:
            word = line.strip()
            print(get_embedding_in_parallel(word).shape)
         
        self.__built = True
        
    def query(embedding):
        pass

    