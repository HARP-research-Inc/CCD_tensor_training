import pytest
import src.DisCoBERT.DisCoBERT as DCB
from src.DisCoBERT.categories import Circuit
from src.DisCoBERT.pos import *
import torch.nn as nn

@pytest.fixture
def sample_data():
    return [1, 2, 3]


def test_init():
    model = DCB.DisCoBERT("en_core_web_lg")
    assert model is not None, "Model initialization failed"


def test_intransitive():
    pass


def test_transitive():
    model = DCB.DisCoBERT("en_core_web_lg")

    embedding1 = model.encode("I ran home")
    embedding2 = model.encode("Alice bites Bob")
    embedding3 = model.encode("Bob hates Alice")

    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.99\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.99 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.99



