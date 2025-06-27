import pytest
import src.DisCoBERT.DisCoBERT as DCB
from src.DisCoBERT.categories import Circuit
from src.DisCoBERT.pos import *
import torch.nn as nn
import src.DisCoBERT.discocat as dcc

@pytest.fixture(scope="module")
def global_model():
    return DCB.DisCoBERT("en_core_web_lg")

def test_init(global_model):
    model = global_model
    assert model is not None, "Model initialization failed"

    nlp = spacy.load("en_core_web_lg")
    dummy = Category("blank")
    dummy.set_nlp(nlp)

    wrapper1 = model.encode("Alice bit Bob")
    _, discourse1 = dcc.driver("Alice bit Bob", nlp)

    wrapper2 = model.encode("Tito led Yugoslavia")
    _, discourse2 = dcc.driver("Tito led Yugoslavia", nlp)

    wrapper3 = model.encode("Green frogs")
    _, discourse3 = dcc.driver("Green frogs", nlp)

    loose1 = discourse1.forward()
    loose2 = discourse2.forward()
    loose3 = discourse3.forward()

    
    assert nn.CosineSimilarity(dim=1)(wrapper1, loose1[1]) > 0.99
    assert nn.CosineSimilarity(dim=1)(wrapper2, loose2[1]) > 0.99
    assert nn.CosineSimilarity(dim=1)(wrapper3, loose3[1]) > 0.99

def test_basic_intransitive(global_model):
    model = global_model

    embedding1 = model.encode("I ran")
    embedding2 = model.encode("Alice bites")
    embedding3 = model.encode("Bob sprints")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.99\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.99 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.99 \
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

def test_basic_transitive(global_model):
    model = global_model

    embedding1 = model.encode("I ran home")
    embedding2 = model.encode("Alice bites Bob")
    embedding3 = model.encode("Bob hates Alice")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.99\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.99 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.99\
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

def test_basic_ditransitive(global_model):
    model = global_model

    embedding1 = model.encode("I gave Alice cakes")
    embedding2 = model.encode("Alice read Bob books")
    embedding3 = model.encode("Bob tossed Alice balls")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.99\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.99 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.99\
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

def test_basic_imperative_verb(global_model):
    model = global_model

    embedding2 = model.encode("Go home!")
    embedding3 = model.encode("Follow me!")
    embedding1 = model.encode("Stop!")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.99\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.99 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.99\
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."


