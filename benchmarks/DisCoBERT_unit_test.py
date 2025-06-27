import pytest
import src.DisCoBERT.DisCoBERT as DCB
from src.DisCoBERT.categories import Circuit
from src.DisCoBERT.pos import *
import torch.nn as nn
import src.DisCoBERT.discocat as dcc
import time

@pytest.fixture(scope="module")
def global_model():
    return DCB.DisCoBERT("en_core_web_lg")

############################
##### Functional tests #####
############################
def test_init(global_model):
    model = global_model #wrapper init
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

def test_caching(global_model):
    model = global_model

    timeN1 = time.time()
    _ = model.encode("Alice gobbled burgers")
    timeN2 = time.time()

    timeC1 = time.time()
    for _ in range(10):
        _ = model.encode("Alice gobbled burgers")
    timeC2 = time.time()

    assert timeC2 - timeC1 < timeN2 - timeN1, "Caching did not work as expected. Model should be faster on repeated calls."

##################################
####### basic P.O.S. tests #######
##################################

def test_basic_intransitive(global_model):
    model = global_model

    embedding1 = model.encode("Alice ran")
    embedding2 = model.encode("Alice bites")
    embedding3 = model.encode("Alice sprints")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.95\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.95 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.95 \
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

def test_basic_transitive(global_model):
    model = global_model

    embedding1 = model.encode("I ran home")
    embedding2 = model.encode("Alice bites Bob")
    embedding3 = model.encode("Alice hates Bob")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.95\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.95 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.95\
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

def test_basic_ditransitive(global_model):
    model = global_model

    embedding1 = model.encode("Bob gave Alice cakes")
    embedding2 = model.encode("Bob read Alice books")
    embedding3 = model.encode("Bob tossed Alice balls")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.95\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.95 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.95\
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

def test_basic_imperative(global_model):
    model = global_model

    embedding2 = model.encode("Go home")
    embedding3 = model.encode("Give her cake")
    embedding1 = model.encode("Strike balls")

    # Check that the embeddings are different and not converging on a trivial solution
    assert nn.CosineSimilarity(dim=1)(embedding1, embedding2) < 0.95\
    and nn.CosineSimilarity(dim=1)(embedding2, embedding3) < 0.95 \
    and nn.CosineSimilarity(dim=1)(embedding1, embedding3) < 0.95 \
    , "Model output is identical between disparate cases. Covergence on a trivial solution or hardcoding is likely."

# def test_basic_linking(global_model):
#     model = global_model

#     embedding1 = model.encode("placeholder")



def test_basic_determiner(global_model):
    model = global_model

    nd_1 = model.encode("Dog")
    nd_2 = model.encode("Man")
    nd_3 = model.encode("Bat strikes ball")
    nd_4 = model.encode("Boys tossed girls balls")
    nd_5 = model.encode("Churchgoers gave visitors cakes")
    #nd_6 = model.encode("Give her cake")

    embedding1 = model.encode("The dog")
    embedding2a = model.encode("A man")
    embedding2b = model.encode("The man")
    embedding3 = model.encode("Your bat strikes her ball")
    embedding4a = model.encode("Few boys tossed most girls certain balls")
    embedding4b = model.encode("All boys tossed the many girls some balls")
    embedding5 = model.encode("Some churchgoers gave certain visitors the many cakes")
    #embedding6 = model.encode("Give her the cake")

    assert nn.CosineSimilarity(dim=1)(embedding1, nd_1) < 0.99, "Determiner 'the' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding2a, nd_2) < 0.99, "Determiner 'a' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding2a, embedding2b) < 0.99, "Determiner 'the' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding3, nd_3) < 0.99, "Determiner 'your' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding4a, nd_4) < 0.99, "Determiner 'few' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding4a, embedding4b) < 0.99, "Determiner 'most' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding5, nd_5) < 0.99, "Determiner 'some' did not work as expected."
    #assert nn.CosineSimilarity(dim=1)(embedding6, nd_6) < 0.99, "Determiner 'her' did not work as expected."

def test_adverb_on_verbs(global_model):
    model = global_model

    na_1 = model.encode("James ran")
    na_2 = model.encode("This piggy ran home")
    na_3 = model.encode("Your bat strikes her ball")
    na_4 = model.encode("The anarchists broke with the united front")

    embedding1a = model.encode("James quickly ran")
    embedding1b = model.encode("James ran quickly")
    embedding1c = model.encode("Quickly James ran")

    embedding2a = model.encode("This piggy ran home quickly")
    embedding2b = model.encode("This piggy quickly ran home")

    embedding3a = model.encode("Your bat barely strikes her ball")
    embedding3b = model.encode("Your bat strikes her ball barely")

    embedding4a = model.encode("The anarchists famously broke with the united front")
    embedding4b = model.encode("The anarchists broke with the united front famously")

    assert nn.CosineSimilarity(dim=1)(embedding1a, na_1) < 0.99, "Application order case 1a failed"
    assert nn.CosineSimilarity(dim=1)(embedding1b, na_1) < 0.99, "Application order case 1b failed"
    assert nn.CosineSimilarity(dim=1)(embedding1c, na_1) < 0.99, "Application order case 1c failed"

    assert nn.CosineSimilarity(dim=1)(embedding2a, na_2) < 0.99, "Application order case 2a failed"
    assert nn.CosineSimilarity(dim=1)(embedding2b, na_2) < 0.99, "Application order case 2b failed"

    assert nn.CosineSimilarity(dim=1)(embedding3a, na_3) < 0.99, "Application order case 3a failed"
    assert nn.CosineSimilarity(dim=1)(embedding3b, na_3) < 0.99, "Application order case 3b failed" 

    assert nn.CosineSimilarity(dim=1)(embedding4a, na_4) < 0.99, "Application order case 4a failed"
    assert nn.CosineSimilarity(dim=1)(embedding4b, na_4) < 0.99, "Application order case 4b failed"

    assert nn.CosineSimilarity(dim=1)(embedding1a, embedding1b) > 0.95, "Identity case 1 failed"
    assert nn.CosineSimilarity(dim=1)(embedding2a, embedding2b) > 0.95, "Identity case 2 failed"
    assert nn.CosineSimilarity(dim=1)(embedding3a, embedding3b) > 0.95, "Identity case 3 failed"
    assert nn.CosineSimilarity(dim=1)(embedding4a, embedding4b) > 0.95, "Identity case 4 failed"

    




##################################
######### semantic tests #########
##################################