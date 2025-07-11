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
####### Basic P.O.S. tests #######
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

def test_basic_adjective(global_model):
    model = global_model

    nd_1 = model.encode("Dog")
    nd_2 = model.encode("Someone")
    nd_3 = model.encode("Churchgoers gave visitors cakes")

    embedding1a = model.encode("Big dog")
    embedding1b = model.encode("Big red dog")

    embedding2a = model.encode("Someone important")
    embedding2b = model.encode("Someone concerned")

    embedding3a = model.encode("Kind churchgoers gave visitors nourishing cakes")
    embedding3b = model.encode("Angry churchgoers gave intruding visitors fake cakes")
    
    assert nn.CosineSimilarity(dim=1)(embedding1a, nd_1) < 0.99
    assert nn.CosineSimilarity(dim=1)(embedding1b, nd_1) < 0.99
    assert nn.CosineSimilarity(dim=1)(embedding2a, nd_2) < 0.99
    assert nn.CosineSimilarity(dim=1)(embedding2b, nd_2) < 0.99
    assert nn.CosineSimilarity(dim=1)(embedding3a, nd_3) < 0.99
    assert nn.CosineSimilarity(dim=1)(embedding3b, nd_3) < 0.99

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

def test_basic_adverb_on_verbs(global_model):
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

def test_interjections(global_model):
    model = global_model

    ni_1 = model.encode("The fox jumped")
    ni_2 = model.encode("Your bat struck her ball")

    embedding1a = model.encode("Dang the fox jumped")
    embedding1b = model.encode("Cool, the fox jumped")
    
    embedding2a = model.encode("Wow your bat struck her ball")
    embedding2b = model.encode("Wow, your bat struck her ball")

    assert nn.CosineSimilarity(dim=1)(embedding1a, ni_1) < 0.99, "(non-comma) Interjection 'dang' did not work as expected (non-comma)"
    assert nn.CosineSimilarity(dim=1)(embedding1b, ni_1) < 0.99, "(comma case) Interjection 'cool' did not work as expected "
    assert nn.CosineSimilarity(dim=1)(embedding2a, ni_2) < 0.99, "(non-comma) Interjection 'wow' did not work as expected"
    assert nn.CosineSimilarity(dim=1)(embedding2b, ni_2) < 0.99, "(comma case) Interjection 'wow,' did not work as expected"
    
def test_auxilliary(global_model):
    model = global_model

    na_1 = model.encode("The fox jumped")
    na_2 = model.encode("Your bat struck her ball")

    embedding1a = model.encode("The fox has jumped")
    embedding1b = model.encode("The fox is jumping")
    
    embedding2a = model.encode("Your bat was striking her ball")
    embedding2b = model.encode("Your bat had struck her ball")

    assert nn.CosineSimilarity(dim=1)(embedding1a, na_1) < 0.99, "Auxilliary verb 'has' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding1b, na_1) < 0.99, "Auxilliary verb 'had' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding2a, na_2) < 0.99, "Auxilliary verb 'is' did not work as expected."
    assert nn.CosineSimilarity(dim=1)(embedding2b, na_2) < 0.99, "Auxilliary verb 'was' did not work as expected."

##################################
######## Edge P.O.S. tests #######
##################################

def test_multiple_verbs_for_subj_connected_by_conj(global_model):
    model = global_model

    _ = model.encode("Bobby runs and jumps")
    # _ = model.encode("Triathletes swim, bike and run")
    # _ = model.encode("Lebron eats, sleeps, balls, and repeats")

def test_multiple_adj_connected_by_conj(global_model):
    model = global_model

    _ = model.encode("The big and red dog")
    # _ = model.encode("The big, red and fluffy dog")
    # _ = model.encode("The big, red, fluffy and cute dog")

##################################
######### Semantic tests #########
##################################

def test_coreference_resolution(global_model):
    """
    Test attempts to measure model's ability to resolve coreference.
    May not work perfectly. Best approach is aggregate results over many sentences.
    """
    model = global_model

    ambiguous1 = model.encode("Alice bit Bob and he cried")
    ambiguous2 = model.encode("Alice bit Bob and she cried")

    ambiguous3 = model.encode("John cycles. He is very fast.")

    candidateA = model.encode("Alice bit Bob and Bob cried")
    candidateB = model.encode("Alice bit Bob and Alice cried")

    canddiateC = model.encode("John cycles. Stacy is very fast.")
    canddiateD = model.encode("John cycles very fast.")

    similarity1A = nn.CosineSimilarity(dim=1)(ambiguous1, candidateA)
    similarity1B = nn.CosineSimilarity(dim=1)(ambiguous1, candidateB)

    similarity2A = nn.CosineSimilarity(dim=1)(ambiguous2, candidateA)
    similarity2B = nn.CosineSimilarity(dim=1)(ambiguous2, candidateB)

    similarity3C = nn.CosineSimilarity(dim=1)(ambiguous3, canddiateC)
    similarity3D = nn.CosineSimilarity(dim=1)(ambiguous3, canddiateD)

    assert similarity1A > similarity1B
    assert similarity2A < similarity2B

    # 0.5 is our general threshold for semantic similarity
    assert nn.CosineSimilarity(dim=1)(ambiguous3, canddiateD) > 0.5
    assert similarity3C < similarity3D
    

