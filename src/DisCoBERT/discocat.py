from .pos import *
from .categories import *
from ..temporal_spacy.temporal_parsing import SUBORDINATING_CONJUNCTIONS

import torch
import torch.nn.functional as F
import time

from src.regression import TwoWordTensorRegression
import re
from src.regression import CPTensorRegression, TwoWordTensorRegression


###############################
###### PARSING FUNCTIONS ######
###############################

MODEL_PATH = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training/"
CONJUNCTION_LIST = SUBORDINATING_CONJUNCTIONS["temporal"] | SUBORDINATING_CONJUNCTIONS["causal"] | \
    SUBORDINATING_CONJUNCTIONS["conditional"] | SUBORDINATING_CONJUNCTIONS["concessive"] | \
    SUBORDINATING_CONJUNCTIONS["purpose"] | SUBORDINATING_CONJUNCTIONS["result/consequence"] | \
    SUBORDINATING_CONJUNCTIONS["comparison"] | SUBORDINATING_CONJUNCTIONS["manner"] | \
    SUBORDINATING_CONJUNCTIONS["relative (nominal)"] | SUBORDINATING_CONJUNCTIONS["exception"] |\
    {"and", "but", "or", "nor", "for", "so", "yet", "either", "neither", "and/or"}

PUNCTUATION_DELIMS = {",", ".", "!", "?", ";", ":"}

def parse_driver(circuit: Circuit, parent: Box, leaves: list, token: spacy.tokens.Token, factory: Box_Factory, is_one_deep: bool):
    """
    Parameter field names indicate the parent/child relationship in reference
    to the tree structure, NOT the circuit structure. 
    """
    if is_one_deep:
        circuit.set_root(parent)
    
    pos = token.pos_
    
    child_box = factory.create_box(token.text, pos)

    #print(pos, type(child_box))

    #traversal is in the opposite direction of the tree.
    circuit.add_wire(child_box, parent) # order swapped from tree traversal order

    if(token.n_lefts == 0 and token.n_rights == 0):
        #base case
        leaves.append(child_box)
    
    for child in token.children:
        #print(token.text, child.text)
        parse_driver(circuit, child_box, leaves, child, factory, False)

def tree_parse(circuit: Circuit, string, spacy_model: spacy.load, factory: Box_Factory, source: Box = None):
    """
    Parsing traversal order should be in the opposite direction of the circuit.
    Parameter field names indicate the parent/child relationship in reference
    to the tree structure, NOT the circuit structure. 

    args:
        source: source box for the whole tree (in the prototype, it is a composer spider)
    """
    doc = spacy_model(string)
    root = [token for token in doc if token.head == token][0]

    leaves = list()

    parse_driver(circuit, source, leaves, root, factory, True)

    return leaves


def split_clauses_with_markers(sentence, nlp: spacy.load):
    # Build regex for conjunctions (prioritize multi-word)
    sorted_conjs = sorted(CONJUNCTION_LIST, key=lambda x: -len(x))
    escaped_conjs = [r'\b' + re.escape(conj) + r'\b' for conj in sorted_conjs]
    conj_pattern = '|'.join(escaped_conjs)
    
    # Build regex for punctuation
    punct_pattern = '|'.join(re.escape(p) for p in PUNCTUATION_DELIMS)

    # Combined pattern: capture all splitters
    pattern = r'\s*(%s|%s)\s*' % (conj_pattern, punct_pattern)

    # Split and keep delimiters
    parts = re.split(pattern, sentence)

    # Group into clauses and splitters
    clauses = parts[::2]
    markers = parts[1::2]

    # Clean up
    clauses = [c.strip() for c in clauses if c.strip()]
    markers = [m.strip() for m in markers if m.strip()]

    return clauses, markers

def driver(discourse: str, nlp: spacy.load):
    """
    returns: circuit object containing circuit reprsenentation of the discourse.

    """
    clauses, conjunctions = split_clauses_with_markers(discourse, nlp)

    factory = Box_Factory(nlp, MODEL_PATH)

    circuit = Circuit("*****DISCOURSE*****")

    # Create a root box for the circuit
    root_box = factory.create_box("REFERENCE", "bureaucrat")

    # Composer box to combine clauses
    composer = factory.create_box("SPIDER COMPOSE", "spider")

    for i, clause in enumerate(clauses):
        #print("CLAUSE", i+1, ":", clause)
        new_circuit = Circuit(f"Clause {i+1}")
        sources = tree_parse(new_circuit, clause, nlp, factory, composer)

        new_circuit.set_sources(sources)

        #print("Sources:", [source.get_label() for source in sources])

        #print(new_circuit.root)

        circuit.concactenate(new_circuit)
        
    circuit.add_wire(composer, root_box)

    return root_box, circuit

if __name__ == "__main__":

    #version 0.1.0 - bag of clauses approach

    path_to_models = "/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training"
    spacy_model = "en_core_web_trf"

    one_clause = "the big fat deformed french man eats a small helpless newborn baby"
    one_clause2 = "small dog eats big man"
    annoying = "she should have been being watched carefully"

    nlp = spacy.load(spacy_model)

    dummy = Category("blank")

    dummy.set_nlp(nlp)


    ref, discourse = driver("hey the french freak quickly ate the baby", nlp)
    ref, discourse2 = driver("the french freak quickly ate the baby", nlp)

    embedding = discourse.forward()
    embedding2 = discourse2.forward()

    print(F.cosine_similarity(embedding[1], embedding2[1], dim=1))

    
    # print(discourse)

    # [print(source) for source in discourse.sources]

    #print(discourse)
    # final_embedding = discourse1.forward()
    # comparison_embedding = discourse2.forward()
    #ground_truth = Box.model_cache.retrieve_BERT(one_clause)

    # start_time = time.time()
    # for i in range(1000):
    #     print("DisCoBERT iteration:", i)
    #     _, _ = driver("I eat food", nlp)
    # end_time = time.time()

    # DCBERT_time = end_time - start_time

    # start_time = time.time()
    # for i in range(1000):
    #     print("SBERT iteration:", i)
    #     Box.model_cache.retrieve_BERT("I eat food")
    # end_time = time.time()

    # SBERT_time = end_time - start_time

    # print("DisCoBERT time:", DCBERT_time)
    # print("SBERT time:", SBERT_time)



    #print(F.cosine_similarity(final_embedding[1], comparison_embedding[1], dim=1))
    

