from .pos import *
from .categories import *
from ..temporal_spacy.temporal_parsing import SUBORDINATING_CONJUNCTIONS

###############################
###### PARSING FUNCTIONS ######
###############################

def parse_driver(circuit: Circuit, parent: Box, leaves: list, token: spacy.tokens.Token):
    
    child_box = Box(token.text)

    #traversal is in the opposite direction of the tree.
    circuit.add_wire(parent,child_box)

    if(token.n_lefts == 0 and token.n_rights == 0):
        #base case
        leaves.append(child_box)
    
    for child in token.children:
        #print(token.text, child.text)
        parse_driver(circuit, child_box, leaves, child)

def tree_parse(circuit: Circuit, string, spacy_model: spacy.load, source: Box = None):
    doc = spacy_model(string)
    root = [token for token in doc if token.head == token][0]

    leaves = list()

    parse_driver(circuit, source, leaves, root)

    return leaves

CONJUNCTION_LIST = SUBORDINATING_CONJUNCTIONS["temporal"] | SUBORDINATING_CONJUNCTIONS["causal"] | \
    SUBORDINATING_CONJUNCTIONS["conditional"] | SUBORDINATING_CONJUNCTIONS["concessive"] | \
    SUBORDINATING_CONJUNCTIONS["purpose"] | SUBORDINATING_CONJUNCTIONS["result/consequence"] | \
    SUBORDINATING_CONJUNCTIONS["comparison"] | SUBORDINATING_CONJUNCTIONS["manner"] | \
    SUBORDINATING_CONJUNCTIONS["relative (nominal)"] | SUBORDINATING_CONJUNCTIONS["exception"] |\
    {"and", "but", "or", "nor", "for", "so", "yet", "either", "neither", "and/or"}

PUNCTUATION_DELIMS = {",", ".", "!", "?", ";", ":"}


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

    circuit = Circuit("*****DISCOURSE*****")

    # Create a root box for the circuit
    root_box = Bureacrat("REFERENCE")

    # Composer box to combine clauses
    composer = Spider("SPIDER COMPOSE")

    # this will be the main output wire of the circuit

    print(circuit.add_wire(composer, root_box))

    for i, clause in enumerate(clauses):
        print("CLAUSE", i+1, ":", clause)
        new_circuit = Circuit(f"Clause {i+1}")
        new_circuit.add_wire(composer, root_box)
        tree_parse(new_circuit, clause, nlp, composer)
        print(new_circuit)
        circuit.concactenate(new_circuit)

    return root_box, circuit

if __name__ == "__main__":
    sentence1 = "hey, the quick brown fox jumps over the lazy dog and I watched it happen, it was cool but I was sad."
    sentence2 = "Good morning, I hope you are doing well. I am looking forward to our meeting tomorrow."
    spacy_model = "en_core_web_trf"

    Sentence3 = "Hey, the quick brown fox jumps over the lazy dog and I watched it happen, it was cool but I was sad. Good morning, I hope you are doing well. I am looking forward to our meeting tomorrow."
    nlp = spacy.load(spacy_model)

    ref, discourse = driver(Sentence3, nlp)

    print(discourse)


    # circuit1 = Circuit("DISCOURSE 1")
    # circuit2 = Circuit("DISCOURSE 2")

    # circuit3 = Circuit("DISCOURSE 3")

    # __tree_parse(circuit1, sentence1, nlp)
    # __tree_parse(circuit2, sentence2, nlp)

    # circuit3.concactenate(circuit1)
    # circuit3.concactenate(circuit2)

    # print(circuit3)

    #print(split_clauses_with_markers(sentence))