from .pos import *
from .categories import *
from ..temporal_spacy.temporal_parsing import SUBORDINATING_CONJUNCTIONS

###############################
###### PARSING FUNCTIONS ######
###############################

def parse_driver(circuit: Circuit, parent: Box, leaves: list, token: spacy.tokens.Token, is_one_deep: bool):
    """
    Parameter field names indicate the parent/child relationship in reference
    to the tree structure, NOT the circuit structure. 
    """
    if is_one_deep:
        circuit.set_root(parent)
    
    child_box = Box(token.text)

    #traversal is in the opposite direction of the tree.
    circuit.add_wire(child_box, parent)

    if(token.n_lefts == 0 and token.n_rights == 0):
        #base case
        leaves.append(child_box)
    
    for child in token.children:
        #print(token.text, child.text)
        parse_driver(circuit, child_box, leaves, child, False)

def tree_parse(circuit: Circuit, string, spacy_model: spacy.load, source: Box = None):
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

    parse_driver(circuit, source, leaves, root, True)

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
    root_box = Bureaucrat("REFERENCE")

    # Composer box to combine clauses
    composer = Spider("SPIDER COMPOSE")


    for i, clause in enumerate(clauses):
        print("CLAUSE", i+1, ":", clause)
        new_circuit = Circuit(f"Clause {i+1}")
        sources = tree_parse(new_circuit, clause, nlp, composer)

        new_circuit.set_sources(sources)

        print("Sources:", [source.get_label() for source in sources])

        #print(new_circuit.root)

        circuit.concactenate(new_circuit)
        
    circuit.add_wire(composer, root_box)

    return root_box, circuit

if __name__ == "__main__":
    spacy_model = "en_core_web_trf"

    many_clauses = "Hey, the quick brown fox jumps over the lazy dog and I watched it happen, it was cool but I was sad. Good morning, I hope you are doing well. I am looking forward to our meeting tomorrow."

    sample_sentence = "Quick brown fox jumps lazy dog. Little John ate leafy greens."

    nlp = spacy.load(spacy_model)

    ref, discourse = driver(many_clauses, nlp)

    print(discourse)

    [print(source) for source in discourse.sources]