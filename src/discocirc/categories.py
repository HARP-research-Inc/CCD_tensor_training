import torch
from sentence_transformers import SentenceTransformer
import stanza
import spacy
import re

from ..temporal_spacy.temporal_parsing import SUBORDINATING_CONJUNCTIONS

def atomic_compose(word, model: SentenceTransformer):
    """
    For nouns. Returns atomic type 
    I -> 
    """
    pass

def two_compose(obj1, obj2, model):
    """
    composes two objects with a function
    """
    pass

def three_compose(obj1, obj2, obj3, model):
    """
    composes three objects with a function
    """
    pass

def dummy_compose():
    pass

class Category(object):
    """
    Abstract category class.

    """
    # static variables
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __init__(self, label):
        self.label = label
    
    def inward(self):
        """
        abstract inward method.
        """
        return self
    def forward(self):
        """
        abstract forward method.
        """
        return self
    def __str__(self):
        """
        string representation of the category.
        """
        return self.label
    
    def __eq__(self, other):
        """
        equality operator.
        """
        return id(self) == id(other)
    
    def __hash__(self):
        """
        hash operator.
        """
        return hash(self.label)

class Box(Category):
    """
    HO-DisCoCat-eque Box. Defined as a category with a label and a composition
    function. The composition function is a function that takes in
    a set of objects and returns a set of objects. 
    """
    def __init__(self, label, dimension=384):
        super().__init__(label)
        self.dimension = dimension
        self.composing_function = dummy_compose
        
        self.grammar = "s"

        #these store labels, not the wires themselves.
        self.in_wires: tuple[ Wire ] = None
        self.out_wires: tuple[ Wire ] = None
    
    def get_label(self):
        return self.label
    
    def is_state_or_effect(self):
        """
        returns: pair of booleans. Returns true at index 0 if the box
        is a state, and true at index 1 if the box is an effect.
        """
        return self.in_wires is None, self.out_wires is None
    
    def set_in_wires(self, wires):
        """
        sets the in wires of the box.
        """
        self.in_wires = wires
        return self
    def set_out_wires(self, wires):
        """
        sets the out wires of the box.
        """
        self.out_wires = wires
        return self
    
    def add_in_wire(self, wire):
        """
        adds a wire to the in wires of the box.
        """
        if self.in_wires is None:
            self.in_wires = list()
        self.in_wires.append(wire)
        return self
    
    def add_out_wire(self, wire):
        """
        adds a wire to the out wires of the box.
        """
        if self.out_wires is None:
            self.out_wires = list()
        self.out_wires.append(wire)
        return self

    def check_rep():
        """
        Checks representation of box. Ensures there are no cycles identifies
        sources.
        """
        pass
    def forward(self):
        
        pass

class Spider(Box):
    """
    DisCoCat Spider. Defined as a Box utilzing a composition
    function equivalent to the adposition "and", and equivalent in the 
    graphical calculus to a logical "and".

    Categorial abstraction:     
    """
    def __init__(self, label):
        super().__init__(label)

class Custodian(Box):

    #static


    def __init__(self,label):
        super.__init__(label)

class Wire(Category):
    def __init__(self, label: str, child: Box, grammar = "n", dimension=384):
        super().__init__(label)
        self.grammar = grammar
        self.dimension = dimension
        self.embedding = torch.nn.Embedding(1, dimension)
        self.sink = child
    
    def get_label(self):
        return self.label

    def get_sink_label(self):
        return self.sink.get_label()
    
    def get_sink(self):
        return self.sink
    
    def __str__(self):
        return self.label

class Actor(Wire):
    """
    Actor wire. Works by "infecting" forward wires.
    """
    def __init__(self, label: str, child: Box, actor_label: str, grammar = "n", dimension=384, ):
        super().__init__(label, child, grammar, dimension)
        self.name = actor_label

class Circuit(Category):
    """
    Circuit arranged as a modified adjacency list. 

    Abstraction function: Circuit -> adjacency_list -> C = (B, W)
    where B is a list of boxes containing compositional functions and 
    W is a list of wires.

    Representation invariant:
    Circuit maps onto a DAG, edge order preserved by pregroup structure.

    """
    def __init__(self, label, dimension=384):
        super().__init__(label)
        self.adjacency_list: dict[Box, list[Wire]] = {}
        self.root = None #root token
        self.leaves = list()

    def __str__(self):
        """
        string representation of the circuit.
        """
        return_string = self.label + "\n"
        for box, wires in self.adjacency_list.items():
            return_string += f"{box.get_label()}:"
            for wire in wires:
                return_string += f" {wire.get_label()}"
            return_string += "\n"
        return_string += "\n"
        return return_string
    
    def add_node(self, node:Box):
        """
        adds either spider or box.
        """
        if node is None:
            return False
        
        if(node in self.adjacency_list):
            return True
        
        self.adjacency_list[node] = list()
        return True
        
        
    def add_wire(self, parentBox: Box, childBox: Box):
        
        #really dumb way to do this, but it works.
        #if add_node returns false in either case, then the wire is not added.

        added_parent = self.add_node(parentBox)
        added_child = self.add_node(childBox)

        if added_parent and added_child:
            wire = Wire(f"{parentBox.get_label()} -> {childBox.get_label()},", childBox)

            parentBox.add_out_wire(wire)
            childBox.add_in_wire(wire)

            self.adjacency_list[parentBox].append(wire)
            return True
        return False
    
    def get_adjacency_list(self):
        """
        returns the adjacency list of the circuit.
        """
        return self.adjacency_list

    def concactenate(self, other, conjunction = None):
        self.adjacency_list.update(other.get_adjacency_list())

###############################
###### PARSING FUNCTIONS ######
###############################

def __parse_driver(circuit: Circuit, parent: Box, leaves: list, token: spacy.tokens.Token):
    
    child_box = Box(token.text)

    #traversal is in the opposite direction of the tree.
    circuit.add_wire(parent,child_box)

    if(token.n_lefts == 0 and token.n_rights == 0):
        #base case
        leaves.append(child_box)
    
    for child in token.children:
        #print(token.text, child.text)
        __parse_driver(circuit, child_box, leaves, child)

def __tree_parse(circuit: Circuit, string, spacy_model: spacy.load):
    doc = spacy_model(string)
    root = [token for token in doc if token.head == token][0]

    #print(root.text, root.pos_)
    leaves = list()
    #circuit.add_node(root_box)

    

    __parse_driver(circuit, None, leaves, root)

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

    circuit = Circuit("Discourse:"+ discourse)

    for i, clause in enumerate(clauses):
        new_circuit = Circuit(f"Clause {i+1}")
        __tree_parse(new_circuit, clause, nlp)
        circuit.concactenate(new_circuit)
    return circuit

if __name__ == "__main__":
    sentence1 = "hey, the quick brown fox jumps over the lazy dog and I watched it happen, it was cool but I was sad."
    sentence2 = "Good morning, I hope you are doing well. I am looking forward to our meeting tomorrow."
    spacy_model = "en_core_web_trf"

    Sentence3 = "Hey, the quick brown fox jumps over the lazy dog and I watched it happen, it was cool but I was sad. Good morning, I hope you are doing well. I am looking forward to our meeting tomorrow."
    nlp = spacy.load(spacy_model)

    print(driver(Sentence3, nlp))


    # circuit1 = Circuit("DISCOURSE 1")
    # circuit2 = Circuit("DISCOURSE 2")

    # circuit3 = Circuit("DISCOURSE 3")

    # __tree_parse(circuit1, sentence1, nlp)
    # __tree_parse(circuit2, sentence2, nlp)

    # circuit3.concactenate(circuit1)
    # circuit3.concactenate(circuit2)

    # print(circuit3)

    #print(split_clauses_with_markers(sentence))
    

    
