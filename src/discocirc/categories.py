import torch
from sentence_transformers import SentenceTransformer
import stanza
import spacy
import re


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
    def __init__(self, label: str, dimension=384):
        super().__init__(label)
        self.dimension = dimension
        self.composing_function = dummy_compose
        
        self.grammar = "s"

        #these store labels, not the wires themselves.
        self.in_wires: list[ Wire ] = None
        self.out_wires: list[ Wire ] = None
    
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


class Wire(Category):
    def __init__(self, label: str, child: Box, grammar = "n", dimension=384):
        super().__init__(label)
        self.grammar = grammar
        self.dimension = dimension
        self.embedding = torch.nn.Embedding(1, dimension)
        self.packet = None
        self.sink = child
    
    def get_label(self):
        return self.label

    def get_sink_label(self):
        return self.sink.get_label()
    
    def get_sink(self):
        return self.sink

    def get_embedding(self):
        """
        returns the embedding of the wire.
        """
        return self.embedding(torch.tensor([0]))
    
    def __str__(self):
        return self.label


class Bureaucrat(Box):
    """
    Bureacratic box. Closes open wires in order to maintain DAG definition. Also 
    reads wire output.
    """

    #static
    references: dict[Wire, list[Box]] = dict()


    def __init__(self,label: str):
        super().__init__(label)
        self.embedding = None
    
    def inward(self, input: Wire):
        input.embedding = input.get_embedding()

    def forward(self):
        return self.embedding


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
        self.root = None #root node
        self.sources = list()

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
        
        if node in self.adjacency_list:
            return True
        
        self.adjacency_list[node] = list()
        return True
    
    def set_root(self, root: Box):
        self.root = root
        
        
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

    def set_sources(self, sources: list[Box]):
        """
        sets the sources of the circuit.
        """
        self.sources = sources

    def concatenate_sources(self, sources: list[Box]):
        """
        concatenates the sources of the circuit with the given sources.
        """
        self.sources.extend(sources)
    
    def get_sources(self):
        """
        returns the sources of the circuit.
        """
        return self.sources

    def concactenate(self, other, conjunction: Box = None):
        #checklist: move all boxes and wires, move all source references, update root
        
        if conjunction is not None: #if no conjunction is passed, then the circuit will have two discrete parts. 
            pass

        
        self.adjacency_list.update(other.get_adjacency_list())  

        self.sources.extend(other.get_sources())

        