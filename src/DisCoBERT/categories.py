import torch
from sentence_transformers import SentenceTransformer
from .ann import ModelBank
import spacy

def interpret_requirement(actual: int, expected: str) -> bool:
    """
    'inf' -> infinite
    b1:b2 -> b1 to b2 inclusive

    0 <= b1,b2 <= inf
    """

    if ":" in expected:
        b1, b2 = expected.split(":")
    elif( expected == "inf"):
        b1 = "0"
        b2 = "inf"
    else:
        b1 = expected
        b2 = expected
    
    if b1.isnumeric():
        b1 = int(b1)
    elif b1 == "inf":
        b1 = 0
    else:
        raise ValueError(f"Invalid lower bound: {b1}")
    
    if b2.isnumeric():
        b2 = int(b2)
    elif b2 == "inf":
        b2 = float('inf')
    else:
        raise ValueError(f"Invalid upper bound: {b2}")
    
    return b1 <= actual <= b2

class Category(object):
    """
    Abstract category class.

    """

    model_cache: ModelBank = ModelBank("/mnt/ssd/user-workspaces/aidan-svc/CCD_tensor_training")

    def __init__(self, label):
        self.label = label
    
    def inward(self):
        """
        abstract inward method.
        """
        return self
    def forward(self) -> list:
        """
        abstract forward method.

        packet format: (POS: str, data, data, ...)
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
    def set_nlp(self, nlp: spacy.load):
        Category.model_cache.set_nlp(nlp)


class Box(Category):
    """
    HO-DisCoCat-eque Box. Defined as a category with a label and a composition
    function. The composition function is a function that takes in
    a set of objects and returns a set of objects. 
    """
    def __init__(self, label: str, model_path: str = None):
        super().__init__(label)
        
        self.grammar = "s"

        #these store labels, not the wires themselves.

        self.model_path = model_path

        self.in_wires: list[ Wire ] = []
        self.out_wires: list[ Wire ] = []

        self.packets: list[list] = []

        self.grammar_data_cache: dict[str, int] = {}

        self.inward_requirements: dict = {(None, "0:inf")}  # default requirement, can be overridden

        self.type = None
    
    def get_label(self):
        return self.label

    
    def is_state_or_effect(self):
        """
        returns: pair of booleans. Returns true at index 0 if the box
        is a state, and true at index 1 if the box is an effect.
        """
        return len(self.in_wires) == 0, len(self.out_wires) == 0
    
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

    def check_packet_status(self):
        """
        checks if box has right number of packets of each type.
        """
        status = True
        for pair in self.inward_requirements:
            # if pair[0] not in self.grammar_data_cache:
            #     status = status and interpret_requirement(0, pair[1])
            # else:
            #     status = status and interpret_requirement(self.grammar_data_cache[pair[0]], pair[1])

            status = status and interpret_requirement(0 if pair[0] not in self.grammar_data_cache else self.grammar_data_cache[pair[0]], pair[1])
            
            if status == False:
                return status
        return status 

    def inward(self, input: list):
        """
        """
        if input[0] not in self.grammar_data_cache:
            self.grammar_data_cache[input[0]] = 0
        self.grammar_data_cache[input[0]] += 1

        self.packets.append(input)

    def forward_helper(self):
        return None

    def forward(self):

        packet = list()
        packet.append(self.type)

        packet.append(self.forward_helper())
        #print(packet)
        #print([item[1] for item in self.packets if type(item) is torch.Tensor])

        # 

        for wire in self.out_wires:
            
            wire.inward(packet)
            wire.forward()
        
        return packet
        

class Wire(Category):
    def __init__(self, label: str, child: Box, grammar = "n", dimension=384):
        super().__init__(label)
        self.grammar = grammar
        self.dimension = dimension
        self.embedding = torch.nn.Embedding(1, dimension)
        self.packet = []
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
    
    def inward(self, packet):
        self.packet = packet
    
    def forward(self):
        self.sink.inward(self.packet)

    def __str__(self):
        return self.label


class Bureaucrat(Box):
    """
    Bureacratic box. Closes open wires in order to maintain DAG definition. Also 
    reads wire output.
    """

    #static
    #references: dict[Wire, list[Box]] = dict()


    def __init__(self,label: str):
        super().__init__(label)
        self.type = "REFERENCE"
        self.embedding = None
    
    def forward_helper(self):
        #print("inwire length", len(self.in_wires))
        #print("inwire label", self.in_wires[0].label)

        if len(self.in_wires) != 1:
            raise TypeError(f"Expected input of 1 vector, {len(self.in_wires)}.")
        
        self.embedding = self.in_wires[0].packet[1]

        return self.embedding

# class Actor(Wire):
#     """
#     Actor wire. Works by "infecting" forward wires.
#     """
#     def __init__(self, label: str, child: Box, actor_label: str, grammar = "n", dimension=384, ):
#         super().__init__(label, child, grammar, dimension)
#         self.name = actor_label

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
        self.sources: list[Box] = list()

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
            return

        
        self.adjacency_list.update(other.get_adjacency_list())  

        self.sources.extend(other.get_sources())
    
    def forward(self):
        """
        modified BFS traversal.

        TRAVERSAL, "FORWARD COMPOSITION" PROBLEM
        """
        queue: list[Box] = self.sources.copy()

        explored = set()
        last_output = None

        while len(queue) > 0:
            v = queue.pop(0)
            print("\n\n********Current node:********:", v.get_label())
            print("Breadth-first traversal queue:\n\n", [q.get_label() for q in queue])

            if v.check_packet_status():
                #print(v.get_label())
                last_output = v.forward()
                #print(v.out_wires)
                for wire in v.out_wires:
                    if wire.get_sink() not in explored:
                        explored.add(wire.get_sink())
                        queue.append(wire.get_sink())
            else:
                queue.append(v)
        
        return last_output
    
    def query_wire(self, parent: str, child: str):
        pass
        
if __name__ == "__main__":
    # interpret requrements test
    print(interpret_requirement(2, "0:inf"))  # True
    print(interpret_requirement(2, "2:2"))    # True
    print(interpret_requirement(2, "1:3"))    # True
    print(interpret_requirement(2, "3:inf"))  # False
    print(interpret_requirement(2, "inf"))    # True
    print(interpret_requirement(2, "0:1"))    # False

    #1 adverb 1 noun
    print(interpret_requirement(1, "0:inf") or interpret_requirement(2, "2:2"))