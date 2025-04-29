import torch

def dummy_compose():
    pass

def compose(obj1, obj2, function):
    """
    composes two objects with a function
    """
    pass

class Category(object):
    """
    Abstract category class.

    """
    def __init__(self, label):
        self.label = label
    
    def inward(self):
        """
        abstract inward method. In the context of the circuit,
        the abstract category is a ghost that is there but also not.
        """
        return self
    def forward(self):
        """
        abstract forward method. In the context of the circuit,
        the abstract category is a ghost that is there but also not.
        """
        return self
    def __eq__(self, other):
        """

        """
        if isinstance(other, self.__class__):
            return self.label == other.label
        return False
    
    def __hash__(self):
        return hash(self.label)

class Box(Category):
    def __init__(self, label, dimension=384):
        super().__init__(label)
        self.dimension = dimension
        self.composing_function = dummy_compose
        
        self.grammar = None
        self.constituents: tuple[str] = tuple()
    
    def get_label(self):
        return self.label
    
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
        self.sink = child
    
    def get_label(self):
        return self.label

    def get_sink_label(self):
        return self.sink.get_label()
    
    def get_sink(self):
        return self.sink

class Actor(Wire):
    def __init__():
        pass
    def forward():
        pass

class Circuit(Category):
    """
    Circuit arranged as a modified adjacency list. 

    Abstraction function: Circuit -> adjacency_list -> C = (B, W)
    where B is a list of boxes containing compositional functions and 
    W is a list of wires.

    """
    def __init__(self, label, dimension=384):
        super().__init__(label)
        self.adjacency_list = dict[Box, list[Wire]]
    def __str__(self):
        """
        string representation of the circuit.
        """
        return_string = ""
        for box, wires in self.adjacency_list.items():
            return_string += f"{box.get_label()}: {wires}\n"
        return return_string
    
    def add_node():
        """
        adds either spider or box.
        """
        pass
    def add_wire():
        pass 


if __name__ == "__main__":
    test = Box("test")
    print(test.__class__ == Category)