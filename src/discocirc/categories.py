import torch

def compose(obj1, obj2, function):
    """
    composes two objects with a function
    """
    pass

class Category(object):
    """
    Abstract category class.

    """
    def __init__(self, label, dimension=384):
        self.label = label
        self.dimension = dimension
    
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

   

class Wire(Category):
    def __init__():
        pass

class Box(Category):
    def __init__(self, label, dimension=384):
        super().__init__(label, dimension)
        self.composing_function = None

class Spider(Box):
    """
    DisCoCat Spider. Defined as a Box utilzing a composition
    function equivalent to the adposition "and", and equivalent in the 
    graphical calculus to a logical "and".

    Categorial abstraction:     
    """
    def __init__():
        pass

class Circuit(Category):
    """
    Circuit arranged as an adjacency list.

    Abstraction function: Circuit -> adjacency_list -> C = (B, W)
    where B is a list of boxes containing compositional functions and 
    W is a list of wires.

    """
    def __init__(self, label, dimension=384):
        super().__init__(label)
        self.adjacency_list = dict[Box, list[Wire]]
    def add_node():
        """
        adds either spider or box.
        """
        pass
    def add_wire():
        pass 
    


class Actor(Wire):
    def __init__():
        pass
    def forward():
        pass


if __name__ == "__main__":
    print("test")