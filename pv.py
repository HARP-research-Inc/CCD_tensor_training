# take in an underlying set
us = underlying_set = {"a", "b", "c", "d", "e"}

class named_set(set):
    """A set with a name attribute for better representation."""
    def __init__(self, elements=None, name=None):
        if elements is None:
            super().__init__()
        else:
            super().__init__(elements)
        self.name = name if isinstance(name, str) else None
    
    def __str__(self):
        if self.name:
            return self.name
        return super().__str__()
    
    def __repr__(self):
        if self.name:
            return f"named_set({super().__repr__()}, name='{self.name}')"
        return f"named_set({super().__repr__()})"

set = named_set

class set_function:
    def __init__(self, domain, codomain, function_components, name=None):
        if (not isinstance(domain, set)) or (not isinstance(codomain, set)):
            raise TypeError("codomain or domain don't form a set")
        if not isinstance(function_components, dict):
            raise TypeError("function components do not form a dictionary")
        
        for x in domain:
            if (x not in function_components.keys()):
                raise ValueError("Function only partially defined over domain. Was undefined for "+str(x))
            if (function_components[x] not in codomain):
                raise ValueError("Function's image lays outside of codomain. Failed on element "+str(function_components[x]))
            
        self.domain = domain
        self.codomain = codomain
        self.map = function_components


        self.name = name if isinstance(name, str) else None
        self.dname = domain.name if domain.name else None
        self.cdname = codomain.name if codomain.name else None

    def __call__(self, input):
        if input not in self.domain:
            raise ValueError("input "+str(input)+"not in domain")
        return self.map[input]

    def __str__(self) -> str:
        if self.name:
            if self.cdname:
                return self.name+"("+self.dname+") -> "+self.cdname
            else:
                return self.name+"("+self.dname+")"
        elif self.dname:
            if self.cdname:
                return "f("+self.dname+") -> "+self.cdname
            else:
                return "f("+self.dname+")"



class quiver:
    def __init__(self, Q0, Q1, src, tgt, name = None):
        # check types of inputs
        if not isinstance(Q0, set):
            raise TypeError("vertices don't form a set")
        
        if not isinstance(Q1, set):
            raise TypeError("arrows don't form a set")
        
        if not isinstance(src, set_function):
            raise TypeError("source function isn't a set function")
        
        if not isinstance(tgt, set_function):
            raise TypeError("target function isn't set function")
        
        if not Q1.issubset(src.domain):
            raise ValueError("The set of arrows is not completely in the domain of the source function")
        
        if Q0.isdisjoint(src.codomain):
            raise ValueError("The vertices aren't defined in the codomain of the source function")
        
        if not Q1.issubset(tgt.domain):
            raise ValueError("The set of arrows is not completely in the domain of the target function")
        
        if Q0.isdisjoint(tgt.codomain):
            raise ValueError("The vertices aren't defined in the codomain of the target function")
        
        self.Q0 = Q0
        self.Q1 = Q1
        self.tgt = tgt
        self.src = src
        self.name = name if isinstance(name, str) else None

    def __str__(self):
        arrows = [f"{a}: {self.src(a)} → {self.tgt(a)}" for a in self.Q1]
        string_rep = ""
        string_rep += self.name if self.name else "Quiver"
        string_rep += " with vertices "
        
        # Check if Q0 is a named_set and has a name
        if isinstance(self.Q0, named_set) and self.Q0.name:
            string_rep += self.Q0.name
        else:
            string_rep += str(self.Q0)
            
        string_rep += " and arrows:\n  " + "\n  ".join(arrows)
        return string_rep
    
    def arrows_from(self, vertex):
        return {a for a in self.Q1 if self.src(a) == vertex}

    def arrows_to(self, vertex):
        return {a for a in self.Q1 if self.tgt(a) == vertex}

    def vertices(self):
        return self.Q0

    def arrows(self):
        return self.Q1
    
    @staticmethod
    def is_preorder(Q) -> bool:
        for v in Q.Q0:
            if not any(a for a in Q.Q1 if Q.src(a) == v and Q.tgt(a) == v):
                return False
        for a1 in Q.Q1:
            for a2 in Q.Q1:
                if Q.tgt(a1) == Q.src(a2):
                    x, z = Q.src(a1), Q.tgt(a2)
                    if not any(a3 for a3 in Q.Q1 if Q.src(a3) == x and Q.tgt(a3) == z):
                        return False
        return True




# Example usage of named_set, set_function, and quiver
vertices = set({"a", "b", "c"}, name="Vertices")
arrows = set({"f", "g", "h"}, name="Arrows")

# Define source and target functions
src_mapping = {"f": "a", "g": "b", "h": "c"}
tgt_mapping = {"f": "b", "g": "c", "h": "a"}

# Create source and target functions
src_func = set_function(arrows, vertices, src_mapping, name="src")
tgt_func = set_function(arrows, vertices, tgt_mapping, name="tgt")

# Create a quiver
my_quiver = quiver(vertices, arrows, src_func, tgt_func, name="Quiver 1")
print(my_quiver)

class Preorder(quiver):
    def __init__(self, Q0, Q1, src, tgt, name="Preorder"):
        super().__init__(Q0, Q1, src, tgt, name=name)

        # Reflexivity: every vertex must have a loop
        for v in self.Q0:
            if not any(a for a in self.Q1 if self.src(a) == v and self.tgt(a) == v):
                raise ValueError(f"Missing reflexive arrow at vertex '{v}'")

        # Transitivity: for every composable pair a1: x→y and a2: y→z,
        # there must exist a3: x→z
        for a1 in self.Q1:
            for a2 in self.Q1:
                if self.tgt(a1) == self.src(a2):
                    x = self.src(a1)
                    z = self.tgt(a2)
                    if not any(a3 for a3 in self.Q1 if self.src(a3) == x and self.tgt(a3) == z):
                        raise ValueError(
                            f"Missing transitive arrow: required from '{x}' to '{z}' due to path {a1} ∘ {a2}"
                        )

    def __str__(self):
        return f"{self.name} with {len(self.Q0)} elements and {len(self.Q1)} arrows"


# Example usage of Preorder
# Create a set of elements
elements = set({"x", "y", "z"}, name="Elements")

# Define the arrows for a preorder
# Need reflexive arrows (loops) for each element
# and arrows representing the order relation
preorder_arrows = set({
    "x_x", "y_y", "z_z",  # Reflexive arrows (loops)
    "x_y", "y_z", "x_z"   # Order relations (x ≤ y, y ≤ z, x ≤ z for transitivity)
}, name="Relations")

# Create source and target functions for the arrows
# Each arrow "a_b" means a ≤ b
source_map = {
    "x_x": "x", "y_y": "y", "z_z": "z",
    "x_y": "x", "y_z": "y", "x_z": "x"
}
target_map = {
    "x_x": "x", "y_y": "y", "z_z": "z",
    "x_y": "y", "y_z": "z", "x_z": "z"
}

# Create the source and target functions
source_func = set_function(preorder_arrows, elements, source_map, name="source")
target_func = set_function(preorder_arrows, elements, target_map, name="target")

# Create the preorder
my_preorder = Preorder(elements, preorder_arrows, source_func, target_func, name="≤ on {x,y,z}")
print(my_preorder)

# Test the helper methods
print(f"Arrows from 'y': {my_preorder.arrows_from('y')}")
print(f"Arrows to 'z': {my_preorder.arrows_to('z')}")
print(f"Elements: {my_preorder.vertices()}")
