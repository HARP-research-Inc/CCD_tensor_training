
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
        
        if not src.codomain.issubset(Q0):
            raise ValueError("The vertices aren't defined in the codomain of the source function")
        
        if not Q1.issubset(tgt.domain):
            raise ValueError("The set of arrows is not completely in the domain of the target function")
        
        if not tgt.codomain.issubset(Q1):
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

    def hasse_mixed_quiver(self):
        V, Q1, src, tgt = self.Q0, self.Q1, self.src, self.tgt

        eq_arrows    = set()  # will collect all (x,y) with x~y
        cover_arrows = set()  # will collect all covering (x,y), x<y w/o midpoints

        for x in V:
            for y in V:
                if x == y:
                    continue

                # is there arrow x→y?
                has_xy = any(a for a in Q1 if src(a)==x and tgt(a)==y)
                if not has_xy:
                    continue

                # is there arrow y→x? if so, mark as equivalence
                has_yx = any(a for a in Q1 if src(a)==y and tgt(a)==x)
                if has_yx:
                    eq_arrows.add((x,y))
                    continue

                # otherwise x<y strictly — check for an intermediate z
                mid = False
                for z in V - {x,y}:
                    if (any(a for a in Q1 if src(a)==x and tgt(a)==z)
                    and any(a for a in Q1 if src(a)==z and tgt(a)==y)):
                        mid = True
                        break
                if not mid:
                    cover_arrows.add((x,y))

        # combine them
        new_arrows = eq_arrows | cover_arrows

        # build fresh source/target maps
        src_map = { (x,y): x for (x,y) in new_arrows }
        tgt_map = { (x,y): y for (x,y) in new_arrows }

        src_fn = set_function(new_arrows, V, src_map, name=self.src.name+"_Hsrc")
        tgt_fn = set_function(new_arrows, V, tgt_map, name=self.tgt.name+"_Htgt")

        return quiver(V, new_arrows, src_fn, tgt_fn,
                      name=f"HasseMixed({self.name})")

class Poset(Preorder):
    """
    A partially ordered set (poset) is a preorder that is also antisymmetric:
    if x ≤ y and y ≤ x then x must equal y.
    """
    def __init__(self, Q0, Q1, src, tgt, name="Poset"):
        # First verify reflexivity + transitivity via the Preorder constructor
        super().__init__(Q0, Q1, src, tgt, name=name)

        # Antisymmetry check
        for x in self.Q0:
            for y in self.Q0:
                if x == y:
                    continue                     # loops are fine
                # Does an arrow x → y exist?
                x_le_y = any(a for a in self.Q1
                             if self.src(a) == x and self.tgt(a) == y)
                # Does an arrow y → x exist?
                y_le_x = any(a for a in self.Q1
                             if self.src(a) == y and self.tgt(a) == x)
                if x_le_y and y_le_x:
                    raise ValueError(
                        f"Antisymmetry violated: both {x} ≤ {y} and {y} ≤ {x} "
                        "present, so the relation is not a poset."
                    )

        # Thinness check
        if sum(1 for a in self.Q1 if self.src(a)==x and self.tgt(a)==y) > 1:
            raise ValueError("Non‑thin: multiple arrows from {x} to {y}")

    def __str__(self):
        return f"{self.name} with {len(self.Q0)} elements and {len(self.Q1)} arrows"
    
    def hasse_quiver(self):
        """
        Return a new quiver whose arrows are exactly the covering relations
        of this poset (i.e. the Hasse diagram).
        """
        # 1) find all covering pairs (x,y):
        covers = set()
        for x in self.Q0:
            for y in self.Q0:
                if x == y:
                    continue

                # check x ≤ y (i.e. some arrow x→y exists in the transitive closure)
                if any(a for a in self.Q1 if self.src(a) == x and self.tgt(a) == y):
                    # now ensure there's no z with x < z < y
                    is_cover = True
                    for z in self.Q0:
                        if z == x or z == y:
                            continue
                        if (any(a1 for a1 in self.Q1 if self.src(a1) == x and self.tgt(a1) == z)
                         and any(a2 for a2 in self.Q1 if self.src(a2) == z and self.tgt(a2) == y)):
                            is_cover = False
                            break
                    if is_cover:
                        covers.add((x, y))

        # 2) build a fresh quiver on the same vertices but only these arrows
        arrows   = covers
        src_map  = { (x,y): x for (x,y) in covers }
        tgt_map  = { (x,y): y for (x,y) in covers }
        src_fn   = set_function(arrows, self.Q0, src_map, name="hasse_src")
        tgt_fn   = set_function(arrows, self.Q0, tgt_map, name="hasse_tgt")
        return quiver(self.Q0, arrows, src_fn, tgt_fn, name=f"Hasse({self.name})")
