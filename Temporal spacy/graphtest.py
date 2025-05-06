import networkx as nx
import matplotlib.pyplot as plt

# Helper to draw a digraph with a title
def draw_digraph(G, title):
    plt.figure()
    pos = nx.spring_layout(G, seed=42)  # layout (fixed seed just for reproducibility)
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title(title)
    plt.axis('off')
    plt.show()

# -------------------------------------------------------------------
# 1) Valid Chisari quiver (equivalences {a,d}, {c,e}; covers [ad]->b->[ce])
G1 = nx.DiGraph()
G1.add_nodes_from(['a', 'd', 'b', 'c', 'e'])
# 2‑cycles inside equivalence classes
G1.add_edge('a', 'd'); G1.add_edge('d', 'a')
G1.add_edge('c', 'e'); G1.add_edge('e', 'c')
# cover arrows between classes
G1.add_edge('a', 'b')
G1.add_edge('b', 'c')
draw_digraph(G1, "Valid Chisari quiver – example 1")

# -------------------------------------------------------------------
# 2) Valid Chisari quiver with no equivalence classes (simple chain a→b→c)
G2 = nx.DiGraph()
G2.add_nodes_from(['a', 'b', 'c'])
G2.add_edge('a', 'b')
G2.add_edge('b', 'c')
draw_digraph(G2, "Valid Chisari quiver – example 2")

# -------------------------------------------------------------------
# 3) Non‑example: contains redundant transitive edge a→c
G3 = nx.DiGraph()
G3.add_nodes_from(['a', 'b', 'c'])
G3.add_edge('a', 'b')
G3.add_edge('b', 'c')
G3.add_edge('a', 'c')  # transitive extra edge ⟹ not minimal
draw_digraph(G3, "Non‑example – redundant transitive edge")

# -------------------------------------------------------------------
# 4) Non‑example: equivalence 2‑cycle plus a redundant cross‑class edge
G4 = nx.DiGraph()
G4.add_nodes_from(['x', 'y', 'z'])
# 2‑cycle (equivalence) between x and y
G4.add_edge('x', 'y'); G4.add_edge('y', 'x')
# cover x→z would be valid; here we add both x→z and y→z (duplicate)
G4.add_edge('x', 'z')
G4.add_edge('y', 'z')  # duplicates information already carried via x↔y and x→z
draw_digraph(G4, "Non‑example – duplicate cover across equivalence class")
