def visualize_preorder(threads, eqs):
    """
    threads: List of lists of node‐labels, e.g. [['A','B','C'], ['D','E']]
    eqs:     List of (t1,d1,t2,d2) meaning threads[t1][d1] ≡ threads[t2][d2].
    """
    # 1) Build DSU to find connected groups of threads
    parent = list(range(len(threads)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for t1,d1,t2,d2 in eqs:
        union(t1, t2)
    groups = {}
    for i in range(len(threads)):
        groups.setdefault(find(i), []).append(i)

    # 2) Helpers to build each row-string for a single chain
    def node_row(chain, stretch_to=None):
        """Draw a row of nodes, with optional stretching to match another chain length"""
        m = len(chain)
        # If stretching is requested, use the target length
        if stretch_to and stretch_to > m:
            width = (stretch_to-1)*4 + 1
        else:
            width = (m-1)*4 + 1
            
        row = [' ']*width
        for i, lbl in enumerate(chain):
            row[i*4] = lbl
        return ''.join(row)

    def edge_row(chain, stretch_to=None):
        """Draw edges between nodes, with optional stretching to match another chain length"""
        # e.g. for 3 nodes: "|->-|->-|"
        m = len(chain)
        # If stretching is requested, use the target length
        if stretch_to and stretch_to > m:
            width = (stretch_to-1)*4 + 1
            final_pipe_pos = width
        else:
            width = (m-1)*4 + 1
            final_pipe_pos = (m-1)*4
            
        row = [' ']*width
        for i in range(m-1):
            seg = "|->-"
            for j,ch in enumerate(seg):
                row[i*4 + j] = ch
        
        # Add final pipe character
        if final_pipe_pos < width:
            row[final_pipe_pos] = '|'
            
        return ''.join(row)

    def eq_row(depth, width, align_pos=None):
        # Create a row with a '|' at position depth*4
        row = [' ']*width
        row[depth*4] = '|'
        return ''.join(row)

    # 3) For each group, either draw as one tree (if eqs) or as standalone chains
    out_lines = []
    for group in groups.values():
        # collect eqs restricted to this group
        sub_eqs = [e for e in eqs if e[0] in group and e[2] in group]

        if not sub_eqs:
            # completely independent chains → blank line between them
            for t in group:
                out_lines.append(node_row(threads[t]))
                out_lines.append(edge_row(threads[t]))
                out_lines.append("")    # gap
            out_lines.pop()              # drop last extra blank
        else:
            # build adjacency for eq-tree
            adj = {t: [] for t in group}
            for t1,d1,t2,d2 in sub_eqs:
                adj[t1].append((t2,d1,d2))
                adj[t2].append((t1,d2,d1))

            # Store thread depth info and positions for proper alignment
            thread_info = {}
            visited = set()
            
            def dfs(t, from_t=None, depth=0):
                if t in visited:
                    return
                visited.add(t)
                
                # Record this thread's position in output
                start_line = len(out_lines)
                thread_info[t] = {
                    'start_line': start_line,
                    'depth': depth,
                    'width': (len(threads[t])-1)*4 + 1
                }
                
                # Draw this chain
                out_lines.append(node_row(threads[t]))
                out_lines.append(edge_row(threads[t]))
                
                # Follow equivalence edges
                for (nbr, d_here, d_nbr) in adj[t]:
                    if nbr == from_t:
                        continue
                    
                    # Draw vertical connection
                    conn_width = max((len(threads[t])-1)*4 + 1, d_here*4 + 1)
                    out_lines.append(eq_row(d_here, conn_width))
                    
                    # Continue DFS with neighbor
                    dfs(nbr, t, depth+1)

            # Start DFS at the first thread in group
            dfs(group[0])
        
        # Separate groups
        out_lines.append("")

    # Drop trailing blank
    if out_lines and out_lines[-1]=="":
        out_lines.pop()

    print("\n".join(out_lines))


# ——— example usage:

threads = [
    ['A','B','C'],   # thread 0
    ['D','E'],       # thread 1
]
# say A≡D  ⇒ (0,0) ≡ (1,0)
equivalences = [
    (0,0, 1,0)
]

visualize_preorder(threads, equivalences)

# More complex example
threads_complex = [
    ['A', 'B', 'C', 'D'],         # thread 0
    ['E', 'F', 'G'],              # thread 1
    ['H', 'I', 'J', 'K'],         # thread 2
    ['L', 'M', 'N'],              # thread 3
]

# Structured equivalences that form a tree
equivalences_complex = [
    (0, 0, 1, 0),  # A≡E (connects thread 0 and 1)
    (0, 2, 2, 0),  # C≡H (connects thread 0 and 2)
    (2, 3, 3, 0),  # K≡L (connects thread 2 and 3)
]

print("\nMore complex example:")
visualize_preorder(threads_complex, equivalences_complex)

# After our complex example, add a simple test case with different length threads
print("\nDifferent length threads with equivalent endpoints:")
threads_endpoints = [
    ['A', 'B'],         # thread 0 (shorter)
    ['C', 'D', 'E'],    # thread 1 (longer)
]

# Equivalent endpoints
endpoints_eq = [
    (0, 1, 1, 2),  # B≡E (the endpoints)
]

visualize_preorder(threads_endpoints, endpoints_eq)

# Now let's add another complex example
print("\nMultiple threads with different lengths:")
threads_mixed = [
    ['A', 'B', 'C'],          # thread 0
    ['D', 'E', 'F', 'G', 'H'], # thread 1 (longer)
    ['I', 'J'],               # thread 2 (shorter)
]

# Equivalences connecting threads of different lengths
mixed_eq = [
    (0, 2, 1, 4),  # C≡H (endpoint of 0 with endpoint of 1)
    (1, 2, 2, 1),  # F≡J (middle of 1 with endpoint of 2)
]

visualize_preorder(threads_mixed, mixed_eq)
