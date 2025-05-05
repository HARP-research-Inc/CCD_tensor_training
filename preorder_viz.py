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

    # 2) Find positions where nodes should be placed for alignment
    def calc_positions(group, eqs_in_group):
        """Calculate positions for each node in the threads."""
        # Start with default positions (4 spaces between nodes)
        positions = {}  # (thread_idx, node_idx) -> position
        for t in group:
            for i in range(len(threads[t])):
                positions[(t, i)] = i * 4
        
        # Process each equivalence relation
        for t1, d1, t2, d2 in eqs_in_group:
            pos1 = positions[(t1, d1)]
            pos2 = positions[(t2, d2)]
            
            # Find the maximum position
            max_pos = max(pos1, pos2)
            
            # Determine how much stretching is needed for each thread
            if pos1 < max_pos:
                # Thread 1 needs stretching
                stretch(t1, d1, max_pos, positions)
            
            if pos2 < max_pos:
                # Thread 2 needs stretching
                stretch(t2, d2, max_pos, positions)
        
        return positions

    def stretch(thread_idx, node_idx, target_pos, positions):
        """Stretch a thread so that the specified node is at the target position."""
        current_pos = positions[(thread_idx, node_idx)]
        
        if current_pos == target_pos:
            return  # Already at target position
        
        # Calculate the stretch factor for this thread
        if node_idx > 0:
            # Stretch nodes before the aligned node
            original_width = current_pos
            stretch_factor = target_pos / original_width
            
            for i in range(node_idx):
                # Adjust all previous positions proportionally
                original_i_pos = i * 4
                new_i_pos = int(original_i_pos * stretch_factor)
                positions[(thread_idx, i)] = new_i_pos
        
        # Set the aligned node to the target position
        positions[(thread_idx, node_idx)] = target_pos
        
        # Adjust positions of all nodes after the aligned node
        for i in range(node_idx + 1, len(threads[thread_idx])):
            positions[(thread_idx, i)] = target_pos + (i - node_idx) * 4

    # 3) Helpers to draw the nodes and edges
    def node_row(thread_idx, positions):
        """Draw a row of nodes using calculated positions."""
        thread = threads[thread_idx]
        if not thread:
            return ""
            
        # Find maximum position
        max_pos = max(positions.get((thread_idx, i), 0) for i in range(len(thread)))
        width = max_pos + 1
        
        row = [' '] * width
        for i, lbl in enumerate(thread):
            pos = positions.get((thread_idx, i), i*4)
            if pos < width:
                row[pos] = lbl
        
        return ''.join(row)

    def edge_row(thread_idx, positions):
        """Draw edges between nodes using calculated positions."""
        thread = threads[thread_idx]
        if len(thread) <= 1:
            return ""
            
        # Find maximum position
        max_pos = max(positions.get((thread_idx, i), 0) for i in range(len(thread)))
        width = max_pos + 1
        
        row = [' '] * width
        
        # For each consecutive pair of nodes
        for i in range(len(thread)-1):
            pos1 = positions.get((thread_idx, i), i*4)
            pos2 = positions.get((thread_idx, i+1), (i+1)*4)
            
            # Place the vertical bars
            if pos1 < width:
                row[pos1] = '|'
            if pos2 < width:
                row[pos2] = '|'
            
            # Fill in the arrow and dashes between them
            if pos2 - pos1 > 3:  # Enough space for proper arrow
                middle = pos1 + (pos2 - pos1) // 2
                
                # Place arrow in middle
                if middle > pos1 and middle < pos2:
                    row[middle - 1:middle + 2] = ['-', '>', '-']
                
                # Fill with dashes (leaving space for arrow)
                for j in range(pos1+1, middle-1):
                    row[j] = '-'
                for j in range(middle+2, pos2):
                    row[j] = '-'
            elif pos2 - pos1 == 3:  # Just enough for >
                row[pos1+1:pos1+3] = ['-', '>']
            elif pos2 - pos1 == 2:  # Just enough for >
                row[pos1+1] = '>'
            # If gap is 1, just have two |'s next to each other
        
        return ''.join(row)

    def connection_row(t1, d1, t2, d2, positions):
        """Draw a vertical connection between equivalent nodes."""
        pos1 = positions.get((t1, d1))
        max_pos = max(pos1, positions.get((t2, d2)))
        width = max_pos + 1
        
        row = [' '] * width
        if pos1 < width:
            row[pos1] = '|'
        
        return ''.join(row)

    # 4) Generate the visualization
    out_lines = []
    for group in groups.values():
        # Collect equivalences for this group
        eqs_in_group = [e for e in eqs if e[0] in group and e[2] in group]
        
        if not eqs_in_group:
            # Independent chains - simple layout
            for t in group:
                thread = threads[t]
                out_lines.append("   ".join(thread))
                out_lines.append("|" + "->-|".join([""] * len(thread)))
                out_lines.append("")
            
        else:
            # Connected threads - calculate positions for alignment
            positions = calc_positions(group, eqs_in_group)
            
            # Build adjacency list for DFS traversal
            adj = {t: [] for t in group}
            for t1, d1, t2, d2 in eqs_in_group:
                adj[t1].append((t2, d1, d2))
                adj[t2].append((t1, d2, d1))
            
            visited = set()
            
            def dfs(t, from_t=None):
                if t in visited:
                    return
                visited.add(t)
                
                # Draw this thread
                out_lines.append(node_row(t, positions))
                out_lines.append(edge_row(t, positions))
                
                # Follow equivalence edges
                for (nbr, d_here, d_nbr) in adj[t]:
                    if nbr == from_t:
                        continue
                    
                    # Draw connection
                    out_lines.append(connection_row(t, d_here, nbr, d_nbr, positions))
                    
                    # Process neighbor
                    dfs(nbr, t)
            
            # Start DFS from first thread in group
            dfs(group[0])
        
        # Add blank line between groups
        out_lines.append("")
    
    # Remove trailing blank line
    if out_lines and out_lines[-1] == "":
        out_lines.pop()
    
    result = "\n".join(out_lines)
    print(result)
    
    # Also write to file for inspection
    try:
        with open("preorder_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        print(f"Note: Could not write to file: {e}")


# ——— example usage:

# Connection between endpoints of threads with different lengths
print("Connecting endpoints of threads with different lengths:")

# Example 1: Connect end of 2-node thread to end of 3-node thread
threads1 = [
    ['A', 'B'],           # thread 0 (length 2)
    ['C', 'D', 'E'],      # thread 1 (length 3)
]
# B≡E (connect endpoint of thread 0 with endpoint of thread 1)
eq1 = [
    (0, 1, 1, 2)  # B≡E
]
visualize_preorder(threads1, eq1)

# Example 2: Connect end of 3-node thread to end of 5-node thread
print("\nConnecting end of 3-node thread to end of 5-node thread:")
threads2 = [
    ['A', 'B', 'C'],                # thread 0 (length 3)
    ['D', 'E', 'F', 'G', 'H'],      # thread 1 (length 5)
]
# C≡H (connect endpoint to endpoint)
eq2 = [
    (0, 2, 1, 4)  # C≡H
]
visualize_preorder(threads2, eq2)

# Example 3: More complex case with three threads
print("\nThree connected threads of different lengths:")
threads3 = [
    ['A', 'B', 'C'],            # thread 0 (length 3)
    ['D', 'E', 'F', 'G'],       # thread 1 (length 4)
    ['H', 'I'],                 # thread 2 (length 2)
]
# Connect endpoints: C≡G, B≡I
eq3 = [
    (0, 2, 1, 3),  # C≡G
    (0, 1, 2, 1)   # B≡I
]
visualize_preorder(threads3, eq3)

# Example 3: A very specific test case to demonstrate proper stretching
print("\nAlign test with stretching:")

# Thread 0: length 2
# Thread 1: length 3
# Thread 2: length 2
# Connect: (0,1) ≡ (1,2), (0,0) ≡ (2,1)
threads_test = [
    ['A', 'B'],      # Thread 0 
    ['C', 'D', 'E'], # Thread 1
    ['F', 'G'],      # Thread 2
]

test_eq = [
    (0, 1, 1, 2),  # B≡E
    (0, 0, 2, 1),  # A≡G
]

# Manual implementation for this case
def visualize_stretch_test():
    # Manually align the threads with proper stretching
    output = []
    
    # Thread 0 with B aligned to position 8
    output.append("A       B")
    output.append("|------>|")
    output.append("|       |")
    
    # Thread 1 with E aligned to position 8
    output.append("C   D   E")
    output.append("|->-|->-|")
    
    # Thread 2 with G aligned to position 4
    output.append("F   G")
    output.append("|->-|")
    
    print("\n".join(output))

# Try both the algorithm and the manual implementation
visualize_preorder(threads_test, test_eq)
print("\nManual implementation for comparison:")
visualize_stretch_test()
