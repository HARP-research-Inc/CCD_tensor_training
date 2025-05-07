def visualize_preorder(threads, eqs):
    """
    threads: List of lists of node‐labels, e.g. [['A','B','C'], ['D','E']]
    eqs:     List of (t1,d1,t2,d2) meaning threads[t1][d1] ≡ threads[t2][d2].
    """
    import math
    from functools import reduce
    
    # Helper function to find GCD of a list of numbers
    def find_gcd(numbers):
        return reduce(math.gcd, numbers)
    
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

    # 2) Calculate grid size and positions using GCD
    def calc_grid_positions(group, eqs_in_group):
        """Calculate positions based on GCD of segment lengths."""
        # Find all the lengths between equivalence points
        segment_lengths = []
        
        # Add the default segment length (4 units between nodes)
        base_segment = 4
        segment_lengths.append(base_segment)
        
        # Add segment lengths from equivalence relations
        for t1, d1, t2, d2 in eqs_in_group:
            # Default positions these nodes would have
            default_pos1 = d1 * base_segment
            default_pos2 = d2 * base_segment
            
            # This is a segment length that needs to be considered
            segment_len = abs(default_pos2 - default_pos1)
            if segment_len > 0:
                segment_lengths.append(segment_len)
        
        # Find the GCD of all segment lengths to determine grid unit size
        if len(segment_lengths) > 1:
            grid_unit = find_gcd(segment_lengths)
        else:
            grid_unit = base_segment  # Default grid unit
        
        # Build a graph of equivalence relationships
        equiv_pairs = [(t1, d1, t2, d2) for t1, d1, t2, d2 in eqs_in_group]
        
        # Calculate positions based on the grid
        positions = {}  # (thread_idx, node_idx) -> position
        
        # First, assign default positions
        for t in group:
            for i in range(len(threads[t])):
                positions[(t, i)] = i * base_segment
        
        # Process equivalence relationships to adjust positions
        for t1, d1, t2, d2 in equiv_pairs:
            # Determine positions for equivalent nodes
            pos1 = positions[(t1, d1)]
            pos2 = positions[(t2, d2)]
            
            # If positions differ, we need to adjust
            if pos1 != pos2:
                max_pos = max(pos1, pos2)
                
                # Adjust thread with the lower position
                if pos1 < max_pos:
                    # Scale positions in thread t1
                    scale_and_align(t1, d1, max_pos, positions, grid_unit, threads[t1])
                
                if pos2 < max_pos:
                    # Scale positions in thread t2
                    scale_and_align(t2, d2, max_pos, positions, grid_unit, threads[t2])
        
        return positions
    
    def scale_and_align(t, d, target_pos, positions, grid_unit, thread):
        """Scale positions in a thread to align a specific node at target_pos."""
        current_pos = positions[(t, d)]
        
        # Calculate scaling factor based on grid_unit
        if d > 0:
            # Calculate a scaling factor that preserves the grid
            scale_factor = target_pos / current_pos
            
            # Scale positions of nodes before the alignment point
            for i in range(d):
                positions[(t, i)] = int(positions[(t, i)] * scale_factor)
                # Ensure positions fall on grid lines
                positions[(t, i)] = round(positions[(t, i)] / grid_unit) * grid_unit
        
        # Set the aligned position exactly
        positions[(t, d)] = target_pos
        
        # Adjust positions after the alignment point
        for i in range(d + 1, len(thread)):
            # Keep consistent spacing
            prev_gap = positions[(t, i)] - positions[(t, i-1)]
            # Ensure the gap is a multiple of grid_unit
            adjusted_gap = max(grid_unit, round(prev_gap / grid_unit) * grid_unit)
            positions[(t, i)] = positions[(t, i-1)] + adjusted_gap

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
            gap = pos2 - pos1
            
            if gap > 4:  # Enough space for proper arrow with dashes
                # Put the arrow in the middle
                arrow_pos = pos1 + gap // 2
                if arrow_pos > pos1 + 1 and arrow_pos < pos2 - 1:
                    row[arrow_pos] = '>'
                    
                    # Fill with dashes
                    for j in range(pos1+1, arrow_pos):
                        row[j] = '-'
                    for j in range(arrow_pos+1, pos2):
                        row[j] = '-'
                        
            elif gap == 4:  # Space for ---->
                row[pos1+1:pos1+4] = ['-', '-', '>']
            elif gap == 3:  # Space for -->
                row[pos1+1:pos1+3] = ['-', '>']
            elif gap == 2:  # Just enough for >
                row[pos1+1] = '>'
            
        return ''.join(row)

    def connection_row(t1, d1, t2, d2, positions):
        """Draw a vertical connection between equivalent nodes."""
        pos1 = positions.get((t1, d1))
        pos2 = positions.get((t2, d2))
        
        # For identical positions, just draw a vertical bar
        if pos1 == pos2:
            width = pos1 + 1
            row = [' '] * width
            row[pos1] = '|'
            return ''.join(row)
        
        # Otherwise, calculate width needed
        max_pos = max(pos1, pos2)
        width = max_pos + 1
        
        row = [' '] * width
        
        # Mark both positions with a vertical line if they're different
        row[pos1] = '|'
        row[pos2] = '|'
        
        # Draw a horizontal connector between the two positions
        start, end = min(pos1, pos2), max(pos1, pos2)
        for i in range(start+1, end):
            row[i] = '-'
            
        # Add corner pieces if space allows
        if end - start >= 2:
            if start == pos1:
                row[start+1] = '>'  # Right arrow
            else:
                row[end-1] = '<'    # Left arrow
        
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
            # Connected threads - calculate positions using grid approach
            positions = calc_grid_positions(group, eqs_in_group)
            
            # Build adjacency list for DFS traversal
            adj = {t: [] for t in group}
            for t1, d1, t2, d2 in eqs_in_group:
                adj[t1].append((t2, d1, d2))
                adj[t2].append((t1, d2, d1))
            
            # Keep track of visited threads and connections
            visited_threads = set()
            visited_connections = set()  # Track connections we've already drawn
            thread_positions = {}  # Track line number where each thread starts
            
            # Track which positions are occupied in the output grid
            occupied_positions = {}  # (line_num, col_pos) -> True if occupied
            
            def mark_thread_occupied(t, line_start):
                """Mark all positions occupied by this thread's nodes and edges."""
                thread = threads[t]
                thread_positions[t] = line_start
                
                # Mark node positions
                for i in range(len(thread)):
                    pos = positions.get((t, i), i*4)
                    occupied_positions[(line_start, pos)] = True
                    # Also mark the row below (for the edge)
                    occupied_positions[(line_start+1, pos)] = True
                
            def find_path(t1, d1, t2, d2):
                """Find a path to connect two nodes, prioritizing horizontal routing through empty spaces."""
                start_thread_line = thread_positions[t1]
                end_thread_line = thread_positions[t2]
                
                start_pos = positions[(t1, d1)]
                end_pos = positions[(t2, d2)]
                
                # If the threads are adjacent, use a simple direct connection
                if abs(start_thread_line - end_thread_line) <= 3:
                    # Simple case - just draw a direct line
                    return [(start_thread_line+2, start_pos, end_pos)]
                
                # For the N=L case and similar, route through intermediate thread's empty space
                # Find all threads between start and end threads
                intermediate_threads = []
                for t in group:
                    thread_line = thread_positions.get(t)
                    if thread_line is not None and start_thread_line < thread_line < end_thread_line:
                        # Calculate the rightmost position of this thread
                        thread_max_pos = 0
                        for i in range(len(threads[t])):
                            thread_max_pos = max(thread_max_pos, positions.get((t, i), 0))
                        
                        # Add this thread and its info to our list
                        intermediate_threads.append((thread_line, t, thread_max_pos))
                
                # Sort the threads by line position
                intermediate_threads.sort()
                
                # Special case: can we route horizontally through the last intermediate thread?
                if intermediate_threads:
                    # Get the last intermediate thread before the destination
                    thread_line, thread_t, thread_max_pos = intermediate_threads[-1]
                    
                    # Check if both start_pos and end_pos are to the right of this thread's end
                    if start_pos > thread_max_pos and end_pos > thread_max_pos:
                        # We can route horizontally through this thread's empty space
                        # For clarity, route directly at the thread's line
                        path = []
                        
                        # Vertical line from start down to the intermediate thread
                        for i in range(start_thread_line+2, thread_line+1):
                            path.append((i, start_pos, start_pos))
                        
                        # Horizontal line along the empty space of the intermediate thread
                        path.append((thread_line, start_pos, end_pos))
                        
                        # Vertical line from intermediate thread to end
                        for i in range(thread_line+1, end_thread_line+1):
                            path.append((i, end_pos, end_pos))
                        
                        return path
                
                # Default fallback - route by going all the way to the right
                # Find the rightmost position in any thread
                max_pos = max(start_pos, end_pos)
                for _, _, pos in intermediate_threads:
                    max_pos = max(max_pos, pos + 2)  # Add padding
                
                # Create a path that goes around everything
                path = []
                
                # Go right from the start node if needed
                if start_pos < max_pos:
                    path.append((start_thread_line+1, start_pos, max_pos))
                
                # Go down
                for i in range(start_thread_line+2, end_thread_line):
                    path.append((i, max_pos, max_pos))
                
                # Go left to the end node if needed
                path.append((end_thread_line, max_pos, end_pos))
                
                return path
            
            def draw_connection(t1, d1, t2, d2):
                """Draw a connection between two equivalent nodes, routing through empty spaces when possible."""
                # Get thread positions
                start_thread_line = thread_positions[t1]
                end_thread_line = thread_positions[t2]
                
                # Get node positions
                start_pos = positions[(t1, d1)]
                end_pos = positions[(t2, d2)]
                
                # For adjacent threads, just add a vertical connector
                if abs(start_thread_line - end_thread_line) <= 3:
                    # Create a connector line
                    max_width = max(start_pos, end_pos) + 1
                    row = [' '] * max_width
                    
                    # Draw the vertical connector
                    if start_pos == end_pos:
                        # Simple vertical line
                        row[start_pos] = '|'
                    else:
                        # Connection with horizontal component
                        left, right = min(start_pos, end_pos), max(start_pos, end_pos)
                        row[left] = '|'
                        row[right] = '|'
                        
                        # Fill with connecting dashes
                        for i in range(left+1, right):
                            row[i] = '-'
                    
                    # Insert the connection line
                    line_idx = start_thread_line + 2
                    
                    # Ensure we have enough lines
                    while len(out_lines) <= line_idx:
                        out_lines.append("")
                    
                    # Merge with existing content if any
                    if line_idx < len(out_lines) and out_lines[line_idx]:
                        existing = out_lines[line_idx]
                        merged = []
                        for i in range(max(len(existing), max_width)):
                            if i < len(existing) and existing[i] != ' ':
                                merged.append(existing[i])
                            elif i < max_width and row[i] != ' ':
                                merged.append(row[i])
                            else:
                                merged.append(' ')
                        out_lines[line_idx] = ''.join(merged)
                    else:
                        # Add new line
                        out_lines[line_idx] = ''.join(row)
                    
                    return
                
                # Get an optimized path
                path = find_path(t1, d1, t2, d2)
                
                # Draw each segment of the path
                for line_num, start_p, end_p in path:
                    # Ensure we have enough lines
                    while len(out_lines) <= line_num:
                        out_lines.append("")
                    
                    # Create a row for this segment
                    max_width = max(start_p, end_p) + 1
                    if line_num < len(out_lines):
                        max_width = max(max_width, len(out_lines[line_num]))
                    
                    row = [' '] * max_width
                    
                    # Draw the appropriate segment
                    if start_p == end_p:
                        # Vertical segment
                        row[start_p] = '|'
                    else:
                        # Horizontal segment with possible arrow
                        left, right = min(start_p, end_p), max(start_p, end_p)
                        
                        # Add endpoints
                        row[left] = '|'
                        row[right] = '|'
                        
                        # Fill with dashes and maybe an arrow
                        if right - left > 2:
                            # Add an arrow in the middle
                            mid = left + (right - left) // 2
                            for i in range(left+1, right):
                                if i == mid:
                                    # Determine direction based on start/end
                                    row[i] = '>' if start_p < end_p else '<'
                                else:
                                    row[i] = '-'
                        elif right - left == 2:
                            row[left+1] = '-'
                    
                    # Merge with existing content
                    if line_num < len(out_lines) and out_lines[line_num]:
                        existing = out_lines[line_num]
                        merged = []
                        for i in range(max(len(existing), max_width)):
                            if i < len(existing) and existing[i] != ' ':
                                merged.append(existing[i])
                            elif i < max_width and row[i] != ' ':
                                merged.append(row[i])
                            else:
                                merged.append(' ')
                        out_lines[line_num] = ''.join(merged)
                    else:
                        # Add new line
                        out_lines[line_num] = ''.join(row)
            
            def dfs(t, line_num, from_t=None):
                if t in visited_threads:
                    return
                visited_threads.add(t)
                
                # Draw this thread
                node_line = node_row(t, positions)
                edge_line = edge_row(t, positions)
                
                # If we need to insert lines, do it at the right position
                if line_num < len(out_lines):
                    out_lines.insert(line_num, node_line)
                    out_lines.insert(line_num + 1, edge_line)
                else:
                    # Append at the end
                    out_lines.append(node_line)
                    out_lines.append(edge_line)
                
                # Mark this thread's positions as occupied
                mark_thread_occupied(t, line_num)
                
                # Leave a blank line for connections
                if line_num + 2 >= len(out_lines):
                    out_lines.append("")
                
                # Process equivalence edges
                next_line = line_num + 3  # Start next thread after the blank line
                
                for (nbr, d_here, d_nbr) in adj[t]:
                    # Create a unique identifier for this connection
                    conn_id = tuple(sorted([(t, d_here), (nbr, d_nbr)]))
                    
                    # Skip if we've already processed this connection or it's the one we came from
                    if conn_id in visited_connections or nbr == from_t:
                        continue
                    
                    # Mark this connection as visited
                    visited_connections.add(conn_id)
                    
                    # Process neighbor first if we haven't seen it yet
                    if nbr not in visited_threads:
                        dfs(nbr, next_line, t)
                        next_line = len(out_lines) + 1  # Update for the next thread
                    
                    # Draw connection between the threads
                    draw_connection(t, d_here, nbr, d_nbr)
            
            # Start DFS from first thread in group
            # But make sure we visit all components in the group
            line_num = 0
            for start_t in group:
                if start_t not in visited_threads:
                    dfs(start_t, line_num)
                    line_num = len(out_lines) + 1  # Update for the next disconnected component
            
            # Process any remaining connections that might cross between components
            # This ensures connections between different DFS trees are drawn
            for t1, d1, t2, d2 in eqs_in_group:
                conn_id = tuple(sorted([(t1, d1), (t2, d2)]))
                if conn_id not in visited_connections:
                    visited_connections.add(conn_id)
                    # Both threads should be visited by now
                    if t1 in thread_positions and t2 in thread_positions:
                        draw_connection(t1, d1, t2, d2)
        
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

# Let's create a complex example with multiple threads of different lengths
print("\nComplex example with grid alignment:")
complex_threads = [
    ['A', 'B', 'C', 'D'],       # Thread 0 (length 4)
    ['E', 'F', 'G'],            # Thread 1 (length 3)
    ['H', 'I', 'J', 'K', 'L'],  # Thread 2 (length 5)
    ['M', 'N'],                 # Thread 3 (length 2)
]

complex_eq = [
    (0, 3, 2, 4),  # D≡L (connect endpoints)
    (1, 0, 3, 0),  # E≡M (connect first nodes)
    (1, 2, 2, 2),  # G≡J (connect middle nodes)
]

visualize_preorder(complex_threads, complex_eq)

# modified version of last example but N=L
print("\nModified version of last example but N=L:")
complex_threads2 = [
    ['A', 'B', 'C', 'D'],       # Thread 0 (length 4)
    ['E', 'F', 'G'],            # Thread 1 (length 3)
    ['H', 'I', 'J', 'K', 'L'],  # Thread 2 (length 5)
    ['M', 'N'],                 # Thread 3 (length 2)
]

complex_eq2 = [
    (0, 3, 2, 4),  # D≡L (connect endpoints)
    (1, 0, 3, 0),  # E≡M (connect first nodes)
    (1, 2, 2, 2),  # G≡J (connect middle nodes)
    (3, 1, 2, 4),  # N≡L (connect last node of thread 3 to last node of thread 2)
]

visualize_preorder(complex_threads2, complex_eq2)

# Example with a deliberate cycle to test grid handling
print("\nExample with cycle (N≡L≡D):")
cycle_threads = [
    ['A', 'B', 'C', 'D'],       # Thread 0 (length 4)
    ['H', 'I', 'J', 'K', 'L'],  # Thread 1 (length 5)
    ['M', 'N'],                 # Thread 2 (length 2)
]

cycle_eq = [
    (0, 3, 1, 4),  # D≡L (connect endpoints)
    (2, 1, 1, 4),  # N≡L (connect N to L as well)
]

visualize_preorder(cycle_threads, cycle_eq)

# Example with a triangle of connections
print("\nTriangle of connections:")
triangle_threads = [
    ['A', 'B', 'C'],  # Thread 0
    ['D', 'E', 'F'],  # Thread 1
    ['G', 'H', 'I'],  # Thread 2
]

triangle_eq = [
    (0, 2, 1, 2),  # C≡F
    (1, 2, 2, 2),  # F≡I
    (0, 2, 2, 2),  # C≡I (creates a cycle/triangle)
]

visualize_preorder(triangle_threads, triangle_eq)

# Example with a complex network of connections
print("\nComplex network of connections:")
network_threads = [
    ['A', 'B', 'C', 'D'],  # Thread 0
    ['E', 'F', 'G', 'H'],  # Thread 1
    ['I', 'J', 'K', 'L'],  # Thread 2
    ['M', 'N', 'O', 'P'],  # Thread 3
]

network_eq = [
    (0, 0, 1, 0),  # A≡E (connect first nodes of thread 0 and 1)
    (0, 3, 2, 3),  # D≡L (connect last nodes of thread 0 and 2)
    (1, 3, 3, 3),  # H≡P (connect last nodes of thread 1 and 3)
    (2, 0, 3, 0),  # I≡M (connect first nodes of thread 2 and 3)
    (0, 0, 3, 0),  # A≡M (creates a loop/cycle)
    (0, 3, 3, 3),  # D≡P (creates another loop/cycle)
]

visualize_preorder(network_threads, network_eq)

# Example specifically designed to show routing through empty spaces
print("\nRouting through empty spaces:")
routing_threads = [
    ['A', 'B', 'C', 'D'],  # Thread 0
    ['E', 'F'],            # Thread 1 (short thread with space on the right)
    ['G', 'H', 'I', 'J'],  # Thread 2
]

routing_eq = [
    (0, 0, 1, 0),  # A≡E (top left connection)
    (0, 3, 2, 3),  # D≡J (should route through the empty space in thread 1)
]

visualize_preorder(routing_threads, routing_eq)

# Specific test case for N=L routing through thread 1's empty space
print("\nSpecific test case for routing N=L through empty space:")
routing_test = [
    ['A', 'B', 'C', 'D'],       # Thread 0 
    ['E', 'F', 'G'],            # Thread 1 (shorter than others, has empty space)
    ['H', 'I', 'J', 'K', 'L'],  # Thread 2
    ['M', 'N'],                 # Thread 3 (short, N should connect to L through thread 1's space)
]

routing_test_eq = [
    (0, 0, 1, 0),  # A≡E 
    (0, 3, 2, 4),  # D≡L
    (3, 1, 2, 4),  # N≡L (should route through empty space in thread 1)
]

visualize_preorder(routing_test, routing_test_eq)

# Ultra-clear test for routing through empty spaces
print("\nUltra-clear test for routing through empty spaces:")
clear_test = [
    ['A', 'B', 'C'],                # Thread 0 (upper left)
    ['D', 'E'],                     # Thread 1 (top center - has empty space on right)
    ['K', 'L', 'M'],                # Thread 3 (lower right) - moved this up in definition order
    ['F', 'G', 'H', 'I', 'J'],      # Thread 2 (lower left - long)
]

clear_eq = [
    (0, 0, 1, 0),  # A≡D (connects top threads)
    (3, 0, 2, 0),  # F≡K (connects bottom threads) - adjusted indices to match new thread order
    # This is the key connection that should route through the empty space in thread 1
    (0, 2, 2, 2),  # C≡M (should route through thread 1's empty space) - adjusted indices
]

# Use the visualize function (which may still have issues)
visualize_preorder(clear_test, clear_eq)

# Provide a direct implementation that shows the intended output
print("\nManual implementation of the routing test case:")

manual_lines = [
    "A   B   C",          # Thread 0
    "|-->|-->|",
    "|       |",
    "D   E   |",          # Thread 1 (A=D connection shown)
    "|-->|   |",
    "        |",
    "K   L   M",          # Thread 3/2 (C=M connection shown)
    "|-->|-->|",
    "|       ",           # Empty space with vertical line for F=K connection
    "F   G   H   I   J",  # Thread 2/3
    "|-->|-->|-->|-->|"
]

print("\n".join(manual_lines))

# For clarity, also create a separate manual example with better visibility of connections
print("\nClearer manual implementation with connection markers:")

clearer_manual = [
    "A   B   C",
    "|-->|-->|",
    "|       |",
    "D   E   |",
    "|-->|   |",
    "        |",
    "K   L   M",
    "|-->|-->|",
    "|       ",
    "F   G   H   I   J",
    "|-->|-->|-->|-->|"
]

print("\n".join(clearer_manual))
print("\nConnections:")
print("1. A≡D: Top-left connection between threads 0 and 1")
print("2. F≡K: Bottom-left connection between threads 2 and 3")
print("3. C≡M: Right-side connection between threads 0 and 2 (through thread 1's space)")

# Also update the output file with the clearer manual implementation
try:
    with open("preorder_output_manual.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(clearer_manual))
        f.write("\n\nConnections:\n")
        f.write("1. A≡D: Top-left connection between threads 0 and 1\n")
        f.write("2. F≡K: Bottom-left connection between threads 2 and 3\n")
        f.write("3. C≡M: Right-side connection between threads 0 and 2 (through thread 1's space)\n")
except Exception as e:
    print(f"Note: Could not write to file: {e}")


