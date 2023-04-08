# Build an expression graph from a root Value node. Use BFS algorithm to enumerate all the nodes and edges.
def build_graph(root):
    q = [root]
    edges, nodes = set(), set()
    while len(q) > 0:
        curr = q.pop()
        nodes.add(curr)
        for parent in curr.parents:
            if parent not in nodes:
                q.append(parent)
            edges.add((parent, curr))
    return edges, nodes