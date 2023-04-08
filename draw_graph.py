from graphviz import Digraph
from graph import build_graph
from value import Value


def visualize_graph(root: Value):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    edges, nodes = build_graph(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular node for it
        dot.node(name = uid, label = "{%s | %.8f | grad %.4f}" % (n.label, n.val, n.grad), shape='record')
        if n.operator:
            # if this value is a result of an operation, add a new node for the operation
            dot.node(name = uid + n.operator, label = n.operator)
            # connect operation to the node
            dot.edge(uid + n.operator, uid)
    for n1, n2 in edges:
        # connect operation to the node
        uid_n1, uid_n2 = str(id(n1)), str(id(n2))
        dot.edge(uid_n1, uid_n2 + n2.operator)
    return dot
