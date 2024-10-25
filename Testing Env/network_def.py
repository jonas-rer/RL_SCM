import networkx as nx
import matplotlib.pyplot as plt

# Variables
# I = Inventory
# D = Demand
# L = Lead time

# Initialize the graph
graph = nx.DiGraph()

# Raw material supplier
graph.add_node('S')

# Manufacturer
graph.add_node('A', I=20)
graph.add_node('B', I=30)
graph.add_node('C', I=40)

# Distributor
graph.add_node('D')

# Add edges with lead times
graph.add_edge('S', 'A', L=3)
graph.add_edge('S', 'B', L=3)
graph.add_edge('S', 'C', L=3)
graph.add_edge('A', 'D', L=5)
graph.add_edge('B', 'D', L=5)
graph.add_edge('C', 'D', L=5)

# Draw the graph
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True)

# Add node labels
node_labels = nx.get_node_attributes(graph, 'I')
nx.draw_networkx_labels(graph, pos, labels=node_labels)

# Add edge labels
edge_labels = nx.get_edge_attributes(graph, 'L')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

plt.show()