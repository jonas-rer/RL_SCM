import networkx as nx
import matplotlib.pyplot as plt

edge_list = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)]


G = nx.Graph()
G.add_edges_from(edge_list)


nx.draw_spring(G, with_labels=True)