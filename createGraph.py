import networkx as nx
from stellargraph import StellarDiGraph

def create_undirected_graph(dataset, stellar_df):
    #TODO: see if we can directly create Stellar graph without networkx
    DG = nx.Graph()
    DG = nx.from_pandas_edgelist(dataset,edge_attr='weight')
    G = StellarDiGraph(DG, node_features = stellar_df)
    return G