from preprocessing import make_stellar_graph
from createGraph import create_undirected_graph
from trainGCN import trainGCN
from testGCN import testGCN

import pandas as pd

def main():
    #TODO implement calibrated prediction
    dataset = pd.read_csv('dataset.csv')
    protein_to_fasta = pd.read_csv('proteins_to_fasta.csv', index_col=0)
    stellar_df = make_stellar_graph(proteins_to_fasta) #dataset = prota, protb columns
    graph = create_undirected_graph(dataset, stellar_df)
    model = trainGCN(graph)
    testGCN(model, test_flow)

if __name__ == "__main__":
    main()