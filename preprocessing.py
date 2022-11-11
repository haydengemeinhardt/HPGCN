import biovec
import numpy as np
import pandas as pd

def make_stellar_graph(proteins_to_fasta):
    protein_list = make_protein_list(proteins_to_fasta)
    prot_vectors, prot_vectors_keys, missing_prots = make_protein_vectors(proteins_to_fasta, protein_list)
    prot_vectors_2d = make_protein_vectors_2d(prot_vectors)
    stellar_df = pd.DataFrame(prot_vectors_2d, index = prot_vectors_keys)
    return stellar_df

def make_protein_list(proteins_to_fasta):
    protein_list = proteins_to_fasta['Protein'].values.ravel('K')
    return protein_list

def make_protein_vectors(proteins_to_fasta, protein_list):
    pv = biovec.models.load_protvec('swissprot-reviewed-protvec.model')
    prot_vectors = []
    prot_vectors_keys = []
    missing_prots = []
    for prot in protein_list:
        try:
            fasta = proteins_to_fasta.loc[proteins_to_fasta['1'] == prot].values[0,1]
            prot_vectors.append(pv.to_vecs(fasta))
            prot_vectors_keys.append(prot)
        except:
            missing_prots.append(prot)
            print(prot,'not trained in ProtVec model')
    print(len(missing_prots), 'proteins not trained in ProtVec model')
    return prot_vectors, prot_vectors_keys, missing_prots

def make_protein_vectors_2d(prot_vectors):
    prot_vectors_2d = []
    for vec in prot_vectors:
        prot_vectors_2d.append(np.concatenate((vec[0], vec[1], vec[2])))
    return prot_vectors_2d