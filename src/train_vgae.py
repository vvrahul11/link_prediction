import torch
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import plotly.express as px
#from plotly import graph_objs as go
import random
import torch_geometric.transforms as T

from torch_geometric.data import download_url, extract_gz
from torch_geometric.nn import VGAE
from torch_geometric.utils import degree

from utils import initialize_data
from model import VariationalGCNEncoder
from utils import plot_training_stats, plot_roc_curve
from utils import get_ranked_edges

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data():
    url = 'http://snap.stanford.edu/biodata/datasets/10012/files/DG-AssocMiner_miner-disease-gene.tsv.gz'
    extract_gz(download_url(url, 'data/'), 'data/')

    data_path = "data/DG-AssocMiner_miner-disease-gene.tsv"
    df = pd.read_csv(data_path, sep="\t")
    print(df.head(), '\n')
    return df, data_path


def vgae_train(train_data, vgae_model, optimizer):
    optimizer.zero_grad()
    vgae_model.train()
    z = vgae_model.encode(train_data.x, train_data.edge_index)
    loss = (
        vgae_model.recon_loss(z, train_data.pos_edge_label_index.to(device)) +
        (1 / train_data.num_nodes) * vgae_model.kl_loss()
    )
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def vgae_test(test_data, vgae_model):
    vgae_model.eval()
    z = vgae_model.encode(test_data.x, test_data.edge_index)
    return vgae_model.test(z, test_data.pos_edge_label_index,
                      test_data.neg_edge_label_index)



if __name__ == '__main__':
    NUM_FEATURES =   20
    HIDDEN_SIZE = 200
    OUT_CHANNELS = 20
    EPOCHS =   40    
    
    losses = []
    test_auc = []
    test_ap = []
    train_aucs = []
    train_aps = []


    df, data_path = get_data()
    data_object, gene_mapping, dz_mapping = initialize_data(data_path)
    reverse_dz_mapping = {j: i for i,j in dz_mapping.items()}
    reverse_gene_mapping = {j: i for i,j in gene_mapping.items()}

    degrees = degree(data_object.edge_index[0]).numpy()
    sorted_degrees_i = np.argsort(-1* degrees)    

    data_object.x = torch.ones((data_object.num_nodes, NUM_FEATURES))
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True,
                        split_labels=True, add_negative_train_samples=True),
    ])
    train_dataset, val_dataset, test_dataset = transform(data_object)
    
    vgae_model = VGAE(VariationalGCNEncoder(NUM_FEATURES, HIDDEN_SIZE, OUT_CHANNELS, dropout=0.5))
    vgae_model = vgae_model.to(device)

    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.01)

    for epoch in range(1, EPOCHS + 1):
        loss = vgae_train(train_dataset, vgae_model, optimizer)
        losses.append(loss)
        auc, ap = vgae_test(test_dataset, vgae_model)
        test_auc.append(auc)
        test_ap.append(ap)

        train_auc, train_ap = vgae_test(train_dataset, vgae_model)

        train_aucs.append(train_auc)
        train_aps.append(train_ap)
        print('Epoch: {:03d}, test AUC: {:.4f}, test AP: {:.4f}, train AUC: {:.4f}, train AP: {:.4f}, loss:{:.4f}'.format(epoch, auc, ap, train_auc, train_ap, loss))

    
    # ---------------
    gene_ids_data_path = "data/gene_ids.tsv"

    gene_ids_df = pd.read_csv(gene_ids_data_path, sep="\t")
    gene_ids_df = gene_ids_df.rename(columns={
        "Gene stable ID": "ENSEMBL Gene ID",
        "Gene stable ID version": "ENSEMBL Gene ID Version",
        "NCBI gene (formerly Entrezgene) ID": "Gene ID"})
    gene_ids_df = gene_ids_df.loc[:, ["ENSEMBL Gene ID", "Gene ID"]]

    # Add new "ENSEMBL Gene ID" column to our existing Gene-Disease Assoc table.
    new_df = pd.merge(df, gene_ids_df, left_on="Gene ID", right_on="Gene ID", how="left")
    new_df["Gene ID"] = new_df["Gene ID"].astype(int)
    print(new_df.loc[:, ["Gene ID", "ENSEMBL Gene ID", ]])  # To confirm mapping.


    # Check how many NCBI Gene IDs were unable to be mapped to ENSEMBL Genes IDs.
    print("Number of unmapped NCBI Genes:",
        new_df.drop_duplicates(subset="Gene ID")['ENSEMBL Gene ID'].isna().sum())
    
    # -----------
    data_path = "data/G-SynMiner_miner-geneHUGO.tsv.gz"
    genes_df = pd.read_csv(data_path, sep="\t")
    # print('\n', genes_df.loc[:,["# ensembl_gene_id", "name"]])

    # Add the appropriate "name" data to the dataframe for each gene.
    full_df = pd.merge(
        new_df, genes_df, left_on="ENSEMBL Gene ID",
        right_on="# ensembl_gene_id", how="left")

    # Cut out any extraneous columns from the data frame and rename for easier use.
    full_df = full_df.loc[:, ['# Disease ID', 'Disease Name', 'Gene ID', 'name']]
    full_df = full_df.rename(columns={
        '# Disease ID': "Disease ID",
        'Disease Name': "Disease Name",
        'Gene ID': "Gene ID",
        'name': "Gene Name",
    })

    print(full_df)

    # ---------------------------
    # Select particular model and dataset
    data_object_to_analyze = data_object    # Choose from: data_object, data_object_with_features
    model_to_analyze = vgae_model            # Choose from: gae_model, vgae_model, selected_model

    print("DataObject:\n", data_object_to_analyze)
    print("\nModel:\n", model_to_analyze)

    ranked_edge_list, ranked_dot_products = get_ranked_edges(data_object_to_analyze, model_to_analyze)

    # Select for particular examples
    select_gene_substrings = ["TNF"]
    select_disease_substrings = ['Prostate cancer']


    if select_disease_substrings:
        dz_regex = "(?i)" + "|".join(select_disease_substrings)
        query_df = full_df[full_df['Disease Name'].str.contains(dz_regex)]
        query_dz_nodes = [dz_mapping[dz_id] for dz_id in query_df['Disease ID'].drop_duplicates().tolist()]
        # print("\nQueried Disease Nodes:\n", query_df['Disease Name'])
    else:
        query_dz_nodes = None

    # Filter data frame by the selected gene terms, and get their node ids.
    if select_gene_substrings:
        gene_regex = "(?i)" + "|".join(select_gene_substrings)
        query_df = full_df[full_df['Gene Name'].notna()]
        query_df = query_df[query_df['Gene Name'].str.contains(gene_regex)]
        query_gene_nodes = [gene_mapping[gene_id] for gene_id in query_df['Gene ID'].drop_duplicates().tolist()]
        # print("\nQueried Gene Nodes:\n", query_df['Gene Name'])
    else:
        query_gene_nodes = None


    # Get reverse dz and gene mappings, to print out all needed info.
    reverse_dz_mapping = {j: i for i,j in dz_mapping.items()}
    reverse_gene_mapping = {j: i for i,j in gene_mapping.items()}

    print("\nTop Predicted Edges")
    top_k = 50
    curr_k = 0
    
    for dz_i, gene_i in ranked_edge_list:
        # Skip edges that do not include the selected dz and gene nodes
        if query_dz_nodes and dz_i not in query_dz_nodes:
            continue
        if query_gene_nodes and gene_i not in query_gene_nodes:
            continue

        # Get all the info needed (dz_i and gene_i are the node indeces)
        dz_id, gene_id = reverse_dz_mapping[dz_i], reverse_gene_mapping[gene_i]
        dz_description = full_df[full_df["Disease ID"] == dz_id]["Disease Name"].drop_duplicates().iloc[0]
        gene_description = full_df[full_df["Gene ID"] == gene_id]["Gene Name"].drop_duplicates().iloc[0]
        dot_product = ranked_dot_products[curr_k]

        print('edge=({},{}), \t dotprod={:.2f},\t descriptions=({},{})'.format(dz_i, gene_i, dot_product, dz_description, gene_description))

        curr_k += 1
        if curr_k > top_k:
            break
