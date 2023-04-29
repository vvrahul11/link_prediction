import torch
import numpy as np
import pandas as pd
import random
import torch_geometric.transforms as T

#import matplotlib.pyplot as plt
#import plotly.express as px
#from plotly import graph_objs as go
from torch_geometric.data import download_url, extract_gz
from torch_geometric.nn import GAE
from torch_geometric.utils import degree
from utils import initialize_data
from model import GCNEncoder
#from utils import plot_training_stats, plot_roc_curve

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


def gae_train(train_data, gae_model, optimizer):
    gae_model.train()
    optimizer.zero_grad()
    z = gae_model.encode(train_data.x, train_data.edge_index)
    loss = gae_model.recon_loss(z, train_data.pos_edge_label_index.to(device))
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)

@torch.no_grad()
def gae_test(test_data, gae_model):
    gae_model.eval()
    z = gae_model.encode(test_data.x, test_data.edge_index)
    return gae_model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)



if __name__ == '__main__':
    NUM_FEATURES =   20
    HIDDEN_SIZE = 200
    OUT_CHANNELS = 25
    EPOCHS =   100
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
    
    gae_model = GAE(GCNEncoder(NUM_FEATURES, HIDDEN_SIZE, OUT_CHANNELS, 0.5))
    gae_model = gae_model.to(device)
    optimizer = torch.optim.Adam(gae_model.parameters(), lr=0.01)

    for epoch in range(1, EPOCHS + 1):
        loss = gae_train(train_dataset, gae_model, optimizer)
        losses.append(loss)
        auc, ap = gae_test(test_dataset, gae_model)
        test_auc.append(auc)
        test_ap.append(ap)

        train_auc, train_ap = gae_test(train_dataset, gae_model)

        train_aucs.append(train_auc)
        train_aps.append(train_ap)        
        print('Epoch: {:03d}, test AUC: {:.4f}, test AP: {:.4f}, train AUC: {:.4f}, train AP: {:.4f}, loss:{:.4f}'.format(epoch, auc, ap, train_auc, train_ap, loss))

    #plot_training_stats('GAE', losses, test_auc, test_ap, train_aucs, train_aps)
    #plot_roc_curve(gae_model, test_dataset)

    
