from modules import readTUds, temporal_graph_from_TUds
import pickle
import torch
import numpy as np
import torch
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
from torch_sparse import coalesce
from torch_scatter import scatter_add
file_path_template = "datasets/{datasetname}/{datasetname}"
def load_data_feature(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('betti_vectors_mp_hks_allt.pkl', 'rb') as g:
        topo_vec = pickle.load(g)
    vec = np.array(topo_vec[data_name])
    vec = torch.tensor(vec)
    X0 = vec.permute(0, 3, 1, 2)
    y0 = np.array(graphs_label)
    return X0,y0
def load_MP_Dos(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('betti_vectors_mp_hks (1).pkl', 'rb') as g:
        MP_hks = pickle.load(g)
    vec = np.array(MP_hks[data_name])
    vec = torch.tensor(vec)
    X0 = vec.permute(0, 3, 1, 2)
    with open('dos_vec_4_1.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    return X0,X1,y0
def load_SW_Dos_Betti(data_name):
    file_path = file_path_template.format(datasetname=data_name)
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(file_path)
    temporal_graphs,temp_edge_idx = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    with open('sw_betti_3_2.pkl', 'rb') as g:
        sw_betti = pickle.load(g)
    X0 = torch.tensor(np.array(sw_betti[data_name]), dtype=torch.float32)
    with open('dos_vec_3_2.pkl', 'rb') as g:
        dos_vec = pickle.load(g)
    X1 = torch.tensor(np.array(dos_vec[data_name]), dtype=torch.float32)
    y0 = np.array(graphs_label)
    data_list = []
    k = 10  # fallback dimensionality if needed

    for i in range(num_graphs):
        src, dst = temp_edge_idx[i]
        edge_index = torch.stack([src, dst], dim=0)
        num_nodes = edge_index.max().item() + 1

        # Get Laplacian as COO
        edge_index_lap, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)

        # Reconstruct sparse Laplacian matrix
        laplacian = torch.sparse_coo_tensor(
            edge_index_lap,
            edge_weight,
            size=(num_nodes, num_nodes)
        ).to_dense()  # Convert to dense matrix to extract rows

        # Use each row as node feature
        x = laplacian  # Shape: [num_nodes, num_nodes]

        # Optional: reduce dimension (if large) using PCA or similar
        if x.shape[1] > k:
            # Simple dimensionality reduction using SVD
            U, S, V = torch.linalg.svd(x)
            x = U[:, :k] @ torch.diag(S[:k])  # shape: [num_nodes, k]

        y = torch.tensor(graphs_label[i])
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list,X0,X1,y0


