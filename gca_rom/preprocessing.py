import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gca_rom import scaling


def graphs_dataset(dataset, AE_Params):
    """
    graphs_dataset: function to process and scale the input dataset for graph autoencoder model.

    Inputs:
    dataset: an object containing the dataset to be processed.
    AE_Params: an object containing the hyperparameters of the graph autoencoder model.

    Outputs:
    dataset_graph: an object containing the processed and scaled dataset.
    loader: a DataLoader object of the processed and scaled dataset.
    train_loader: a DataLoader object of the training set.
    test_loader: a DataLoader object of the test set.
    val_loader: a DataLoader object of the validation set.
    scaler_all: a scaler object to scale the entire dataset.
    scaler_test: a scaler object to scale the test set.
    xyz: an list containig array of the x, y and z-coordinate of the nodes.
    var: an array of the node features.
    VAR_all: an array of the scaled node features of the entire dataset.
    VAR_test: an array of the scaled node features of the test set.
    train_snapshots: a list of indices of the training set.
    test_snapshots: a list of indices of the test set.
    """

    xx = dataset.xx
    yy = dataset.yy
    xyz = [xx, yy]
    if dataset.dim == 3:
       zz = dataset.zz
       xyz.append(zz)
    var = dataset.U

    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]

    print("Number of nodes processed: ", num_nodes)
    print("Number of graphs processed: ", num_graphs)
    total_sims = int(num_graphs)
    rate = AE_Params.rate/100
    train_sims = int(rate * total_sims)
    test_sims = total_sims - train_sims
    main_loop = np.arange(total_sims).tolist()
    np.random.shuffle(main_loop)

    train_snapshots = main_loop[0:train_sims]
    train_snapshots.sort()
    test_snapshots = main_loop[train_sims:total_sims]
    test_snapshots.sort()

    ## FEATURE SCALING
    var_test = dataset.U[:, test_snapshots]

    scaling_type = AE_Params.scaling_type
    scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, AE_Params.scaler_number)
    scaler_test, VAR_test = scaling.tensor_scaling(var_test, scaling_type, AE_Params.scaler_number)

    graphs = []
    edge_index = torch.t(dataset.E) - 1
    for graph in range(num_graphs):
        if dataset.dim == 2:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1)), 1)
        elif dataset.dim == 3:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1), zz[:, graph].unsqueeze(1)), 1)
        ei = torch.index_select(pos, 0, edge_index[0, :])
        ej = torch.index_select(pos, 0, edge_index[1, :])
        edge_diff = ej - ei
        if dataset.dim == 2:
            edge_attr = torch.sqrt(torch.pow(edge_diff[:, 0], 2) + torch.pow(edge_diff[:, 1], 2))
        elif dataset.dim == 3:
            edge_attr = torch.sqrt(torch.pow(edge_diff[:, 0], 2) + torch.pow(edge_diff[:, 1], 2) + torch.pow(edge_diff[:, 2], 2))
        node_features = VAR_all[graph, :]
        dataset_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        graphs.append(dataset_graph)

    AE_Params.num_nodes = dataset_graph.num_nodes
    train_dataset = [graphs[i] for i in train_snapshots]
    test_dataset = [graphs[i] for i in test_snapshots]

    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))

    loader = DataLoader(graphs, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=train_sims, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_sims, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return dataset_graph, loader, train_loader, test_loader, \
            val_loader, scaler_all, scaler_test, xyz, var, VAR_all, VAR_test, train_snapshots, test_snapshots
