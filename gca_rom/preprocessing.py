import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gca_rom import scaling


def graphs_dataset(dataset, HyperParams, param_sample=None):
    """
    graphs_dataset: function to process and scale the input dataset for graph autoencoder model.

    Inputs:
    dataset: an object containing the dataset to be processed.
    HyperParams: an object containing the hyperparameters of the graph autoencoder model.

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
    if HyperParams.comp == 1:
        var = dataset.U
    else:
        var1 = dataset.VX
        var2 = dataset.VY
        var = torch.stack((dataset.VX, dataset.VY), dim=2)

    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]

    print("Number of nodes processed: ", num_nodes)
    print("Number of graphs processed: ", num_graphs)
    rate = HyperParams.rate/100
    total_sims = int(num_graphs)

    if param_sample is None:
        train_sims = int(rate * total_sims)
        test_sims = total_sims - train_sims
        main_loop = list(range(total_sims))
        np.random.shuffle(main_loop)

        train_snapshots = main_loop[0:train_sims]
        train_snapshots.sort()
        test_snapshots = main_loop[train_sims:total_sims]
        test_snapshots.sort()
    else:
        train_param_sims = int(rate * param_sample)
        main_loop = list(range(param_sample))
        np.random.shuffle(main_loop)

        train_param_snap = main_loop[0:train_param_sims]
        train_param_snap.sort()
        test_param_snap = main_loop[train_param_sims:param_sample]
        test_param_snap.sort()
        n_time = total_sims//param_sample
        train_snapshots = [i*n_time+j for i in train_param_snap for j in range(n_time)]
        test_snapshots = [i*n_time+j for i in test_param_snap for j in range(n_time)] 
        train_sims = len(train_snapshots)
        test_sims = len(test_snapshots)        


    ## SCALING
    scaling_type = HyperParams.scaling_type
    if HyperParams.comp == 1:
        var_test = dataset.U[:, test_snapshots]
        scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, HyperParams.scaler_number)
        scaler_test, VAR_test = scaling.tensor_scaling(var_test, scaling_type, HyperParams.scaler_number)
    else:
        var1_test = var1[:, test_snapshots]
        var2_test = var2[:, test_snapshots]
        scaler_var1_all, VAR1_all = scaling.tensor_scaling(var1, scaling_type, HyperParams.scaler_number)
        scaler_var1_test, VAR1_test = scaling.tensor_scaling(var1_test, scaling_type, HyperParams.scaler_number)
        scaler_var2_all, VAR2_all = scaling.tensor_scaling(var2, scaling_type, HyperParams.scaler_number)
        scaler_var2_test, VAR2_test = scaling.tensor_scaling(var2_test, scaling_type, HyperParams.scaler_number)
        VAR_all = torch.cat((VAR1_all, VAR2_all), dim=2)
        VAR_test = torch.cat((VAR1_test, VAR2_test), dim=2)
        scaler_all = [scaler_var1_all, scaler_var2_all]
        scaler_test = [scaler_var1_test, scaler_var2_test]


    graphs = []
    edge_index = torch.t(dataset.E) - 1
    for graph in range(num_graphs):
        if dataset.dim == 2:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1)), 1)
        elif dataset.dim == 3:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1), zz[:, graph].unsqueeze(1)), 1)
        ei = torch.index_select(pos, 0, edge_index[0, :])
        ej = torch.index_select(pos, 0, edge_index[1, :])
        edge_attr = torch.abs(ej - ei)
        if dataset.dim == 2:
            edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2)).unsqueeze(1)
        elif dataset.dim == 3:
            edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2) + torch.pow(edge_attr[:, 2], 2)).unsqueeze(1)
        if HyperParams.comp == 1:
            node_features = VAR_all[graph, :]
        else:
            node_features = VAR_all[graph, :, :]
        dataset_graph = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, pos=pos)
        graphs.append(dataset_graph)

    HyperParams.num_nodes = dataset_graph.num_nodes
    train_dataset = [graphs[i] for i in train_snapshots]
    test_dataset = [graphs[i] for i in test_snapshots]

    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))

    max_batch_size = 100

    loader = DataLoader(graphs, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=train_sims if train_sims<max_batch_size else max_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_sims if test_sims<max_batch_size else max_batch_size, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return loader, train_loader, test_loader, \
            val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_test, \
                train_snapshots, test_snapshots


def delete_initial_condition(dataset, params, mu_space, n_comp, n_snap_time):
    params = params[params[:, -1] != 0.]
    mu_space[-1] = np.delete(mu_space[-1], 0)
    if n_comp == 1:
        indices = torch.ones(dataset.U.shape[1], dtype=torch.bool)
        indices[::n_snap_time] = 0
        dataset.U = dataset.U[:, indices]
    elif n_comp == 2:
        indices = torch.ones(dataset.VX.shape[1], dtype=torch.bool)
        indices[::n_snap_time] = 0
        dataset.VX = dataset.VX[:, indices]
        dataset.VY = dataset.VY[:, indices]
    else:
        print("Invalid dimension. Please enter 1 or 2.")
    
    dataset.xx = dataset.xx[:, indices]
    dataset.yy = dataset.yy[:, indices]
    return dataset, params, mu_space


def shrink_dataset(dataset, mu_space, n_sim, n_snap2keep, n_comp):
    time = mu_space[-1]
    n_time = len(time)
    idx_time = np.round(np.linspace(0, n_time-1, n_snap2keep)).astype(int)
    mu_space[-1] = time[idx_time]

    idx = np.copy(idx_time)
    for i in range(1, n_sim):
        idx_time += n_time
        idx = np.hstack((idx, idx_time))

    if n_comp == 1:
        dataset.U = dataset.U[:, idx]
    elif n_comp == 2:
        dataset.VX = dataset.VX[:, idx]
        dataset.VY = dataset.VY[:, idx]
    dataset.xx = dataset.xx[:, idx]
    dataset.yy = dataset.yy[:, idx]

    return dataset, mu_space