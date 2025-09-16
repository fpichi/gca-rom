import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from gca_rom import scaling
from abc import ABC, abstractmethod
import random


class DataProcessor(ABC):
    """
    Abstract base class for data processing of graph-structured datasets.

    This class defines the general interface and utility methods used to prepare
    datasets for graph-based machine learning tasks. Subclasses should implement
    `compute_indices` to define how train/test splits are created for different
    types of problems (steady, unsteady, extrapolating).
    """

    @abstractmethod
    def compute_indices(self, num_graphs, rate, param_sample=None, extrapolate=False):
        """
        Compute train/test indices based on the problem type.

        Args:
            num_graphs (int): Total number of graphs (snapshots).
            rate (float): Fraction of data to be used for training.
            param_sample (int, optional): Number of different simulations (used for time-dependent problems).
            extrapolate (bool, optional): Whether to use extrapolation for time-dependent problems.

        Returns:
            tuple:
                train_snapshots (list[int]): Indices of training graphs.
                test_snapshots (list[int]): Indices of testing graphs.
                train_sims (int): Number of training snapshots.
                test_sims (int): Number of testing snapshots.
        """
        pass

    def prepare_var(self, dataset, HyperParams):
        """
        Prepare spatial coordinates and variables from dataset.

        Args:
            dataset: Object containing data attributes (xx, yy, zz, U, VX, VY, etc.).
            HyperParams: Object containing hyperparameters.

        Returns:
            tuple:
                xx, yy, zz: Node coordinate tensors (zz is None for 2D).
                xyz (list): List of coordinate tensors.
                var (torch.Tensor): Main variable tensor (stacked if comp > 1).
                var1, var2 (torch.Tensor or None): Component-wise variables if comp > 1.
                num_graphs (int): Number of graphs/snapshots.
                rate (float): Training data fraction in [0, 1].
        """
        xx = dataset.xx
        yy = dataset.yy
        xyz = [xx, yy]
        var1 = None
        var2 = None
        if dataset.dim == 3:
            zz = dataset.zz
            xyz.append(zz)
        else:
            zz = None
        if HyperParams.comp == 1:
            var = dataset.U
        else:
            var1 = dataset.VX
            var2 = dataset.VY
            var = torch.stack((dataset.VX, dataset.VY), dim=2)

        num_nodes = var.shape[0]
        num_graphs = var.shape[1]

        print("Number of nodes processed: ", num_nodes)
        print("Number of graphs processed: ", num_graphs)
        rate = HyperParams.rate / 100
        return xx, yy, zz, xyz, var, var1, var2, num_graphs, rate
    
    def scale(self, HyperParams, dataset, test_snapshots_indices, var, var1, var2):
        """
        Scale variables using selected normalization/scaling strategy.

        Args:
            HyperParams: Object containing scaling parameters (scaling_type, scaler_number, comp).
            dataset: Dataset containing ground truth variables.
            test_snapshots_indices (list[int]): Indices of test snapshots.
            var, var1, var2 (torch.Tensor): Variables to scale.

        Returns:
            tuple:
                VAR_all (torch.Tensor): Scaled full dataset variables.
                VAR_test (torch.Tensor): Scaled test dataset variables.
                scaler_all: Fitted scaler for full dataset.
                scaler_test: Fitted scaler for test dataset.
        """
        scaling_type = HyperParams.scaling_type
        if HyperParams.comp == 1:
            var_test = dataset.U[:, test_snapshots_indices]
            scaler_all, VAR_all = scaling.tensor_scaling(var, scaling_type, HyperParams.scaler_number)
            scaler_test, VAR_test = scaling.tensor_scaling(var_test, scaling_type, HyperParams.scaler_number)
        else:
            var1_test = var1[:, test_snapshots_indices]
            var2_test = var2[:, test_snapshots_indices]
            scaler_var1_all, VAR1_all = scaling.tensor_scaling(var1, scaling_type, HyperParams.scaler_number)
            scaler_var1_test, VAR1_test = scaling.tensor_scaling(var1_test, scaling_type, HyperParams.scaler_number)
            scaler_var2_all, VAR2_all = scaling.tensor_scaling(var2, scaling_type, HyperParams.scaler_number)
            scaler_var2_test, VAR2_test = scaling.tensor_scaling(var2_test, scaling_type, HyperParams.scaler_number)
            VAR_all = torch.cat((VAR1_all, VAR2_all), dim=2)
            VAR_test = torch.cat((VAR1_test, VAR2_test), dim=2)
            scaler_all = [scaler_var1_all, scaler_var2_all]
            scaler_test = [scaler_var1_test, scaler_var2_test]
        return VAR_all, VAR_test, scaler_all, scaler_test
    
    def append_graphs(self, HyperParams, VAR_all, dataset, num_graphs, xx, yy, train_snapshots_indices, test_snapshots_indices, zz=None):
        """
        Construct PyTorch Geometric graphs from dataset variables and coordinates.

        Args:
            HyperParams: Object containing model hyperparameters.
            VAR_all (torch.Tensor): Scaled variables for all snapshots.
            dataset: Dataset containing edge connectivity, coordinates, and dimensions.
            num_graphs (int): Total number of graphs.
            xx, yy, zz (torch.Tensor): Node coordinates (zz may be None for 2D).
            train_snapshots_indices, test_snapshots_indices (list[int]): Train/test snapshot indices.

        Returns:
            tuple:
                graphs (list[torch_geometric.data.Data]): All graph objects.
                train_dataset (list): Training graph subset.
                test_dataset (list): Testing graph subset.
        """
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
        train_dataset = [graphs[i] for i in train_snapshots_indices]
        test_dataset = [graphs[i] for i in test_snapshots_indices]

        print("Length of train dataset: ", len(train_dataset))
        print("Length of test dataset: ", len(test_dataset))

        return graphs, train_dataset, test_dataset
    
    def return_loaders(self, HyperParams, graphs, train_dataset, test_dataset, train_sims, test_sims):
        """
        Create PyTorch DataLoaders for training, testing, and validation.

        Args:
            HyperParams: Object containing batch size and other parameters.
            graphs (list): All graph objects.
            train_dataset (list): Training graphs.
            test_dataset (list): Testing graphs.
            train_sims (int): Number of training graphs.
            test_sims (int): Number of testing graphs.

        Returns:
            tuple:
                loader (DataLoader): Loader for full dataset.
                train_loader (DataLoader): Loader for training data.
                test_loader (DataLoader): Loader for test data.
                val_loader (DataLoader): Loader for validation (test) data.
        """
        loader = DataLoader(graphs, batch_size=1)
        train_loader = DataLoader(train_dataset, batch_size=train_sims if train_sims<HyperParams.batch_size else HyperParams.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=test_sims if test_sims<HyperParams.batch_size else HyperParams.batch_size, shuffle=False)
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return loader, train_loader, test_loader, val_loader

    def graphs_dataset(self, dataset, HyperParams, param_sample=None, extrapolate=False):
        """
        Prepare dataset and construct train/test splits with loaders.

        Args:
            dataset: Dataset object containing nodes, edges, and variables.
            HyperParams: Object with training and scaling hyperparameters.
            param_sample (int, optional): Number of parameter samples (time-dependent problems).
            extrapolate (bool, optional): Whether to use extrapolation in time.

        Returns:
            tuple:
                loader, train_loader, test_loader, val_loader: DataLoaders for training/evaluation.
                scaler_all, scaler_test: Scalers for normalization.
                xyz (list): Node coordinates.
                VAR_all, VAR_test: Scaled variables.
                train_snapshots, test_snapshots: Indices used for train/test split.
        """
        xx, yy, zz, xyz, var, var1, var2, num_graphs, rate = self.prepare_var(dataset, HyperParams)

        train_snapshots, test_snapshots, train_sims, test_sims = self.compute_indices(num_graphs, rate, param_sample, extrapolate)

        VAR_all, VAR_test, scaler_all, scaler_test = self.scale(HyperParams, dataset, test_snapshots, var, var1, var2)
        graphs, train_dataset, test_dataset = self.append_graphs(HyperParams, VAR_all, dataset, num_graphs, xx, yy, train_snapshots, test_snapshots, zz)
        loader, train_loader, test_loader, val_loader = self.return_loaders(HyperParams, graphs, train_dataset, test_dataset, train_sims, test_sims)
        
        return loader, train_loader, test_loader, \
                val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_test, \
                    train_snapshots, test_snapshots

    
class SteadyDataProcessor(DataProcessor):
    """ Data processor for steady-state problems. """

    def compute_indices(self, num_graphs, rate, param_sample=None, extrapolate=False):
        """
        Randomly split steady-state snapshots into train/test sets.

        Args:
            num_graphs (int): Total number of graphs.
            rate (float): Fraction of graphs used for training (0-1).

        Returns:
            tuple:
                train_snapshots, test_snapshots, train_sims, test_sims
        """
        total_sims = int(num_graphs)
        train_sims = int(rate * total_sims)
        test_sims = total_sims - train_sims
        main_loop = list(range(total_sims))
        np.random.shuffle(main_loop)

        train_snapshots = main_loop[0:train_sims]
        train_snapshots.sort()
        test_snapshots = main_loop[train_sims:total_sims]
        test_snapshots.sort()

        return train_snapshots, test_snapshots, train_sims, test_sims


class UnsteadyDataProcessor(DataProcessor):
    """ Data processor for time-dependent problems without temporal extrapolation. """

    def compute_indices(self, num_graphs, rate, param_sample=None, extrapolate=False):
        """
        Select a random subset of parameter trajectories for training,
        including all time steps for selected parameters.

        Args:
            num_graphs (int): Total number of graphs.
            rate (float): Fraction of parameter samples used for training (0-1).
            param_sample (int): Total number of parameter samples.

        Returns:
            tuple:
                train_snapshots, test_snapshots, train_sims, test_sims
        """
        total_sims = int(num_graphs)
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

        return train_snapshots, test_snapshots, train_sims, test_sims


class ExtrapolatingDataProcessor(DataProcessor):
    """
    Data processor for time-dependent problems with temporal extrapolation.

    Instead of using full trajectories, selects a fraction of trajectories
    and truncates their time horizon, forcing the model to extrapolate in time.
    """

    def compute_indices(self, num_graphs, rate, param_sample=None, extrapolate=False):
        """
        Select parameter samples and partial time steps for training (extrapolation setup).

        Args:
            num_graphs (int): Total number of graphs.
            rate (float): Fraction of parameter samples and time steps used.
            param_sample (int): Total number of parameter samples.

        Returns:
            tuple:
                train_snapshots, test_snapshots, train_sims, test_sims
        """
        time_rate = rate

        # Extract number of params and time steps
        total_sims = int(num_graphs)
        n_times = total_sims // param_sample

        n_train_params = int(param_sample * rate)
        train_param_indices = sorted(random.sample(range(param_sample), n_train_params))
        test_param_indices = sorted(set(range(param_sample)) - set(train_param_indices))

        time_cutoff = int(n_times * time_rate)

        # Build snapshot indices
        train_snapshots = [p * n_times + t for p in train_param_indices for t in range(time_cutoff)]
        test_snapshots = [p * n_times + t for p in test_param_indices for t in range(n_times)]

        train_sims = len(train_snapshots)
        test_sims = len(test_snapshots)

        return train_snapshots, test_snapshots, train_sims, test_sims


def graphs_dataset(dataset, HyperParams, param_sample=None, extrapolate=False):
    """
    High-level utility to process dataset into graph objects and loaders.

    Automatically selects the appropriate `DataProcessor` subclass based on
    whether the problem is steady, unsteady, or requires extrapolation.

    Args:
        dataset: Dataset object containing nodes, edges, and variables.
        HyperParams: Object with hyperparameters (rate, comp, batch_size, etc.).
        param_sample (int, optional): Number of parameter samples (for time-dependent problems).
        extrapolate (bool, optional): Whether to perform temporal extrapolation.

    Returns:
        tuple:
            loader (DataLoader): Loader for full dataset.
            train_loader (DataLoader): Loader for training data.
            test_loader (DataLoader): Loader for test data.
            val_loader (DataLoader): Validation loader (test set).
            scaler_all, scaler_test: Scalers for training and testing.
            xyz (list): List of coordinate tensors.
            VAR_all, VAR_test: Scaled variables.
            train_snapshots, test_snapshots: Indices used for splitting dataset.
    """
    is_steady = param_sample is None
    if is_steady:
        data_processor = SteadyDataProcessor() # for steady problems
    elif not extrapolate: # time-dependent problems, but without extrapolation in time
        data_processor = UnsteadyDataProcessor()
    elif extrapolate: # time-dependent problems, with extrapolation in time
        data_processor = ExtrapolatingDataProcessor()
    else:
        raise ValueError('Invalid arguments passed to graphs_dataset. Please select valid arguments.')
    return data_processor.graphs_dataset(dataset, HyperParams, param_sample, extrapolate)


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