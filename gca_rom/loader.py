import sys
from torch_geometric.data import Dataset
import torch
import scipy


class LoadDataset(Dataset):
    """
    A custom dataset class which loads data from a .mat file using scipy.io.loadmat.

    data_mat : scipy.io.loadmat
        The loaded data in a scipy.io.loadmat object.
    U : torch.Tensor
        The tensor representation of the specified variable from the data_mat.
    xx : torch.Tensor
        The tensor representation of the 'xx' key from the data_mat. Refers to X coordinates of the domain
    yy : torch.Tensor
        The tensor representation of the 'yy' key from the data_mat.Refers to Y coordinates of the domain
    zz : torch.Tensor
        The tensor representation of the 'zz' key from the data_mat.Refers to Z coordinates of the domain
    dim : Integer
        The integer dim denotes the dimensionality of the domain where the pde is posed
    T : torch.Tensor
        The tensor representation of the 'T' key from the data_mat, casted to int. Adjacency Matrix
    E : torch.Tensor
        The tensor representation of the 'E' key from the data_mat, casted to int. Connection Matrix

    __init__(self, root_dir, variable)
        Initializes the LoadDataset object by loading the data from the .mat file at the root_dir location and converting the specified variable to a tensor representation.
    """

    def __init__(self, root_dir, variable):
        self.data_mat = scipy.io.loadmat(root_dir)
        self.U = torch.tensor(self.data_mat[variable])
        self.xx = torch.tensor(self.data_mat['xx'])
        self.yy = torch.tensor(self.data_mat['yy'])
        self.dim = 3
        try:
            self.zz = torch.tensor(self.data_mat['zz'])
        except:
            self.dim = 2
            KeyError
        self.T = torch.tensor(self.data_mat['T'].astype(int))
        self.E = torch.tensor(self.data_mat['E'].astype(int))
