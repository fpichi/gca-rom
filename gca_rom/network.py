import sys
import torch
from torch import nn
from gca_rom import gca, scaling, pde


problem_name, variable, mu1_range, mu2_range = pde.problem(int(sys.argv[1]))


class AE_Params():
    """Class that holds the hyperparameters for the autoencoder model.

    Args:
        sparse_method (str): The method to use for sparsity constraint.
        rate (int): The rate to use for the sparsity constraint.
        seed (int): Seed for the random number generator.
        bottleneck_dim (int): The dimension of the bottleneck layer.
        tolerance (float): The tolerance value for stopping the training.
        lambda_map (float): The weight for the map loss.
        learning_rate (float): The learning rate for the optimizer.
        ffn (int): The number of feed-forward layers.
        in_channels (int): The number of input channels.
        hidden_channels (list): The number of hidden channels for each layer.
        act (function): The activation function to use.
        nodes (int): The number of nodes in each hidden layer.
        skip (int): The number of skipped connections.
        layer_vec (list): The structure of the network.
        net_name (str): The name of the network.
        scaler_name (str): The name of the scaler used for preprocessing.
        weight_decay (float): The weight decay for the optimizer.
        max_epochs (int): The maximum number of epochs to run training for.
        miles (list): The miles for each epoch.
        gamma (float): The gamma value for the optimizer.
        num_nodes (int): The number of nodes in the network.
        scaling_type (int): The type of scaling to use for preprocessing.
        net_dir (str): The directory to save the network in.
        cross_validation (bool): Whether to perform cross-validation.
    """

    def __init__(self):
        self.sparse_method = 'L1_mean'
        self.rate = int(sys.argv[5])
        self.seed = 10
        self.bottleneck_dim = int(sys.argv[8])
        self.tolerance = 1e-6
        self.lambda_map = float(sys.argv[9])
        self.learning_rate = 0.001
        self.ffn = int(sys.argv[6])
        self.in_channels = int(sys.argv[10])
        self.hidden_channels = [1]*self.in_channels
        self.act = torch.tanh
        self.nodes = int(sys.argv[7])
        self.skip = int(sys.argv[4])
        self.layer_vec=[2, self.nodes, self.nodes, self.nodes, self.nodes, self.bottleneck_dim]
        self.net_name = problem_name
        _, self.scaler_name = scaling.scaler_functions(int(sys.argv[3]))
        self.net_run = '_' + self.scaler_name
        self.weight_decay = 0.00001
        self.max_epochs = 5000
        self.miles = []
        self.gamma = 0.0001
        self.num_nodes = 0
        self.scaling_type = int(sys.argv[2])
        self.net_dir = './' + problem_name + '/' + self.net_run+ '/' + variable + '_' + self.net_name + '_lmap' + str(self.lambda_map) + '_btt' + str(self.bottleneck_dim) \
                            + '_seed' + str(self.seed) + '_lv' + str(len(self.layer_vec)-2) + '_hc' + str(len(self.hidden_channels)) + '_nd' + str(self.nodes) \
                            + '_ffn' + str(self.ffn) + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_rate' + str(self.rate) + '/'
        self.cross_validation = True


AE_Params = AE_Params()

class Net(torch.nn.Module):
    """
    Class Net
    ---------

    A PyTorch neural network class which consists of encoder, decoder and mapping modules.

    Attributes
    ----------
    encoder : gca.Encoder
        An encoder module from the gca module.
    decoder : gca.Decoder
        A decoder module from the gca module.
    act_map : AE_Params.act
        The activation map specified in the AE_Params.
    layer_vec : AE_Params.layer_vec
        The layer vector specified in the AE_Params.
    steps : int
        The number of steps for mapping, calculated as the length of the layer_vec minus 1.
    maptovec : nn.ModuleList
        A list of linear modules for mapping.

    Methods
    -------
    solo_encoder(data)
        Encodes the input data using the encoder module.
        Returns the encoded representation.
    solo_decoder(x, data)
        Decodes the encoded representation and the input data using the decoder module.
        Returns the decoded output.
    mapping(x)
        Maps the input using the linear modules in maptovec.
        Returns the mapped output.
    forward(data, parameters)
        Runs a forward pass through the network using the input data and parameters.
        Returns the decoded output, encoded representation, and estimated encoded representation.
    """

    def __init__(self):
        super().__init__()
        self.encoder = gca.Encoder(AE_Params.hidden_channels, AE_Params.bottleneck_dim, AE_Params.num_nodes, ffn=AE_Params.ffn, skip=AE_Params.skip)
        self.decoder = gca.Decoder(AE_Params.hidden_channels, AE_Params.bottleneck_dim, AE_Params.num_nodes, ffn=AE_Params.ffn, skip=AE_Params.skip)

        self.act_map = AE_Params.act
        self.layer_vec = AE_Params.layer_vec
        self.steps = len(self.layer_vec) - 1

        self.maptovec = nn.ModuleList()
        for k in range(self.steps):
            self.maptovec.append(nn.Linear(self.layer_vec[k], self.layer_vec[k+1]))

    def solo_encoder(self,data):
        x = self.encoder(data)
        return x

    def solo_decoder(self,x, data):
        x = self.decoder(x, data)
        return x

    def mapping(self, x):
        idx = 0
        for layer in self.maptovec:
            if(idx==self.steps): x = layer(x)
            else: x =self.act_map(layer(x))
            idx += 1
        return x

    def forward(self, data, parameters):
        z = self.solo_encoder(data)
        z_estimation = self.mapping(parameters)
        x = self.solo_decoder(z, data)
        # x = self.solo_decoder(z_estimation, data)
        return x, z, z_estimation