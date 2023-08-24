import numpy as np
import matplotlib.pyplot as plt
from gca_rom import scaling
from collections import defaultdict
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def plot_loss(HyperParams):
    """
    Plots the history of losses during the training of the autoencoder.

    Attributes:
    HyperParams (namedtuple): An object containing the parameters of the autoencoder.
    """

    history = np.load(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    history_test = np.load(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    ax = plt.figure().gca()
    ax.semilogy(history['l1'])
    ax.semilogy(history['l2'])
    ax.semilogy(history_test['l1'], '--')
    ax.semilogy(history_test['l2'], '--')
    plt.ylabel('Loss')
    plt.ylabel('Epochs')
    plt.title('Loss over training epochs')
    plt.legend(['loss_mse', 'loss_map', 'loss_test_mse', 'loss_test_map'])
    plt.savefig(HyperParams.net_dir+'history_losses'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


def plot_latent(HyperParams, latents, latents_estimation):
    """
    Plot the original and estimated latent spaces
    
    Parameters:
    HyperParams (obj): object containing the Autoencoder parameters 
    latents (tensor): tensor of original latent spaces
    latents_estimation (tensor): tensor of estimated latent spaces
    """

    plt.figure()
    for i1 in range(HyperParams.bottleneck_dim):
        plt.plot(latents[:,i1].detach(), '--')
        plt.plot(latents_estimation[:,i1].detach(),'-')
    plt.savefig(HyperParams.net_dir+'latents'+HyperParams.net_run+'.png', bbox_inches='tight')
    
    green_diamond = dict(markerfacecolor='g', marker='D')
    _, ax = plt.subplots()
    ax.boxplot(latents_estimation.detach().numpy(), flierprops=green_diamond)
    plt.savefig(HyperParams.net_dir+'box_plot_latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    

def plot_error(res, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
    """
    This function plots the relative error between the predicted and actual results.

    Parameters:
    res (ndarray): The predicted results
    VAR_all (ndarray): The actual results
    scaler_all (object): The scaler object used for scaling the results
    HyperParams (object): The HyperParams object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    train_trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """

    u_hf = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(res, scaler_all, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    if n_params > 2:
        rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
        indices_dict = defaultdict(list)
        [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
        error = np.array([np.mean(error[indices]) for indices in indices_dict.values()])
        tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
        tr_pt_1 = [t[0] for t in tr_pt]
        tr_pt_2 = [t[1] for t in tr_pt]
    X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
    output = np.reshape(error, (len(mu1_range), len(mu2_range)))
    fig = plt.figure('Relative Error '+vars)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, output, cmap=cm.coolwarm, color='blue')
    ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=cm.coolwarm)
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$',
           zlabel='$\\epsilon_{GCA}(\\mathbf{\mu})$')
    ax.plot(tr_pt_1, tr_pt_2, output.min()*np.ones(len(tr_pt_1)), '*r')
    plt.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
    ax.set_title('Relative Error '+vars)
    ax.zaxis.offsetText.set_visible(False)
    exponent_axis = np.floor(np.log10(max(ax.get_zticks()))).astype(int)
    ax.text2D(0.9, 0.82, "1e"+str(exponent_axis), transform=ax.transAxes, fontsize="x-large")
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_fields(SNAP, results, scaler_all, HyperParams, dataset, xyz, params):
    """
    Plots the field solution for a given snapshot.

    The function takes in the following inputs:

    SNAP: integer value indicating the snapshot to be plotted.
    results: array of shape (num_samples, num_features), representing the network's output.
    scaler_all: instance of the scaler used to scale the data.
    HyperParams: instance of the Autoencoder parameters class containing information about the network architecture and training.
    dataset: array of shape (num_samples, 3), representing the triangulation of the spatial domain.
    xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
    params: array of shape (num_features,), containing the parameters associated with each snapshot.
    The function generates a plot of the field solution and saves it to disk using the filepath specified in HyperParams.net_dir.
    """

    fig = plt.figure()    
    Z_net = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
    z_net = Z_net[:, SNAP]
    xx = xyz[0]
    yy = xyz[1]
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        cmap = cm.get_cmap(name='jet', lut=None)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z_net, 100, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cmap = cm.get_cmap(name='jet', lut=None) 
        p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=z_net, cmap=cmap, linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'field_solution_'+str(SNAP)+''+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)


def plot_error_fields(SNAP, results, VAR_all, scaler_all, HyperParams, dataset, xyz, params):
    """
    This function plots a contour map of the error field of a given solution of a scalar field.
    The error is computed as the absolute difference between the true solution and the predicted solution,
    normalized by the 2-norm of the true solution.

    Inputs:
    SNAP: int, snapshot of the solution to be plotted
    results: np.array, predicted solution
    VAR_all: np.array, true solution
    scaler_all: np.array, scaling information used in the prediction
    HyperParams: class, model architecture and training parameters
    dataset: np.array, mesh information
    xyz: list of arrays of shape (num_samples, num_features), containing the x, y and z-coordinates of the domain.
    params: np.array, model parameters
    """

    fig = plt.figure()
    Z = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    Z_net = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
    z = Z[:, SNAP]
    z_net = Z_net[:, SNAP]
    error = abs(z - z_net)/np.linalg.norm(z, 2)
    xx = xyz[0]
    yy = xyz[1]
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        cmap = cm.get_cmap(name='jet', lut=None) 
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])   
        cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, error, 100, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cmap = cm.get_cmap(name='jet', lut=None) 
        p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=error, cmap=cmap, linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'error_field_'+str(SNAP)+''+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
