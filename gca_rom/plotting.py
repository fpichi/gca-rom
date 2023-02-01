import numpy as np
import matplotlib.pyplot as plt
from gca_rom import scaling
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_loss(AE_Params):
    """
    Plots the history of losses during the training of the autoencoder.

    Attributes:
    AE_Params (namedtuple): An object containing the parameters of the autoencoder.
    """

    history = np.load(AE_Params.net_dir+'history'+AE_Params.net_run+'.npy', allow_pickle=True).item()
    history_test = np.load(AE_Params.net_dir+'history_test'+AE_Params.net_run+'.npy', allow_pickle=True).item()
    ax = plt.figure().gca()
    ax.semilogy(history['l1'])
    ax.semilogy(history['l2'])
    ax.semilogy(history_test['l1'], '--')
    ax.semilogy(history_test['l2'], '--')
    plt.ylabel('Loss')
    plt.ylabel('Epochs')
    plt.title('Loss over training epochs')
    plt.legend(['loss_mse', 'loss_map', 'loss_test_mse', 'loss_test_map'])
    plt.savefig(AE_Params.net_dir+'history_losses'+AE_Params.net_run+'.png', bbox_inches='tight')


def plot_latent(AE_Params, latents, latents_estimation):
    """
    Plot the original and estimated latent spaces
    
    Parameters:
    AE_Params (obj): object containing the Autoencoder parameters 
    latents (tensor): tensor of original latent spaces
    latents_estimation (tensor): tensor of estimated latent spaces
    """

    plt.figure()
    for i1 in range(AE_Params.bottleneck_dim):
        plt.plot(latents[:,i1].detach(), '--')
        plt.plot(latents_estimation[:,i1].detach(),'-')
    plt.savefig(AE_Params.net_dir+'latents'+AE_Params.net_run+'.png', bbox_inches='tight')
    
    green_diamond = dict(markerfacecolor='g', marker='D')
    _, ax = plt.subplots()
    ax.boxplot(latents_estimation.detach().numpy(), flierprops=green_diamond)
    plt.savefig(AE_Params.net_dir+'box_plot_latents'+AE_Params.net_run+'.png', bbox_inches='tight')
    

def plot_error(res, VAR_all, scaler_all, AE_Params, mu1_range, mu2_range, params, train_trajectories, vars):
    """
    This function plots the relative error between the predicted and actual results.

    Parameters:
    res (ndarray): The predicted results
    VAR_all (ndarray): The actual results
    scaler_all (object): The scaler object used for scaling the results
    AE_Params (object): The AE_Params object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    train_trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """

    u_hf = scaling.inverse_scaling(VAR_all, scaler_all, AE_Params.scaling_type)
    u_app = scaling.inverse_scaling(res, scaler_all, AE_Params.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    X1, X2 = np.meshgrid(mu1_range, mu2_range, indexing='ij')
    output = np.reshape(error, (len(mu1_range), len(mu2_range)))
    fig = plt.figure('Relative Error '+vars)
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, output, cmap=cm.coolwarm, color='blue')
    ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=cm.coolwarm)
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]), ylim=tuple([mu2_range[0], mu2_range[-1]]), xlabel='$\mu_1$', ylabel='$\mu_2$', zlabel='Relative error for '+vars)
    ax.plot(params[train_trajectories,0], params[train_trajectories,1], output.min()*np.ones(len(params[train_trajectories,1])), '*r')
    ax.set_title('Relative Error '+vars)
    plt.tight_layout()
    plt.savefig(AE_Params.net_dir+'relative_error_'+vars+AE_Params.net_run+'.png', bbox_inches='tight')


def plot_fields(SNAP, results, scaler_all, AE_Params, dataset, xx, yy, params):
    """
    Plots the field solution for a given snapshot.

    The function takes in the following inputs:

    SNAP: integer value indicating the snapshot to be plotted.
    results: array of shape (num_samples, num_features), representing the network's output.
    scaler_all: instance of the scaler used to scale the data.
    AE_Params: instance of the Autoencoder parameters class containing information about the network architecture and training.
    dataset: array of shape (num_samples, 3), representing the triangulation of the spatial domain.
    xx: array of shape (num_samples, num_features), containing the x-coordinates of the domain.
    yy: array of shape (num_samples, num_features), containing the y-coordinates of the domain.
    params: array of shape (num_features,), containing the parameters associated with each snapshot.
    The function generates a plot of the field solution and saves it to disk using the filepath specified in AE_Params.net_dir.
    """

    triang = np.asarray(dataset.T - 1)
    cmap = cm.get_cmap(name='jet', lut=None)
    # fig = plt.figure(figsize=(14, 6))
    gs1 = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs1[0, 0])
    Z_net = scaling.inverse_scaling(results, scaler_all, AE_Params.scaling_type)
    z_net = Z_net[:, SNAP]
    ax.triplot(xx[:,SNAP], yy[:,SNAP], triang, lw=0.5, color='black')
    cs = ax.tricontourf(xx[:,SNAP], yy[:,SNAP], triang, z_net, 100, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cs, cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Solution field for mu = '+str(params[SNAP].detach().numpy()), fontsize=15)
    plt.savefig(AE_Params.net_dir+'field_solution_'+str(SNAP)+''+AE_Params.net_run+'.png', bbox_inches='tight')


def plot_error_fields(SNAP, results, VAR_all, scaler_all, AE_Params, dataset, xx, yy, params):
    """
    This function plots a contour map of the error field of a given solution of a scalar field.
    The error is computed as the absolute difference between the true solution and the predicted solution,
    normalized by the 2-norm of the true solution.

    Inputs:
    SNAP: int, snapshot of the solution to be plotted
    results: np.array, predicted solution
    VAR_all: np.array, true solution
    scaler_all: np.array, scaling information used in the prediction
    AE_Params: class, model architecture and training parameters
    dataset: np.array, mesh information
    xx: np.array, x-coordinate of the mesh
    yy: np.array, y-coordinate of the mesh
    params: np.array, model parameters
    """

    triang = np.asarray(dataset.T - 1)
    cmap = cm.get_cmap(name='jet', lut=None) 
    # fig = plt.figure(figsize=(14, 6))
    gs1 = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs1[0, 0])   
    Z = scaling.inverse_scaling(VAR_all, scaler_all, AE_Params.scaling_type)
    Z_net = scaling.inverse_scaling(results, scaler_all, AE_Params.scaling_type)
    z = Z[:, SNAP]
    z_net = Z_net[:, SNAP]
    error = abs(z - z_net)/np.linalg.norm(z, 2)
    ax.triplot(xx[:,SNAP], yy[:,SNAP], triang, lw=0.5, color='black')
    cs = ax.tricontourf(xx[:,SNAP], yy[:,SNAP], triang, error, 100, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cs, cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Error solution for mu = '+str(params[SNAP].detach().numpy()), fontsize=15)
    plt.savefig(AE_Params.net_dir+'error_field_'+str(SNAP)+''+AE_Params.net_run+'.png', bbox_inches='tight')