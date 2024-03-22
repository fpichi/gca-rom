import numpy as np
import matplotlib.pyplot as plt
from gca_rom import scaling
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
import matplotlib.colors as mcolors
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
    plt.xlabel('Epochs')
    plt.title('Loss over training epochs')
    plt.legend(['Autoencoder (train)', 'Map (train)', 'Autoencoder (test)', 'Map (test)'])
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
    plt.title('Evolution in the latent space')
    plt.ylabel('$u_N(\mu)$')
    plt.xlabel('Snaphots')
    plt.legend(['Autoencoder', 'Map'])
    plt.savefig(HyperParams.net_dir+'latents'+HyperParams.net_run+'.png', bbox_inches='tight', dpi=500)
    green_diamond = dict(markerfacecolor='g', marker='D')
    _, ax = plt.subplots()
    ax.boxplot(latents_estimation.detach().numpy(), flierprops=green_diamond)
    plt.title('Variance in the latent space')
    plt.ylabel('$u_N(\mu)$')
    plt.xlabel('Bottleneck')
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
    ax.plot_surface(X1, X2, output, cmap=colormaps['coolwarm'], color='blue')
    ax.contour(X1, X2, output, zdir='z', offset=output.min(), cmap=colormaps['coolwarm'])
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$',
           zlabel='$\\epsilon_{GCA}(\\mathbf{\mu})$')
    ax.plot(tr_pt_1, tr_pt_2, output.min()*np.ones(len(tr_pt_1)), '*r')
    ax.set_title('Relative Error '+vars)
    ax.zaxis.offsetText.set_visible(False)
    exponent_axis = np.floor(np.log10(max(ax.get_zticks()))).astype(int)
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
    ax.text2D(0.9, 0.82, "$\\times 10^{"+str(exponent_axis)+"}$", transform=ax.transAxes, fontsize="x-large")
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_error_2d(res, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=-1):
    """
    This function plots the relative error between the predicted and actual results in 2D

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
    fig = plt.figure('Relative Error 2D '+vars)
    ax = fig.add_subplot()
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    colors = output.flatten()
    sc = plt.scatter(X1.flatten(), X2.flatten(), s=(2e1*colors/output.max())**2, c=colors, cmap=colormaps['coolwarm'])
    plt.colorbar(sc, format=fmt)
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$')
    ax.plot(tr_pt_1, tr_pt_2, '*r')
    ax.set_title('Relative Error 2D '+vars)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_2d_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_fields(SNAP, results, scaler_all, HyperParams, dataset, xyz, params, comp="_U"):
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
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z_net, 100, cmap=colormaps['jet'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Solution field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'field_solution_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_error_fields(SNAP, results, VAR_all, scaler_all, HyperParams, dataset, xyz, params, comp="_U"):
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
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    if dataset.dim == 2:
        triang = np.asarray(dataset.T - 1)
        gs1 = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs1[0, 0])
        cs = ax.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, error, 100, cmap=colormaps['coolwarm'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(cs, cax=cax, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        ax = fig.add_subplot(projection='3d')
        cax = inset_axes(ax, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.15, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        p = ax.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=error, cmap=colormaps['coolwarm'], linewidth=0.5)
        cbar = fig.colorbar(p, ax=ax, cax=cax, format=fmt)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.locator_params(axis='both', nbins=5)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()
    plt.tight_layout()
    ax.set_aspect('equal', 'box')
    ax.set_title('Error field for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)))
    plt.savefig(HyperParams.net_dir+'error_field_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_latent_time(HyperParams, SAMPLE, latents, mu_space, params, param_sample):
    """
    This function plots the evolution of latent states over time and saves the plot as a .png file.

    Parameters:
    SNAP (int): The snapshot number.
    latents (np.ndarray): The latent states.
    params (list): The parameters.
    HyperParams (object): The hyperparameters.

    Returns:
    None
    """

    plt.figure()
    sequence_length = latents.shape[0] // param_sample
    start = SAMPLE * sequence_length
    end = start + sequence_length
    time = mu_space[-1]

    for i in range(HyperParams.bottleneck_dim):
        stn_evolution = latents[start:end, i]
        plt.plot(time, stn_evolution)

    plt.xlabel('$t$')
    plt.ylabel('$s(t)$')
    plt.title('Latent state evolution $\mu = $'+ str(np.around(params[start][0:2].detach().numpy(), 2)))
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(HyperParams.net_dir+'latent_evolution_'+HyperParams.net_run+str(SAMPLE)+'.png', bbox_inches='tight', dpi=500)


def plot_sample(HyperParams, mu_space, params, train_trajectories, test_trajectories, p1=0, p2=1, param_frequency=False):
    """
    This function plots the train/test sample used for the training
    Parameters:
    HyperParams (object): The HyperParams object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    train_trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """

    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    n_params = params.shape[1]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    te_pt_1 = params[test_trajectories, p1]
    te_pt_2 = params[test_trajectories, p2]

    fig = plt.figure('Sample')
    ax = fig.add_subplot()

    if param_frequency is True:
        for i in range(len(params[train_trajectories][0])):
            plot_idx=[]
            plot_val=[]
            vals, counts = np.unique(params[train_trajectories][:, i], return_counts=True)
            args = vals.argsort()
            vals = vals[args]
            counts = counts[args]
            for j in range(len(vals)):
                mu = vals[j]
                val = counts[j]
                plot_idx.append(f'$\mu_{i}={np.around(mu, 2)}$')
                plot_val.append(val)
            plt.bar(plot_idx, plot_val)
        plt.xticks(rotation=90)
        plt.xlabel('Parameter')
        plt.ylabel('Frequency in Training Set')
    else:
        if n_params > 2:
            rows, ind = np.unique(params[:, [p1, p2]], axis=0, return_inverse=True)
            indices_dict = defaultdict(list)
            [indices_dict[tuple(rows[i])].append(idx) for idx, i in enumerate(ind)]
            tr_pt = [i for i in indices_dict if any(idx in train_trajectories for idx in indices_dict[i])]
            te_pt = [i for i in indices_dict if any(idx in test_trajectories for idx in indices_dict[i])]
            tr_pt_1 = [t[0] for t in tr_pt]
            tr_pt_2 = [t[1] for t in tr_pt]
            te_pt_1 = [s[0] for s in te_pt]
            te_pt_2 = [s[1] for s in te_pt]
        ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
            ylim=tuple([mu2_range[0], mu2_range[-1]]),
            xlabel=f'$\mu_{str((p1%n_params)+1)}$',
            ylabel=f'$\mu_{str((p2%n_params)+1)}$')
        ax.scatter(tr_pt_1, tr_pt_2, marker='o', color="red", label='Training')
        ax.scatter(te_pt_1, te_pt_2, marker='s', color="blue", label='Testing')
        ax.legend()

    ax.set_title('Sample')
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'sample'+HyperParams.net_run+'.png', transparent=True, dpi=500)


def plot_comparison_fields(results, VAR_all, scaler_all, HyperParams, dataset, xyz, params, grid="horizontal", comp="_U", adjust_title=None):
    """
    Plots the field solution for a given snapshot, the ground truth, and the error field.

    The function takes in the following inputs:

    SNAP: integer value indicating the snapshot to be plotted.
    results: array of shape (num_samples, num_features), representing the network's output.
    scaler_all: numpy.ndarray of scaling variables.
    HyperParams: instance of the Autoencoder parameters class containing information about the network architecture and training.
    dataset: array of shape (num_samples, 3), representing the Fenics dataset.
    PARAMS: array of shape (num_snap,), containing the parameters associated with each snapshot.
    TIMES: array of shape (num_snap,), containing the time associated with each snapshot.
    
    The function generates a plot of the field solution and saves it to disk using the filepath specified in HyperParams.net_dir.
    """

    plt.figure()
    Z = scaling.inverse_scaling(VAR_all, scaler_all, HyperParams.scaling_type)
    Z_net = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
    error = np.linalg.norm(Z_net - Z, axis=0) / np.linalg.norm(Z, axis=0)
    SNAP = np.argmax(error)
    z = Z[:, SNAP]
    z_net = Z_net[:, SNAP]
    xx = xyz[0]
    yy = xyz[1]

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    triang = np.asarray(dataset.T - 1)
    error_abs = abs(z - z_net)
    error_rel = error_abs/np.linalg.norm(z, 2)

    if dataset.dim == 2:
        if grid == "horizontal":
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            y0=0.7
        elif grid == "vertical":
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            y0=1.1
    elif dataset.dim == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, subplot_kw=dict(projection='3d'))
        y0=1.1
    if adjust_title is not None:
        y0 = adjust_title

    # Subplot 1
    if dataset.dim == 2:
        norm1 = mcolors.Normalize(vmin=z.min(), vmax=z.max())
        cs1 = ax1.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z, 100, cmap=colormaps['jet'], norm=norm1)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = plt.colorbar(cs1, cax=cax1, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        cax1 = inset_axes(ax1, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
        p1 = ax1.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=z, cmap=colormaps['jet'], linewidth=0.5)
        cbar1 = fig.colorbar(p1, ax=ax1, cax=cax1, format=fmt)
        # ax1.set_xlabel('$x$')
        # ax1.set_ylabel('$y$')
        # ax1.set_zlabel('$z$')
        ax1.locator_params(axis='x', nbins=2)
        ax1.yaxis.set_ticklabels([])
        ax1.zaxis.set_ticklabels([])        
    tick_locator = MaxNLocator(nbins=3)
    cbar1.locator = tick_locator
    cbar1.ax.yaxis.set_offset_position('left')
    cbar1.update_ticks()
    ax1.set_aspect('equal', 'box')
    ax1.set_title('Truth')

    # Subplot 2
    if dataset.dim == 2:
        norm2 = mcolors.Normalize(vmin=z_net.min(), vmax=z_net.max())
        cs2 = ax2.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, z_net, 100, cmap=colormaps['jet'], norm=norm2)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(cs2, cax=cax2, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        cax2 = inset_axes(ax2, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax2.transAxes, borderpad=0)
        p2 = ax2.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=z_net, cmap=colormaps['jet'], linewidth=0.5)
        cbar2 = fig.colorbar(p2, ax=ax2, cax=cax2, format=fmt)
        # ax2.set_xlabel('$x$')
        # ax2.set_ylabel('$y$')
        # ax2.set_zlabel('$z$')
        ax2.locator_params(axis='x', nbins=2)
        ax2.yaxis.set_ticklabels([])
        ax2.zaxis.set_ticklabels([])
    tick_locator = MaxNLocator(nbins=3)
    cbar2.locator = tick_locator
    cbar2.ax.yaxis.set_offset_position('left')
    cbar2.update_ticks()
    ax2.set_aspect('equal', 'box')
    ax2.set_title('Prediction')

    # Subplot 3
    if dataset.dim == 2:
        norm3 = mcolors.Normalize(vmin=error_rel.min(), vmax=error_rel.max())
        cs3 = ax3.tricontourf(xx[:, SNAP], yy[:, SNAP], triang, error_rel, 100, cmap=colormaps['coolwarm'], norm=norm3)
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cbar3 = plt.colorbar(cs3, cax=cax3, format=fmt)
    elif dataset.dim == 3:
        zz = xyz[2]
        cax3 = inset_axes(ax3, width="5%", height="60%", loc="center left", 
                         bbox_to_anchor=(1.5, 0., 1, 1), bbox_transform=ax3.transAxes, borderpad=0)
        p3 = ax3.scatter(xx[:, SNAP], yy[:, SNAP], zz[:, SNAP], c=error_rel, cmap=colormaps['coolwarm'], linewidth=0.5)
        cbar3 = fig.colorbar(p3, ax=ax3, cax=cax3, format=fmt)
        # ax3.set_xlabel('$x$')
        # ax3.set_ylabel('$y$')
        # ax3.set_zlabel('$z$')
        ax3.locator_params(axis='x', nbins=2)
        ax3.yaxis.set_ticklabels([])
        ax3.zaxis.set_ticklabels([])
    tick_locator = MaxNLocator(nbins=3)
    cbar3.locator = tick_locator
    cbar3.ax.yaxis.set_offset_position('left')
    cbar3.update_ticks()
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Error')

    # Adjust layout
    plt.tight_layout()
    fig.suptitle('Maximum error for $\mu$ = '+str(np.around(params[SNAP].detach().numpy(), 2)), y=y0)
    plt.savefig(HyperParams.net_dir+'comparison_field_'+str(SNAP)+''+HyperParams.net_run+comp+'.png', bbox_inches='tight', dpi=500)


def plot_error_3d(reconstruction, solution, scaler, HyperParams, mu_space, params, train_trajectories, vars, test_trajectories=None):
    """
    This function plots the relative error between the predicted and actual results in 3D

    Parameters:
    reconstruction (ndarray): The predicted results
    VAR_all (ndarray): The actual results
    scaler (object): The scaler object used for scaling the results
    HyperParams (object): The HyperParams object holding the necessary hyperparameters
    mu1_range (ndarray): Range of the first input variable
    mu2_range (ndarray): Range of the second input variable
    params (ndarray): The input variables
    trajectories (ndarray): The indices of the training data
    vars (str): The name of the variable being plotted
    """

    u_hf = scaling.inverse_scaling(solution, scaler, HyperParams.scaling_type)
    u_app = scaling.inverse_scaling(reconstruction, scaler, HyperParams.scaling_type)
    error = np.linalg.norm(u_app - u_hf, axis=0) / np.linalg.norm(u_hf, axis=0)
    p1 = 0
    p2 = 1
    p3 = 2
    mu1_range = mu_space[p1]
    mu2_range = mu_space[p2]
    mu3_range = mu_space[p3]
    n_params = params.shape[0]
    tr_pt_1 = params[train_trajectories, p1]
    tr_pt_2 = params[train_trajectories, p2]
    tr_pt_3 = params[train_trajectories, p3]    
    if test_trajectories:
        pt_1 = params[test_trajectories, p1]
        pt_2 = params[test_trajectories, p2]
        pt_3 = params[test_trajectories, p3]
    else:
        pt_1 = params[:, p1]
        pt_2 = params[:, p2]
        pt_3 = params[:, p3]
    fig = plt.figure('Relative Error 3D '+vars)
    ax = fig.add_subplot(projection='3d')
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    colors = error.flatten()
    sc = ax.scatter(pt_1, pt_2, pt_3, s=1000*colors, c=colors, cmap=colormaps['coolwarm'])
    cbar = plt.colorbar(sc, format=fmt, shrink=0.5, pad=0.1)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()    
    ax.set(xlim=tuple([mu1_range[0], mu1_range[-1]]),
           ylim=tuple([mu2_range[0], mu2_range[-1]]),
           zlim=tuple([mu3_range[0], mu3_range[-1]]),
           xlabel=f'$\mu_{str((p1%n_params)+1)}$',
           ylabel=f'$\mu_{str((p2%n_params)+1)}$',
           zlabel=f'$\mu_{str((p3%n_params)+1)}$')
    ax.scatter(tr_pt_1, tr_pt_2, tr_pt_3, marker="*", color="red", s=10)
    ax.set_title('Relative Error 3D '+vars)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_3d_'+vars+HyperParams.net_run+'.png', transparent=True, dpi=500)
