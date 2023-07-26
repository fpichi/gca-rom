import torch
from tqdm import tqdm
import numpy as np


def evaluate(VAR, model, loader, params, HyperParams, test):
    """
    This function evaluates the performance of a trained Autoencoder (AE) model.
    It encodes the input data using both the model's encoder and a mapping function,
    and decodes the resulting latent representations to obtain predicted solutions.
    The relative error between the two latent representations is also computed.

    Inputs:
    VAR: np.array, ground truth solution
    model: object, trained AE model
    loader: object, data loader for the input data
    params: np.array, model parameters
    HyperParams: class, model architecture and training parameters

    Returns:
    results: np.array, predicted solutions
    latents_map: np.array, latent representations obtained using the mapping function
    latents_gca: np.array, latent representations obtained using the AE encoder
    """

    results = torch.zeros(VAR.shape[0], VAR.shape[1], 1)
    latents_map = torch.zeros(VAR.shape[0], HyperParams.bottleneck_dim)
    latents_gca = torch.zeros(VAR.shape[0], HyperParams.bottleneck_dim)
    index = 0
    latents_error = list()
    with torch.no_grad():
        for data in tqdm(loader):
            z_net = model.solo_encoder(data)
            z_map = model.mapping(params[test[index], :])
            latents_map[index, :] = z_map
            latents_gca[index, :] = z_net
            lat_err = np.linalg.norm(z_net - z_map)/np.linalg.norm(z_net)
            latents_error.append(lat_err)
            results[index, :, :] = model.solo_decoder(z_map, data)
            index += 1
        np.savetxt(HyperParams.net_dir+'latents'+HyperParams.net_run+'.csv', latents_map.detach(), delimiter =',')
        latents_error = np.array(latents_error)
        # print("\nMaximum relative error for latent  = ", max(latents_error))
        # print("Mean relative error for latent = ", sum(latents_error)/len(latents_error))
        # print("Minimum relative error for latent = ", min(latents_error))
    return results, latents_map, latents_gca