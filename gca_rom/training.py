import torch
import torch.nn.functional as F


def train(model, optimizer, device, scheduler, params, train_loader, train_trajectories, HyperParams, history):
    """Trains the autoencoder model.
    
    This function trains the autoencoder model using mean squared error (MSE) loss and a map loss, where the map loss
    is the MSE between the estimated z (latent space) and the actual z. The final loss is the sum of the MSE loss and the
    map loss multiplied by the weight `HyperParams.lambda_map`. The model is trained on the data from `train_loader` and 
    the optimization process is performed using the `optimizer`. The learning rate is updated after every iteration 
    using `scheduler`. Use of mini-batching for reducing the computational cost.

    Args:
        model (torch.nn.Module): The autoencoder model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        device (str): The device to run the model on ('cuda' or 'cpu').
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to update the learning rate.
        params (torch.Tensor): Tensor containing the parameters of the model.
        train_loader (torch.utils.data.DataLoader): The data loader to provide the training data.
        train_trajectories (int): The number of training trajectories.
        HyperParams (dict): The dictionary containing the hyperparameters for the autoencoder model.
        history (dict): The dictionary to store the loss history.
        
    Returns:
        float: The average loss over all training examples.
    """

    model.train()
    total_loss_train = total_examples = sum_loss = 0
    total_loss_train_1 = total_loss_train_2 = 0
    sum_loss_1 = sum_loss_2 = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)  
        out, z, z_estimation = model(data, params[train_trajectories, :])
        loss_train_mse = F.mse_loss(out, data.x, reduction='mean')
        loss_train_map = F.mse_loss(z_estimation, z, reduction='mean')
        loss_train = loss_train_mse + HyperParams.lambda_map * loss_train_map
        loss_train.backward()
        optimizer.step()
        sum_loss += loss_train.item()
        sum_loss_1 += loss_train_mse.item()
        sum_loss_2 += loss_train_map.item()
        total_examples += 1

    scheduler.step()
    total_loss_train = sum_loss / total_examples
    total_loss_train_1 = sum_loss_1 / total_examples
    total_loss_train_2 = sum_loss_2 / total_examples
    history['train'].append(total_loss_train)
    history['l1'].append(total_loss_train_1)
    history['l2'].append(total_loss_train_2)
    return total_loss_train


def val(model, device, params, test_loader, test_trajectories, HyperParams, history_test):
    """
    Evaluate the performance of a model on a test set.

    This function calculates the mean of the total loss, the mean of loss_test_mse and the mean of loss_test_map for all test examples. The losses are computed as the mean squared error (MSE) between the model's predictions and the true target variables, and between the estimated latent code and the true latent code. The lambda_map weight balances the contribution of each loss term to the total loss. The function adds the computed loss values to the history_test dictionary.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to use for computations (e.g. 'cpu' or 'cuda').
        params (torch.Tensor): Tensor containing the parameters of the model.
        test_loader (torch.utils.data.DataLoader): The test data to use for evaluation.
        test_trajectories (int): The index of the test trajectory.
        HyperParams (object): Object containing hyperparameters for the model.
        history_test (dict): Dictionary to store the evaluation results.

    Returns:
        float: The mean of the total loss computed over all test examples.
    """

    with torch.no_grad():
        model.eval()

        total_loss_test = total_examples = sum_loss = 0
        total_loss_test_1 = total_loss_test_2 = 0
        sum_loss_1 = sum_loss_2 = 0
        for data in test_loader:
            data = data.to(device)
            out, z, z_estimation = model(data, params[test_trajectories, :])
            loss_test_mse = F.mse_loss(out, data.x, reduction='mean')
            loss_test_map = F.mse_loss(z_estimation, z, reduction='mean')
            loss_test = loss_test_mse +  HyperParams.lambda_map * loss_test_map
            sum_loss += loss_test.item()
            sum_loss_1 += loss_test_mse.item()
            sum_loss_2 += loss_test_map.item()
            total_examples += 1

        total_loss_test = sum_loss / total_examples
        total_loss_test_1 = sum_loss_1 / total_examples
        total_loss_test_2 = sum_loss_2 / total_examples
        history_test['test'].append(total_loss_test)
        history_test['l1'].append(total_loss_test_1)
        history_test['l2'].append(total_loss_test_2)
        return total_loss_test

