import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 

def train(model, optimizer, device, scheduler, params, train_loader, test_loader, train_trajectories, test_trajectories, HyperParams):
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
    train_history = dict(train=[], l1=[], l2=[])
    test_history = dict(test=[], l1=[], l2=[])
    min_test_loss = np.Inf

    model.train()
    loop = tqdm(range(HyperParams.max_epochs))
    for epoch in loop:
        train_rmse = total_examples = sum_loss = 0
        train_rmse_1 = train_rmse_2 = 0
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
        train_rmse = sum_loss / total_examples
        train_rmse_1 = sum_loss_1 / total_examples
        train_rmse_2 = sum_loss_2 / total_examples
        train_history['train'].append(train_rmse)
        train_history['l1'].append(train_rmse_1)
        train_history['l2'].append(train_rmse_2)

        if HyperParams.cross_validation:
            with torch.no_grad():
                model.eval()
                test_rmse = total_examples = sum_loss = 0
                test_rmse_1 = test_rmse_2 = 0
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

                test_rmse = sum_loss / total_examples
                test_rmse_1 = sum_loss_1 / total_examples
                test_rmse_2 = sum_loss_2 / total_examples
                test_history['test'].append(test_rmse)
                test_history['l1'].append(test_rmse_1)
                test_history['l2'].append(test_rmse_2)
            # print("Epoch[{}/{}, train_mse loss:{}, test_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, train_history['train'][-1], test_history['test'][-1]))
            loop.set_postfix({"Loss(training)": train_history['train'][-1], "Loss(validation)": test_history['test'][-1]})
        else:
            test_rmse = train_rmse
            # print("Epoch[{}/{}, train_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, train_history['train'][-1]))
            loop.set_postfix({"Loss(training)": train_history['train'][-1]})

        if test_rmse < min_test_loss:
            min_test_loss = test_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt')
        if HyperParams.tolerance >= train_rmse:
            print('Early stopping!')
            break
        np.save(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', train_history)
        np.save(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', test_history)
    
    print("\nLoading best network for epoch: ", best_epoch)
    model.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt', map_location=torch.device('cpu')))
