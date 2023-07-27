import sys
sys.path.append('../')
import torch
from gca_rom import network, pde, loader, plotting, preprocessing, training, initialization, testing, error, gui
import numpy as np
from itertools import product


# Define problem
problem_name, variable, mu_space, n_param = pde.problem(1)
print("\nProblem: ", problem_name)
print("Variable: ", variable)
print("Parameters: ", n_param)

# st = 4
# sf = 3
# sk = 1
# train_rate = 30
# ffc_nodes = 200
# latent_nodes = 50
# btt_nodes = 15
# lambda_map = 1e1
# hidden_channels = 3
# argv = [problem_name, variable, st, sf, sk, train_rate, ffc_nodes,
#         latent_nodes, btt_nodes, lambda_map, hidden_channels, n_param]

argv = gui.hyperparameters_selection(problem_name, variable, n_param)
HyperParams = network.HyperParams(argv)

# Initialize device and set reproducibility
device = initialization.set_device()
initialization.set_reproducibility(HyperParams)
initialization.set_path(HyperParams)

# Load dataset
dataset_dir = '../dataset/'+problem_name+'_unstructured.mat'
dataset = loader.LoadDataset(dataset_dir, variable)

graph_loader, train_loader, test_loader, \
    val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_test, \
        train_trajectories, test_trajectories = preprocessing.graphs_dataset(dataset, HyperParams)

xx = xyz[0]
yy = xyz[1]

params = torch.tensor(np.array(list(product(*mu_space))))
params = params.to(device)

# Define the architecture 
model = network.Net(HyperParams)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams.learning_rate, weight_decay=HyperParams.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=HyperParams.miles, gamma=HyperParams.gamma)
history = dict(train=[], l1=[], l2=[])
history_test = dict(test=[], l1=[], l2=[])
min_test_loss = np.Inf

# Train or load a pre-trained network
try:
    model.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt', map_location=torch.device('cpu')))
    print('Loading saved network')

except FileNotFoundError:
    print('Training net')
    for epoch in range(HyperParams.max_epochs):
        train_rmse = training.train(model, optimizer, device, scheduler, params, train_loader, train_trajectories, HyperParams, history)
        if HyperParams.cross_validation:
            test_rmse = training.val(model, device, params, test_loader, test_trajectories, HyperParams, history_test)
            print("Epoch[{}/{}, train_mse loss:{}, test_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, history['train'][-1], history_test['test'][-1]))
        else:
            test_rmse = train_rmse
            print("Epoch[{}/{}, train_mse loss:{}".format(epoch + 1, HyperParams.max_epochs, history['train'][-1]))
        if test_rmse < min_test_loss:
            min_test_loss = test_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt')
        if HyperParams.tolerance >= train_rmse:
            print('Early stopping!')
            break
        np.save(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', history)
        np.save(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', history_test)

    print("\nLoading best network for epoch: ", best_epoch)
    model.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt', map_location=torch.device('cpu')))

# Evaluate the model
model.to("cpu")
params = params.to("cpu")
vars = "GCA-ROM"
results, latents_map, latents_gca = testing.evaluate(VAR_all, model, graph_loader, params, HyperParams, range(params.shape[0]))

# Plot the results
plotting.plot_loss(HyperParams)
plotting.plot_latent(HyperParams, latents_map, latents_gca)
plotting.plot_error(results, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars)

N = 5
snapshots = np.arange(params.shape[0]).tolist()
np.random.shuffle(snapshots)
for SNAP in snapshots[0:N]:
    plotting.plot_fields(SNAP, results, scaler_all, HyperParams, dataset, xyz, params)
    plotting.plot_error_fields(SNAP, results, VAR_all, scaler_all, HyperParams, dataset, xyz, params)

results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, HyperParams, test_trajectories)

# Print the errors on the testing set
error_abs, norm = error.compute_error(results_test, VAR_test, scaler_test, HyperParams)
error.print_error(error_abs, norm, vars)
error.save_error(error_abs, norm, HyperParams, vars)
