import sys
sys.path.append('../')
import torch
from gca_rom import network, pde, loader, plotting, preprocessing, training, initialization, testing, error
import numpy as np
from itertools import product


# Define problem
problem_name, variable, mu_space, n_param = pde.problem(2)
print("\nProblem: ", problem_name)
print("Variable: ", variable)
print("Parameters: ", n_param)

st = 4
sf = 3
sk = 1
train_rate = 30
ffc_nodes = 200
latent_nodes = 100
btt_nodes = 15
lambda_map = 1e1
hidden_channels = 2

argv = [problem_name, variable, st, sf, sk, train_rate, ffc_nodes,
        latent_nodes, btt_nodes, lambda_map, hidden_channels, n_param]
AE_Params = network.AE_Params(argv)

# Initialize device and set reproducibility
device = initialization.set_device()
initialization.set_reproducibility(AE_Params)
initialization.set_path(AE_Params)

# Load dataset
dataset_dir = '../dataset/'+problem_name+'_unstructured.mat'
dataset = loader.LoadDataset(dataset_dir, variable)

dataset_graph, graph_loader, train_loader, test_loader, \
    val_loader, scaler_all, scaler_test, xyz, var, VAR_all, VAR_test, \
        train_trajectories, test_trajectories = preprocessing.graphs_dataset(dataset, AE_Params)

xx = xyz[0]
yy = xyz[1]
if dataset.dim == 3:
    zz = xyz[2]

params = torch.tensor(np.array(list(product(*mu_space))))
params = params.to(device)

# Define the architecture 
model = network.Net(AE_Params)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=AE_Params.learning_rate, weight_decay=AE_Params.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=AE_Params.miles, gamma=AE_Params.gamma)
history = dict(train=[], l1=[], l2=[])
history_test = dict(test=[], l1=[], l2=[])
min_test_loss = np.Inf

# Train or load a pre-trained network
try:
    model.load_state_dict(torch.load(AE_Params.net_dir+AE_Params.net_name+AE_Params.net_run+'.pt', map_location=torch.device('cpu')))
    print('Loading saved network')

except FileNotFoundError:
    print('Training net')
    for epoch in range(AE_Params.max_epochs):
        train_rmse = training.train(model, optimizer, device, scheduler, params, train_loader, train_trajectories, AE_Params, history)
        if AE_Params.cross_validation:
            test_rmse = training.val(model, device, params, test_loader, test_trajectories, AE_Params, history_test)
            print("Epoch[{}/{}, train_mse loss:{}, test_mse loss:{}".format(epoch + 1, AE_Params.max_epochs, history['train'][-1], history_test['test'][-1]))
        else:
            test_rmse = train_rmse
            print("Epoch[{}/{}, train_mse loss:{}".format(epoch + 1, AE_Params.max_epochs, history['train'][-1]))
        if test_rmse < min_test_loss:
            min_test_loss = test_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), AE_Params.net_dir+AE_Params.net_name+AE_Params.net_run+'.pt')
        if AE_Params.tolerance >= train_rmse:
            print('Early stopping!')
            break
        np.save(AE_Params.net_dir+'history'+AE_Params.net_run+'.npy', history)
        np.save(AE_Params.net_dir+'history_test'+AE_Params.net_run+'.npy', history_test)

    print("\nLoading best network for epoch: ", best_epoch)
    model.load_state_dict(torch.load(AE_Params.net_dir+AE_Params.net_name+AE_Params.net_run+'.pt', map_location=torch.device('cpu')))

# Evaluate the model
model.to("cpu")
params = params.to("cpu")
vars = "GCA-ROM"
results, latents_map, latents_gca = testing.evaluate(VAR_all, model, graph_loader, params, AE_Params, range(params.shape[0]))

# Plot the results
plotting.plot_loss(AE_Params)
plotting.plot_latent(AE_Params, latents_map, latents_gca)
plotting.plot_error(results, VAR_all, scaler_all, AE_Params, mu_space, params, train_trajectories, vars)

N = 5
snapshots = np.arange(params.shape[0]).tolist()
np.random.shuffle(snapshots)
for SNAP in snapshots[0:N]:
    plotting.plot_fields(SNAP, results, scaler_all, AE_Params, dataset, xyz, params)
    plotting.plot_error_fields(SNAP, results, VAR_all, scaler_all, AE_Params, dataset, xyz, params)

results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, AE_Params, test_trajectories)

# Print the errors on the testing set
error_abs, norm = error.compute_error(results_test, VAR_test, scaler_test, AE_Params)
error.print_error(error_abs, norm, vars)
error.save_error(error_abs, norm, AE_Params, vars)
