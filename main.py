import sys
sys.path.append('../')

import torch
from gca_rom import network, pde, loader, plotting, preprocessing, training, initialization, testing, error
import numpy as np


if __name__ == "__main__":
    problem_name, variable, mu1_range, mu2_range = pde.problem(int(sys.argv[1]))
    print("PROBLEM: ", problem_name, "for variable ", variable, "\n")

AE_Params = network.AE_Params
device = initialization.set_device()
initialization.set_reproducibility(AE_Params)
initialization.set_path(AE_Params)

dataset_dir = '../dataset/'+problem_name+'_unstructured.mat'
dataset = loader.LoadDataset(dataset_dir, variable)

dataset_graph, graph_loader, train_loader, test_loader, \
    val_loader, scaler_all, scaler_test, xyz, var, VAR_all, VAR_test, \
        train_trajectories, test_trajectories = preprocessing.graphs_dataset(dataset, AE_Params)

xx = xyz[0]
yy = xyz[1]
if dataset.dim == 3:
    zz = xyz[2]

mu1, mu2 = np.meshgrid(mu1_range, mu2_range)
params = torch.tensor(np.vstack((mu1.T, mu2.T)).reshape(2, -1).T)
params = params.to(device)
print('Dimension of parameter space:', params.shape[1], '\n')

model = network.Net()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=AE_Params.learning_rate, weight_decay=AE_Params.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=AE_Params.miles, gamma=AE_Params.gamma)
history = dict(train=[], l1=[], l2=[])
history_test = dict(test=[], l1=[], l2=[])
min_test_loss = np.Inf

try:
    model.load_state_dict(torch.load(AE_Params.net_dir+AE_Params.net_name+AE_Params.net_run+'.pt', map_location=torch.device('cpu')))
    print('Loading saved network')

except FileNotFoundError:
    print('Training net')
    # with torch.autograd.profiler.profile() as prof:
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

    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    print("\nLoading best network for epoch: ", best_epoch)
    model.load_state_dict(torch.load(AE_Params.net_dir+AE_Params.net_name+AE_Params.net_run+'.pt', map_location=torch.device('cpu')))

model.to("cpu")
params = params.to("cpu")
vars = "GCA-ROM"

results, latents_map, latents_gca = testing.evaluate(VAR_all, model, graph_loader, params, AE_Params, range(params.shape[0]))

plotting.plot_loss(AE_Params)
plotting.plot_latent(AE_Params, latents_map, latents_gca)
plotting.plot_error(results, VAR_all, scaler_all, AE_Params, mu1_range, mu2_range, params, train_trajectories, vars)

N = 5
snapshots = np.arange(params.shape[0]).tolist()
np.random.shuffle(snapshots)
for SNAP in snapshots[0:N]:
    plotting.plot_fields(SNAP, results, scaler_all, AE_Params, dataset, xyz, params)
    plotting.plot_error_fields(SNAP, results, VAR_all, scaler_all, AE_Params, dataset, xyz, params)

results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, AE_Params, test_trajectories)

error_abs, norm = error.compute_error(results_test, VAR_test, scaler_test, AE_Params)
error.print_error(error_abs, norm, vars)
error.save_error(error_abs, norm, AE_Params, vars)
