import os

# HYPER-PARAMETERS FOR POISSON
problem_names = ["VX_navier_stokes", "VY_navier_stokes",  "P_navier_stokes"]
comp = 0
problem_name = problem_names[comp]
scalers_type = "sampling-feature"
scalers_fun = "standard"
skip_connection = "skip1"

pn = 4 + comp
st = 4
sf = 3
sk = 1
train_rate = 10
ffc_nodes = 200
latent_nodes = 100
btt_nodes = 25
lambda_map = 1e0
hidden_channels = 3

print("\n\nRUN MAIN FOR PROBLEM "+problem_name+" WITH SCALING: "+scalers_type+" WITH "+scalers_fun+" FUNCTION")
print("TRAIN RATE = "+str(train_rate)+" and FFC NODES = "+str(ffc_nodes)+" and HIDDEN CHANNELS = "+str(hidden_channels)+" and SKIP = "+skip_connection)
print("LATENT NODES = "+str(latent_nodes)+" and BTT NODES = "+str(btt_nodes)+" and LAMBDA = "+str(lambda_map))
os.system("python3 ../main.py %s %s %s %s %s %s %s %s %s %s" %(pn, st, sf, sk, train_rate, ffc_nodes, latent_nodes, btt_nodes, lambda_map, hidden_channels))