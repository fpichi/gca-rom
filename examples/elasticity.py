import os

# HYPER-PARAMETERS FOR ELASTICITY
problem_name = "elasticity" 
scalers_type = "sampling-feature"
scalers_fun = "standard"
skip_connection = "skip1"

pn = 9
st = 4
sf = 3
sk = 1
train_rate = 30
ffc_nodes = 200
latent_nodes = 50
btt_nodes = 15
lambda_map = 1e1
hidden_channels = 3

print("\n\nRUN MAIN FOR PROBLEM "+problem_name+" WITH SCALING: "+scalers_type+" WITH "+scalers_fun+" FUNCTION")
print("TRAIN RATE = "+str(train_rate)+" and FFC NODES = "+str(ffc_nodes)+" and HIDDEN CHANNELS = "+str(hidden_channels)+" and SKIP = "+skip_connection)
print("LATENT NODES = "+str(latent_nodes)+" and BTT NODES = "+str(btt_nodes)+" and LAMBDA = "+str(lambda_map))
os.system("python3 ../main.py %s %s %s %s %s %s %s %s %s %s" %(pn, st, sf, sk, train_rate, ffc_nodes, latent_nodes, btt_nodes, lambda_map, hidden_channels))