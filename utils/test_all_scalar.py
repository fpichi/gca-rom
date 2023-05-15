import os

# HYPER-PARAMETERS LIST
problem_names_list = ["poisson", "advection", "graetz", "VX_navier_stokes", "VY_navier_stokes", "P_navier_stokes"]
scalers_type_list = ["sample", "feature", "feature-sampling", "sampling-feature"]
scalers_fun_list = ["minmax", "robust", "standard"]
skip_connection_list = ["skip0", "skip1"]
train_rates_list = [10, 20, 30, 40, 50]
ffc_nodes_list = [50, 100, 200, 300, 400]
latent_nodes_list = [25, 50, 75, 100, 125]
btt_nodes_list = [10, 15, 20, 25, 30]
lambda_map_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
hidden_channels_list = [1, 2, 3, 4, 5]

# HYPER-PARAMETERS TEST
problem_names = ["poisson", "advection", "graetz", "VX_navier_stokes"]
scalers_type = ["sampling-feature"]
scalers_fun = ["standard"]
skip_connection = ["skip1"]
train_rates = [10, 30, 50]
ffc_nodes = [100, 200, 300]
latent_nodes = [50, 100]
btt_nodes = [15, 25]
lambda_map = [1e-1, 1e0, 1e1]
hidden_channels = [1, 2, 3]

# BEST_U_poisson_lmap10.0_btt15_seed10_lv4_hc3_nd50_ffn200_skip1_lr0.001_sc4_rate30
# BEST_U_advection_lmap10.0_btt15_seed10_lv4_hc2_nd100_ffn200_skip1_lr0.001_sc4_rate30
# BEST_U_graetz_lmap10.0_btt25_seed10_lv4_hc2_nd50_ffn200_skip1_lr0.001_sc4_rate30
# BEST_VX_navier_stokes_rid_lmap10.0_btt25_seed10_lv4_hc2_nd100_ffn300_skip1_lr0.001_sc4_rate30
# BEST_VY_navier_stokes_rid_lmap1.0_btt25_seed10_lv4_hc2_nd100_ffn300_skip1_lr0.001_sc4_rate30
# BEST_P_navier_stokes_rid_lmap1.0_btt25_seed10_lv4_hc2_nd100_ffn200_skip1_lr0.001_sc4_rate30

index_pb = [problem_names_list.index(pb)+1 for pb in problem_names]
index_st = [scalers_type_list.index(st)+1 for st in scalers_type]
index_sf = [scalers_fun_list.index(sf)+1 for sf in scalers_fun]
index_sk = [skip_connection_list.index(sk) for sk in skip_connection]

for (i, pb) in zip(index_pb, problem_names):
    for (j, st) in zip(index_st, scalers_type):
        for (k, sf) in zip(index_sf, scalers_fun):
            for (m, sk) in zip(index_sk, skip_connection):
                for rt in train_rates:
                    for ffc in ffc_nodes:
                        for ln in latent_nodes:
                            for bn in btt_nodes:
                                for la in lambda_map:
                                    for hc in hidden_channels:
                                        print("\n\nRUN MAIN FOR PROBLEM "+pb+" WITH SCALING: "+st+" WITH "+sf+" FUNCTION")
                                        print("TRAIN RATE = "+str(rt)+" and FFC NODES = "+str(ffc)+" and HIDDEN CHANNELS = "+str(hc)+" and SKIP = "+str(sk))
                                        print("LATENT NODES = "+str(ln)+" and BTT NODES = "+str(bn)+" and LAMBDA = "+str(la))
                                        os.system("python3 ../main.py %s %s %s %s %s %s %s %s %s %s" %(i, j, k, m, rt, ffc, ln, bn, la, hc))