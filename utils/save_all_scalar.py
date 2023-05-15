import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
import seaborn as sns
import os
warnings.filterwarnings("ignore")

# HYPER-PARAMETERS LIST
model_names_list = ["poisson", "advection", "graetz", "navier_stokes"]
problem_names_list = ["poisson", "advection", "graetz", "VX_navier_stokes", "VY_navier_stokes", "P_navier_stokes"]
scalers_type_list = ["sample", "feature", "feature-sampling", "sampling-feature"]
scalers_fun_list = ["minmax", "robust", "standard"]
skip_connection_list = ["skip0", "skip1"]
train_rates_list = ["rate10", "rate20", "rate30", "rate40", "rate50"]
ffc_nodes_list = ["ffn50", "ffn100", "ffn200", "ffn300", "ffn400"]
latent_nodes = ["nd25", "nd50", "nd75", "nd100", "nd125"]
btt_nodes = ["btt10", "btt15", "btt20", "btt25", "btt30"]
lambda_map_list = ["lmap0.01", "lmap0.1", "lmap1.0", "lmap10", "lmap100"]
hidden_channels_list = ["hc1", "hc2", "hc3", "hc4", "hc5"]

# HYPER-PARAMETERS SAVE
model_names = ["poisson", "advection", "graetz", "navier_stokes", "navier_stokes", "navier_stokes"]
problem_names = ["poisson", "advection", "graetz", "VX_navier_stokes", "VY_navier_stokes", "P_navier_stokes"]
variable_names = ["U_", "U_", "U_", "VX_", "VY_", "P_"]
scalers_fun = ["standard"]
scalers_type = ["sc4"]
skip_connection = ["skip1"]
train_rates = ["rate10", "rate30", "rate50"]
ffn_nodes = ["ffn100", "ffn200", "ffn300"]
latent_nodes = ["nd50", "nd100"]
btt_nodes = ["btt15", "btt25"]
lambda_map = ["lmap0.1", "lmap1.0", "lmap10.0"]
hidden_channels = ["hc1", "hc2", "hc3"]

train_rates_l = [10, 30, 50]
ffn_nodes_l = [100, 200, 300]
latent_nodes_l = [50, 100]
btt_nodes_l = [15, 25]
lambda_map_l = [0.1, 1.0, 10.0]
hidden_channels_l = [1, 2, 3]

folder = ""
for name in [*set(model_names)]:
    folder += str(name)+"/_standard/ "
print("FOLDERS: ", folder)

## SCALAR CASES
output = subprocess.check_output("find " + folder + "-name '*GCA-ROM.txt' -print  > res_all.txt", shell=True)
file = 'res_all.txt'
sim_list = open(file).readlines()

error = dict()
for name in problem_names:
    for sct in scalers_type:
        for scf in scalers_fun:
            for sk in skip_connection:
                for rt in train_rates:
                    for ffn in ffn_nodes:
                        for ln in latent_nodes:
                            for bn in btt_nodes:
                                for la in lambda_map:
                                    for hc in hidden_channels:
                                        error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc] = list()

data = pd.DataFrame()

k = 0
for (j, r) in enumerate(sim_list):
    r = r.replace('//', '/').replace('\n', '')
    e = pd.read_csv(r, sep=" ", header=None).to_numpy().squeeze()
    for it_1, name in enumerate(problem_names):
        for it_2, sct in enumerate(scalers_type):
            for it_3, scf in enumerate(scalers_fun):
                for it_4, sk in enumerate(skip_connection):
                    for it_5, rt in enumerate(train_rates):
                        for it_6, ffn in enumerate(ffn_nodes):
                            for it_7, ln in enumerate(latent_nodes):
                                for it_8, bn in enumerate(btt_nodes):
                                    for it_9, la in enumerate(lambda_map):
                                        for it_10, hc in enumerate(hidden_channels):    
                                            if all(x in r for x in [name, sct, scf, sk, rt, ffn, ln, bn, la, hc]):
                                                print(name, sct, scf, sk, rt, ffn, ln, bn, la, hc, e)
                                                if not error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc]:
                                                    error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc].append(e)
                                                    temp = pd.DataFrame({'problem_names': it_1,
                                                                        'scalers_type': it_2,
                                                                        'scalers_fun': it_3,
                                                                        'skip_connection': it_4,
                                                                        'train_rates': train_rates_l[it_5],
                                                                        'ffn_nodes': ffn_nodes_l[it_6],
                                                                        'latent_nodes': latent_nodes_l[it_7],
                                                                        'btt_nodes': btt_nodes_l[it_8],
                                                                        'lambda_map': lambda_map_l[it_9],
                                                                        'hidden_channels': hidden_channels_l[it_10],
                                                                        'error': e[1]} , index=[k])
                                                    data = pd.concat([data, temp])
                                                    k+=1
                                                else:
                                                    error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc][0] = np.concatenate([error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc][0], np.array(e)])

my_xticks = []
for sct in scalers_type:
    for scf in scalers_fun:
        for sk in skip_connection:
            for rt in train_rates:
                for ffn in ffn_nodes:
                    for ln in latent_nodes:
                        for bn in btt_nodes: 
                            for la in lambda_map:
                                for hc in hidden_channels:                   
                                    my_xticks.append(sct+scf+sk+rt+ffn+ln+bn+la+hc)
num_exp = len(scalers_fun)*len(scalers_type)*len(skip_connection)*len(train_rates)*len(ffn_nodes)*len(latent_nodes)*len(btt_nodes)*len(lambda_map)*len(hidden_channels)
x = range(num_exp)

er = np.array(list(error.items()))[:, 1]
for y in er:
    if not y:
        y.append(np.ones(3))
error_array = np.vstack([y for y in er])

def plot_error(ax, name, x, var, i_min):
    i = 0
    ax.set_title(name)
    for sct in scalers_type:
        for scf in scalers_fun:
            for sk in skip_connection:
                for rt in train_rates:
                    for ffn in ffn_nodes:
                        for ln in latent_nodes:
                            for bn in btt_nodes:
                                for la in lambda_map:
                                    for hc in hidden_channels:  
                                        if np.array_equal(error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc][0], np.ones(3)):
                                            color_min_max = "white"
                                            col_mean = "white"
                                        else:
                                            if i == i_min:
                                                print("HYPERPARAMETERS: ", name, sct, scf, sk, rt, ffn, ln, bn, la, hc)
                                                string = str(model)+"/_standard/"+str(var)+str(model)+"_"+str(la)+"_"+str(bn)+"_seed"+str(10)+"_lv"+str(4)+"_"+str(hc)+"_"+str(ln)+"_"+str(ffn)+"_"+str(sk)+"_lr"+str(0.001)+"_sc"+str(4)+"_"+str(rt)+"/"
                                                print("FOLDER = ", string)
                                                color_min_max = "#1C3144"
                                                col_mean = "#4cb944"
                                            else:
                                                color_min_max = "#1C3144"
                                                col_mean = "#70161E"
                                            ax.scatter(i, error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc][0][0], marker="^", facecolors='white', edgecolors=color_min_max)
                                            ax.scatter(i, error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc][0][1], marker="d", facecolors=col_mean, edgecolors=col_mean)
                                            ax.scatter(i, error[name, sct, scf, sk, rt, ffn, ln, bn, la, hc][0][2], marker="v", facecolors='white', edgecolors=color_min_max)
                                            ax.set_yscale('log')
                                            i+=1
    ax.tick_params(labelsize=8)
    # ax.set_xticks(x[:i], my_xticks[:i], rotation=65)
    ax.set_xlim([0, len(x[:i])])


# dim = np.reshape(range(len(problem_names)), (2, -1)).shape
# fig2, axs2 = plt.subplot_mosaic([['upper left', 'upper mid', 'upper right'],
#                                  ['lower left', 'lower mid', 'lower right']])

for i, (name, model, var) in enumerate(zip(problem_names, model_names, variable_names)):
    model_path = '../plots/'+str(model)+'/tmp'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    fig1, axs1 = plt.subplots(1)
    i_min = np.argmin(error_array[i*num_exp:(i+1)*num_exp, 1], axis=0)
    print("\nBEST NETWORK FOR " + str(name) + " IS CORRESPONDING TO i = ", i_min)
    plot_error(axs1, name, x, var, i_min)
    # plot_error(axs2[k], name, x, var, i_min)
    fig1.tight_layout()
    fig1.savefig('../plots/'+str(model)+'/box_plot_error_all_'+str(name)+'.png', dpi=500)
    plt.close()
# fig2.tight_layout()
# fig2.savefig('../plots/box_plot_error_all.png', dpi=500)    
# plt.show()



### PAIRPLOT

# print("Dataset shape: ", data.shape)
# print("Dataset: ", data)
columns_names = ["train_rates", "ffn_nodes", "latent_nodes", "btt_nodes", "lambda_map", "hidden_channels", "error"]
list_hyper = [train_rates, ffn_nodes, latent_nodes, btt_nodes, lambda_map, hidden_channels]
x_vars = ["train_rates", "ffn_nodes", "latent_nodes", "btt_nodes", "lambda_map", "hidden_channels"]
x_vars_title = ["$r_t$", "ffn", "$n_l$", "$n$", "$\lambda$", "hc"]
y_vars = ["error"]
n_vars = len(x_vars)

mark =["o", "s", "D"]
line =["-", "--", "-."]

def pairplot_box(data, xv, hue_var):
    sns.set(style="ticks")
    return sns.catplot(data=data, x=xv, y=y_vars[0], hue=hue_var, kind="box")

def pairplot_point(data, xv, c, hue_var):
    sns.set(style="ticks")
    return sns.catplot(data=data, x=xv, y=y_vars[0], hue=hue_var, linestyles=line[:len(list_hyper[c])], markers=mark[:len(list_hyper[c])], kind="point")
    # return sns.pairplot(data[columns_names], x_vars=x_vars, y_vars=y_vars, hue=hue_var, markers=mark[:len(col)])

fontsize = 25
for n, (name, model) in enumerate(zip(problem_names, model_names)):
    data_name = data[data["problem_names"]==n]
    # print("Dataset name shape: ", data_name.shape)
    # print("Dataset name: ", data_name)
    j = 0
    for c, col in enumerate(x_vars):
        for tit, xv in enumerate(x_vars):
            counter = 0
            if xv == col:
                counter += 1
            else:
                g1 = pairplot_box(data_name, xv, col)
                g2 = pairplot_point(data_name, xv, c, col)
                g1._legend.remove()
                g2._legend.remove()
                for ax in g1.axes.flat:
                    if ax.get_ylabel() in y_vars:
                        ax.set(yscale="log")
                    ax.set_xlabel(x_vars_title[tit+counter], fontsize=fontsize)
                    ax.set_ylabel(y_vars[0], fontsize=fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize)
                    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
                    ax.tick_params(labelsize=fontsize)
                    if j in [nn - 1 for nn in range(0, n_vars*(n_vars - 1) + 1, n_vars - 1)[1:]]:
                        ax.legend(title=x_vars_title[c+counter], title_fontsize=fontsize)
                        sns.move_legend(ax, "center left", bbox_to_anchor=(1., 0.5), ncol=1, frameon=False, fontsize=fontsize)
                g1.savefig('../plots/'+str(model)+'/tmp/g1_'+str(j)+'_'+str(name)+'.png', dpi=500)
                plt.close(g1.fig)
                for ax in g2.axes.flat:
                    if ax.get_ylabel() in y_vars:
                        ax.set(yscale="log")
                    ax.set_xlabel(x_vars_title[tit+counter], fontsize=fontsize)
                    ax.set_ylabel(y_vars[0], fontsize=fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=fontsize)
                    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
                    ax.tick_params(labelsize=fontsize)
                    if j in [nn - 1 for nn in range(0, n_vars*(n_vars - 1) + 1, n_vars - 1)[1:]]:
                        ax.legend(title=x_vars_title[c+counter], title_fontsize=fontsize)
                        sns.move_legend(ax, "center left", bbox_to_anchor=(1., 0.5), ncol=1, frameon=False, fontsize=fontsize)
                g2.savefig('../plots/'+str(model)+'/tmp/g2_'+str(j)+'_'+str(name)+'.png', dpi=500)
                plt.close(g2.fig)
                j += 1


    f1, axarr1 = plt.subplots(len(x_vars), len(x_vars[:-1]), figsize=(20, 20))
    f2, axarr2 = plt.subplots(len(x_vars), len(x_vars[:-1]), figsize=(20, 20))
    i = 0
    for c, col in enumerate(x_vars):
        for cc, col in enumerate(x_vars[:-1]):
            axarr1[c,cc].imshow(mpimg.imread('../plots/'+str(model)+'/tmp/g1_'+str(i)+'_'+str(name)+'.png'))
            axarr2[c,cc].imshow(mpimg.imread('../plots/'+str(model)+'/tmp/g2_'+str(i)+'_'+str(name)+'.png'))
            i += 1
            [ax.set_axis_off() for ax in axarr1.ravel()]
            [ax.set_axis_off() for ax in axarr2.ravel()]

    # shutil.rmtree('../plots/'+str(model)+'/tmp/')
    f1.tight_layout()
    f2.tight_layout()
    f1.savefig('../plots/'+str(model)+'/sns_error_all_box_'+str(name)+'.png', dpi=500)
    f2.savefig('../plots/'+str(model)+'/sns_error_all_point_'+str(name)+'.png', dpi=500)