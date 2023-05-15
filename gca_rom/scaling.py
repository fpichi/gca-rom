from sklearn import preprocessing
import torch
import sys


def scaler_functions(k):
    match k:
        case 1:
            sc_name = "minmax"
            sc_fun = preprocessing.MinMaxScaler()
        case 2:
            sc_name = "robust"
            sc_fun = preprocessing.RobustScaler()
        case 3:
            sc_name = "standard"
            sc_fun = preprocessing.StandardScaler()
    return sc_fun, sc_name


def tensor_scaling(tensor, scaling_type):
    scaling_fun_1, _ = scaler_functions(int(sys.argv[3]))
    scaling_fun_2, _ = scaler_functions(int(sys.argv[3]))
    match scaling_type:
        case 1:
            # print("SAMPLE SCALING")
            scale = scaling_fun_1.fit(tensor)
            scaled_data = torch.unsqueeze(torch.tensor(scale.transform(tensor)),0).permute(2, 1, 0)
        case 2:
            # print("FEATURE SCALING")
            scale = scaling_fun_1.fit(torch.t(tensor))
            scaled_data = torch.unsqueeze(torch.tensor(scale.transform(torch.t(tensor))), 0).permute(1, 2, 0)
        case 3:
            # print("FEATURE-SAMPLE SCALING")
            scaler_f = scaling_fun_1.fit(torch.t(tensor))
            temp = torch.tensor(scaler_f.transform(torch.t(tensor)))
            scaler_s = scaling_fun_2.fit(temp)
            scaled_data = torch.unsqueeze(torch.tensor(scaler_s.transform(temp)), 0).permute(1, 2, 0)
            scale = [scaler_f, scaler_s]
        case 4:
            # print("SAMPLE-FEATURE SCALING")
            scaler_s = scaling_fun_1.fit(tensor)
            temp = torch.t(torch.tensor(scaler_s.transform(tensor)))
            scaler_f = scaling_fun_2.fit(temp)
            scaled_data = torch.unsqueeze(torch.t(torch.tensor(scaler_f.transform(temp))), 0).permute(2, 1, 0)
            scale = [scaler_s, scaler_f]
    return scale, scaled_data


def inverse_scaling(tensor, scale, scaling_type):
    match scaling_type:
        case 1:
            # print("SAMPLE SCALING")
            rescaled_data = torch.tensor(scale.inverse_transform(torch.t(torch.tensor(tensor[:, :, 0].detach().numpy().squeeze()))))
        case 2:
            # print("FEATURE SCALING")
            rescaled_data = torch.tensor(torch.t(torch.tensor(scale.inverse_transform(tensor[:, :, 0].detach().numpy().squeeze()))))
        case 3:
            # print("FEATURE-SAMPLE SCALING")
            scaler_f = scale[0]
            scaler_s = scale[1]
            rescaled_data = torch.t(torch.tensor(scaler_f.inverse_transform(torch.tensor(scaler_s.inverse_transform(tensor[:, :, 0].detach().numpy().squeeze())))))
        case 4:
            # print("SAMPLE-FEATURE SCALING")
            scaler_s = scale[0]
            scaler_f = scale[1]
            rescaled_data = torch.tensor(scaler_s.inverse_transform(torch.t(torch.tensor(scaler_f.inverse_transform(tensor[:, :, 0].detach().numpy().squeeze())))))
    return rescaled_data 
