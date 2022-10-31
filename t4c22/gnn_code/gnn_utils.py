import torch
import numpy as np

def normalize_ft(data):
    #normalize the traffic counts

    data.x[:,:4] /= 300

    speed, important, length = 0,1,2
    n, fe = data.edge_attr.shape
    e = torch.full(size=(n, 4), fill_value=float("nan"))

    e[:, 3] = data.edge_attr[:,length]/data.edge_attr[:,speed]
    e[:,:3] = data.edge_attr
    data.edge_attr = e


    data.edge_attr[:, speed] /= 120
    data.edge_attr[:, important] /= 5
    data.edge_attr[:, length] /= 100
    data.edge_attr[:, 3] /= 3

    means = torch.tensor(np.nanmean(data.x[:,:4], axis = 0))
    stds  = torch.tensor(np.nanstd(data.x[:,:4], axis = 0))

    global_features = torch.cat((means,stds),dim=0)

    n, fe = data.x.shape
    ft = torch.full(size=(n,14), fill_value=float("nan"))
    ft[:,:6] = data.x
    ft[:,6:] = global_features
    data.x = ft

    return data


def normalize_ft_old(data):
    #normalize the traffic counts

    data.x[:,:4] /= 300

    speed, important, length = 0,1,2
    n, fe = data.edge_attr.shape
    e = torch.full(size=(n, 4), fill_value=float("nan"))

    e[:, 3] = data.edge_attr[:,length]/data.edge_attr[:,speed]
    e[:,:3] = data.edge_attr
    data.edge_attr = e


    data.edge_attr[:, speed] /= 120
    data.edge_attr[:, important] /= 5
    data.edge_attr[:, length] /= 100
    data.edge_attr[:, 3] /= 3


    return data

