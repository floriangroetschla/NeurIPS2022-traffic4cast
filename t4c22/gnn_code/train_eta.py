import os
import sys

import statistics
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from tqdm import tqdm
from IPython.core.display import HTML
from IPython.display import display
from torch import nn
from torch_geometric.nn import MessagePassing
from pathlib import Path
import numpy as np

import t4c22
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import class_fractions
from t4c22.t4c22_config import load_basedir
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.plotting.plot_congestion_classification import plot_segment_classifications_simple
from t4c22.misc.notebook_helpers import restartkernel  # noqa:F401

from gnn_model import *

t4c_apply_basic_logging_config(loglevel="DEBUG")



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
def train(model, dataset, optimizer, batch_size, device, ETA = False):
    model.train()

    losses = []
    losses_eta = []
    optimizer.zero_grad()


    #print(torch.cuda.memory_summary())


    for data in tqdm(
        torch_geometric.loader.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        "train",
        total=len(dataset) // batch_size,
    ):

        data = normalize_ft(data)
        #data = subsample(data)


        data = data.to(device)
    #    print(torch.cuda.memory_summary())

        data.x = data.x.nan_to_num(-1)
        data.edge_attr = data.edge_attr.nan_to_num(-1)

        if ETA:
            y_hat, y_eta = model(data)
        else:
            y_hat = model(data)

        y = data.y.nan_to_num(-1)
        y = y.long()

        loss_f = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
        loss_eta = torch.nn.L1Loss()


        if ETA:
            y_eta = torch.flatten(y_eta)
            cc = loss_f(y_hat, y)
            eta = loss_eta(y_eta, data.y_eta)
            loss = cc + eta/80
        else:
            loss = loss_f(y_hat, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if ETA:
            losses.append(cc.detach().cpu().item())
            losses_eta.append(eta.detach().cpu().item())
        else:
            losses.append(loss.detach().cpu().item())

    if ETA:
        print(statistics.mean(losses_eta))

    return losses

@torch.no_grad()
def test(model, validation_dataset, batch_size, device, use_original = False, eta = False):
    model.eval()
    total_loss = 0.0

    y_hat_list = []
    y_list = []
    y_eta_list = []
    y_eta_hat_list = []

    for data in tqdm(validation_dataset, "test", total=len(validation_dataset)):
        data = normalize_ft(data)

        data = data.to(device)

        data.x = data.x.nan_to_num(-1)
        data.edge_attr = data.edge_attr.nan_to_num(-1)
        if ETA:
            y_hat,y_eta = model(data)
        else:
            y_hat = model(data)
        if use_original:
            y_hat_orig = torch.zeros((data.y_orig.shape[0], 3))
            for i in range(y_hat_orig.shape[0]):
                y_hat_orig[i] = y_hat[data.edge_index_index[i]]
            y_hat = y_hat_orig.to(device)


        y_hat_list.append(y_hat)       
        if use_original:
            y_list.append(data.y_orig)
        else: 
            y_list.append(data.y)
        if ETA:
            y_eta = torch.flatten(y_eta)
            y_eta_hat_list.append(y_eta)
            y_eta_list.append(data.y_eta)
    y_hat = torch.cat(y_hat_list, 0)
    y = torch.cat(y_list, 0)
    y = y.nan_to_num(-1)
    y = y.long()
    loss = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)
    loss_eta = torch.nn.L1Loss()

    total_loss = loss(y_hat, y)
    if ETA:
        a = torch.cat(y_eta_hat_list, 0)
        b = torch.cat(y_eta_list, 0)
        print(loss_eta(a,b).detach().cpu())
    #print(f"total losses {total_loss}")
    return total_loss

#Model configuration
hidden_channels = 256
state_channels = 128
num_layers = 6
batch_size = 1
eval_steps = 1
epochs = 100
runs = 1
dropout = 0.1
num_edge_classes = 3
num_features = 14
num_layer_epred = 3+2

recurrent_layers = False
use_supergraph_edges = True
propagate_to_linegraph = True
propagate_in_linegraph = False



device = 0
device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


# Datasets


COMPRESS = False
valOrig = False

#city = "melbourne"
city = "london"
#city = "madrid"

ef = ['parsed_maxspeed', 'importance', 'length_meters']
EDGE_F = 4
CACHE_NAME = "cache"
CPATH = Path('cache')
BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)
ETA = True

print(f"{hidden_channels=}")
print(f"{state_channels=}")
print(f"{num_layers=}")
print(f"{batch_size=}")
print(f"{epochs=}")
print(f"{runs=}")
print(f"{dropout=}")
print(f"{city=}")
print(f"{recurrent_layers=}")
print(f"{use_supergraph_edges=}")
print(f"{propagate_to_linegraph=}")
print(f"{propagate_in_linegraph=}")

print("comment=eta partial")


#generate dataset

dataset = T4c22GeometricDataset(root=BASEDIR, city=city, split="train", compress = COMPRESS, edge_attributes=ef, cachedir=CPATH)

#dataset = dataset[:100]





spl = int(((0.8 * len(dataset)) // 2) * 2)
train_dataset, val_dataset = torch.utils.data.Subset(dataset,np.random.permutation(np.arange(spl))), torch.utils.data.Subset(dataset,range(spl, len(dataset)))

city_class_fractions = class_fractions[city]
city_class_weights = torch.tensor(get_weights_from_class_fractions([city_class_fractions[c] for c in ["green", "yellow", "red"]])).float()


city_class_weights = city_class_weights.to(device)


model = TrafficModel(
        num_features = num_features,
        state_channels = state_channels,
        hidden_channels = hidden_channels,
        num_layers = num_layers,
        edge_dim = EDGE_F, 
        dropout = dropout,
        num_layer_epred = num_layer_epred,
        num_edge_classes = num_edge_classes,
        eta = ETA,
        recurrent_layers = recurrent_layers,
        use_supergraph_edges = use_supergraph_edges,
        propagate_to_linegraph = propagate_to_linegraph,
        propagate_in_linegraph = propagate_in_linegraph
).to(device)


# trainings loop
train_losses    = defaultdict(lambda: [])
val_losses      = defaultdict(lambda: -1)
val_orig_losses      = defaultdict(lambda: -1)

for run in tqdm(range(runs), desc="runs", total=runs):
    # model.reset_parameters()
    #predictor.reset_parameters()
    optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters()},
            ],
            lr=5e-4,
            weight_decay=0.001
        )

    for epoch in tqdm(range(1, 1 + epochs), "epochs", total=epochs):
        losses = train(model, dataset=train_dataset, optimizer=optimizer, batch_size=batch_size, device=device, ETA = ETA)
        train_losses[(run, epoch)] = losses

        print(statistics.mean(losses))
        if epoch % eval_steps == 0:

            val_loss = test(model, validation_dataset=val_dataset, batch_size=batch_size, device=device, eta=ETA)


            val_losses[(run, epoch)] = val_loss
            print(f"val_loss={val_loss} after epoch {epoch} of run {run}")
            torch.save(model.state_dict(), f"models/GNN_model_eta_{city}_{epoch:03d}.pt")
            if valOrig: 
                val_loss_orig = test(model, validation_dataset=val_dataset, batch_size=batch_size, device=device, use_original = True)
                val_orig_losses[(run,epoch)] = val_loss_orig
                print(f"val_orig_loss={val_loss_orig} after epoch {epoch} of run {run}")

            #torch.save(model.state_dict(), f"GNN_model_{epoch:03d}.pt")
            #torch.save(predictor.state_dict(), f"GNN_predictor_{epoch:03d}.pt")
