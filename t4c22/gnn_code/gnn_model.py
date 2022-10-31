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


activation = torch.nn.ReLU

#TODO this should probably be much better!
def get_mlp(self, input_dim, hidden_dim, output_dim, normalization = torch.nn.Identity, last_relu = True, num_hidden_layers = 2):
    modules = [ torch.nn.Linear(input_dim, int(hidden_dim)), normalization(int(hidden_dim)), activation(), torch.nn.Dropout(self.dropout)]
    for layer in range(num_hidden_layers-1):
        modules += [ torch.nn.Linear(int(hidden_dim), int(hidden_dim)), normalization(int(hidden_dim)), activation(), torch.nn.Dropout(self.dropout)]
    modules += [torch.nn.Linear(int(hidden_dim), output_dim)]
    
    if last_relu:
        modules.append(normalization(output_dim))
        modules.append(activation())
    return torch.nn.Sequential(*modules)

class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_Layer(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    """

    def __init__(self, in_features, out_features, hidden_features, edge_dim = 5, edge_encoder = None, dropout = 0.0):
        super(GNN_Layer, self).__init__(node_dim=-2, aggr="add")
        self.edge_encoder = edge_encoder
        self.dropout = dropout

        self.message_net = get_mlp(self,3*in_features, hidden_features, out_features)

        self.update_net = get_mlp(self,in_features + out_features, hidden_features, out_features)

    def forward(self, x, edge_index, edge_attr, batch):
        """Propagate messages along edges."""
        x = self.propagate(edge_index, edge_attr=edge_attr, x=x)
        # x = self.norm(x, batch)
        return x

    def message(self, edge_attr,x_i, x_j):
        """Message update."""
        Z = torch.cat((x_i, x_j, self.edge_encoder(edge_attr)), dim=-1)
        message = self.message_net(Z)
        return message

    def update(self, message, x):
        """Node update."""
        x = x + self.update_net(torch.cat((x, message), dim=-1))
        return x

class SuperLayer(MessagePassing):
    
    def __init__(self, hidden_features, dropout = 0.0):
        super(SuperLayer, self).__init__(aggr='add')
        self.dropout = dropout

        self.message_net = get_mlp(self, 2*hidden_features, hidden_features, hidden_features)

        self.update_net = get_mlp(self, 2*hidden_features, hidden_features, hidden_features)

    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        return x

    def message(self, x_i, x_j):
        Z = torch.cat((x_i, x_j), dim=-1)
        message = self.message_net(Z)
        return message

    def update(self, message, x):
        x = x + self.update_net(torch.cat((x, message), dim=-1))
        return x


class CongestioNN(torch.nn.Module):
    def __init__(self, num_features=32, hidden_features=32, num_layers=6, edge_dim = 5,vertex_encoder = None, edge_encoder = None, dropout = 0.0, recurrent_layers=True, use_supergraph_edges=False, propagate_to_linegraph=False, propagate_in_linegraph=False):

        super(CongestioNN, self).__init__()
        self.num_features   = num_features
        self.hidden_features = hidden_features
        self.num_layers   = num_layers
        self.dropout = dropout
        self.recurrent_layers = recurrent_layers
        self.use_supergraph_edges = use_supergraph_edges
        self.propagate_to_linegraph = propagate_to_linegraph
        self.propagate_in_linegraph = propagate_in_linegraph

        if self.recurrent_layers:
            self.original_graph_gnn = GNN_Layer(self.num_features, self.num_features, self.hidden_features, edge_dim=edge_dim, edge_encoder = edge_encoder, dropout = self.dropout)
        else:
            self.original_graph_gnns = torch.nn.ModuleList([GNN_Layer(self.num_features, self.num_features, self.hidden_features, edge_dim=edge_dim, edge_encoder=edge_encoder, dropout = self.dropout) for i in range(num_layers)])

        if self.use_supergraph_edges:
            self.supergraph_edge_gnn = SuperLayer(self.num_features, dropout=self.dropout)

        if self.propagate_to_linegraph:
            self.to_line_graph_gnn = SuperLayer(self.num_features, dropout=self.dropout)
            self.from_line_graph_gnn = SuperLayer(self.num_features, dropout=self.dropout)

        if self.propagate_in_linegraph:
            self.in_line_graph_gnn = SuperLayer(self.num_features, dropout=self.dropout)


        #self.skip_network = get_mlp(self,self.out_features + self.in_features, self.hidden_features, self.out_features)

        self.vertex_encoder = vertex_encoder

    def forward(self, data):
        batch = data.batch
        x = data.x
        edge_index = data.edge_index
        if self.propagate_to_linegraph:
            edge_index_from_super = torch.flip(data.edge_index_to_super, dims=[0,1])
        edge_attr  = data.edge_attr
        x = self.vertex_encoder(x)

        for i in range(self.num_layers):
            #x = self.skip_network(torch.cat((x, inp), dim=-1))
            if self.recurrent_layers:
                x = self.original_graph_gnn(x, edge_index, edge_attr, batch)
            else:
                x = self.original_graph_gnns[i](x, edge_index, edge_attr, batch)
            
            if self.use_supergraph_edges and (i % 2 == 0):
                x = self.supergraph_edge_gnn(x, data.edge_index_supergraph)

            if self.propagate_to_linegraph and (i % 3 == 0) and (i != 0):
                x = self.to_line_graph_gnn(x, data.edge_index_to_super)

            if self.propagate_in_linegraph and (i % 3 == 0) and (i != 0):
                x = self.in_line_graph_gnn(x, data.edge_index_linegraph)
                x = self.in_line_graph_gnn(x, data.edge_index_linegraph)

            if self.propagate_to_linegraph and (i % 3 == 0) and (i != 0):
                x = self.from_line_graph_gnn(x, edge_index_from_super)

        return x


class LinkPredictor(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_dim = 5,vertex_encoder = None, edge_encoder = None):
        super(LinkPredictor, self).__init__()
        self.dropout = dropout

        self.edge_encoder = edge_encoder
        self.vertex_encoder = vertex_encoder
        self.net = get_mlp(self, in_channels, 2*hidden_channels, out_channels, last_relu = False, num_hidden_layers=4)

    def forward(self, x_i, x_j, edge_attr, x_i_orig, x_j_orig):
        #        x = x_i * x_j
        x = torch.cat((x_i, x_j, self.edge_encoder(edge_attr), self.vertex_encoder(x_i_orig), self.vertex_encoder(x_j_orig)), dim=-1)
        x = self.net(x)
        return x


class LinkPredictorSuper(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_dim = 5,vertex_encoder = None, edge_encoder = None):
        super(LinkPredictorSuper, self).__init__()
        self.dropout = dropout

        self.edge_encoder = edge_encoder
        self.vertex_encoder = vertex_encoder
        self.net = get_mlp(self, in_channels, 2*hidden_channels, out_channels, last_relu = False, num_hidden_layers=4)

    def forward(self, x_i, x_j, x_i_orig, x_j_orig, supersegments):
        #        x = x_i * x_j
        x = torch.cat((x_i, x_j, self.vertex_encoder(x_i_orig), self.vertex_encoder(x_j_orig), supersegments), dim=-1)
        x = self.net(x)
        return x

import t4c22.t4c22_config


class TrafficModel(torch.nn.Module):
    def __init__(self, num_features = 4, state_channels = 32, hidden_channels=256, num_layers = 10, edge_dim = 5, dropout = 0.0, num_layer_epred = 5, num_edge_classes=3, num_edges = 0, eta=False, recurrent_layers=True, use_supergraph_edges=False, propagate_to_linegraph=False, propagate_in_linegraph=False):
        super(TrafficModel, self).__init__()

        self.dropout = dropout
        self.use_supergraph_edges = use_supergraph_edges
        self.propagate_to_linegraph = propagate_to_linegraph
        self.propagate_in_linegraph = propagate_in_linegraph
        self.recurrent_layers = True

        self.vertex_encoder = get_mlp(self,num_features, hidden_channels, state_channels, num_hidden_layers=3)
        self.edge_encoder   = get_mlp(self,edge_dim    , hidden_channels, state_channels, num_hidden_layers=3)


        # get model
        self.gnn = CongestioNN(state_channels, hidden_channels, num_layers, edge_dim = edge_dim, vertex_encoder = self.vertex_encoder, 
                    edge_encoder = self.edge_encoder, dropout = self.dropout, recurrent_layers=recurrent_layers, use_supergraph_edges = use_supergraph_edges, propagate_to_linegraph = propagate_to_linegraph, propagate_in_linegraph = propagate_in_linegraph)

        self.predictor = LinkPredictor(5*state_channels, hidden_channels, num_edge_classes, num_layer_epred, dropout,
                         edge_dim = edge_dim, vertex_encoder = self.vertex_encoder, edge_encoder = self.edge_encoder)

        self.eta = eta
        
        if self.eta:
            multiplier = 5
        else:
            multiplier = 4
        self.predictor_super = LinkPredictorSuper(multiplier*state_channels, hidden_channels, 1, num_layer_epred, dropout,
                         edge_dim = edge_dim, vertex_encoder = self.vertex_encoder, edge_encoder = self.edge_encoder)

    def forward(self, data):
        h = self.gnn(data)

        x_i = torch.index_select(h, 0, data.edge_index[0])
        x_j = torch.index_select(h, 0, data.edge_index[1])
        x_i_orig = torch.index_select(data.x, 0, data.edge_index[0])
        x_j_orig = torch.index_select(data.x, 0, data.edge_index[1])
        #TODO FIX that prediction is dependent on x_i - > x_j prediction
        y_hat = self.predictor(x_i, x_j, data.edge_attr, x_i_orig, x_j_orig)


        if self.eta:
            if data.x.shape[0] == 63622:
                city = "london"
            elif data.x.shape[0] == 53202:
                city = "melbourne"
            else:
                city = "madrid"
            #city = "london"
            length = t4c22.t4c22_config.NUM_SUPERSEGMENTS[city]
            x_i_s = torch.index_select(h, 0, data.edge_index_supergraph[0][:length])
            x_j_s = torch.index_select(h, 0, data.edge_index_supergraph[1][:length])

            x_i_orig_s = torch.index_select(data.x, 0, data.edge_index_supergraph[0][:length])
            x_j_orig_s = torch.index_select(data.x, 0, data.edge_index_supergraph[1][:length])
            start = t4c22.t4c22_config.NUM_NODES[city]
            supersegments = h[start:(start+length),:]
            y_s = 100*self.predictor_super(x_i_s, x_j_s, x_i_orig_s, x_j_orig_s, supersegments)
            return y_hat, y_s
        return y_hat
