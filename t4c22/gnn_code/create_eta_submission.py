#  Copyright 2022 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import datetime
from functools import partial
from pathlib import Path

import torch

import t4c22
from t4c22.dataloading.t4c22_dataset import T4c22Dataset  # noqa
from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.evaluation.create_submission import create_submission_cc_plain_torch
from t4c22.evaluation.create_submission import create_submission_eta_torch_geometric
from t4c22.evaluation.test_create_submission import apply_model_geometric
from t4c22.evaluation.test_create_submission import apply_model_plain
from t4c22.evaluation.test_create_submission import DummyRandomNN_cc
from t4c22.misc.t4c22_logging import t4c_apply_basic_logging_config
from t4c22.t4c22_config import load_basedir

import gnn_model

def main(basedir: Path, submission_name: str, model_class, dataset_class, geom=False):
    t4c_apply_basic_logging_config(loglevel="DEBUG")

    cities = ["london", "melbourne", "madrid"]

    config = {}
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

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

    COMPRESS = False
    valOrig = False


    ef = ['parsed_maxspeed', 'importance', 'length_meters']
    EDGE_F = 4
    ETA = True

    gnn_cities = ['models/london_eta.pt', 'models/melbourne_eta.pt', 'models/madrid_eta.pt']

    for cnt, city in enumerate(cities):
        gnn_args = {'compress': False, 'cachedir': Path('tmp/cache_test_set'), 'edge_attributes': ['parsed_maxspeed', 'importance', 'length_meters']}
        test_dataset = dataset_class(root=basedir, city=city, split="test", **gnn_args)


        model = gnn_model.TrafficModel(
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

        model.load_state_dict(torch.load(gnn_cities[cnt]))

        if not geom:
            config[city] = (test_dataset, partial(apply_model_plain, device=device, model=model))
        else:
            config[city] = (test_dataset, partial(apply_model_geometric, device=device, model=model))

    if geom:
        create_submission_eta_torch_geometric(config=config, basedir=basedir, submission_name=submission_name)
    else:
        create_submission_cc_plain_torch(config=config, basedir=basedir, submission_name=submission_name)


if __name__ == "__main__":
    model_class = gnn_model.TrafficModel
    dataset_class = T4c22GeometricDataset
    geom = True

    submission_name = f"{model_class.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(submission_name)
    basedir = load_basedir(fn="t4c22_config.json", pkg=t4c22)

    main(basedir=basedir, submission_name=submission_name, model_class=model_class, dataset_class=dataset_class, geom=geom)
