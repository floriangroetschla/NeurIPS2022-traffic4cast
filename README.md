#  Traffic4cast 2022 Competition: Hierarchical graph structures for congestion and ETA prediction

## About
This repo is a fork of the [original competition codebase](https://github.com/iarai/NeurIPS2022-traffic4cast) and contains all the necessary code to train and evaluate our models. Trained models for the cc and eta task are also included.

## Setup instructions
The setup is very similar to the original competition code, instructions can also be found [here](README_competition.md)

First, setup the environment:

```bash
conda env update -f environment.yaml
conda activate t4c22

# Install pyG
CUDA="cu113"
python -m pip install -r install-extras-torch-geometric.txt -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html

python t4c22/misc/check_torch_geometric_setup.py
```

Then follow the instructions [here](README_competition.md) to download the data, extract it and create labels in a data directory of your choice.
Edit `t4c22/t4c22_config.json` to link towards this data directory.
Lastly, set the PYTHONPATH to the base of the repo clone:

```
export PYTHONPATH=<root of the repo clone>
```
Now you are ready to run trainings and create submission files.

## Create submissions with pretrained models
Our pretrained models are located in `t4c22/gnn_code/models` (one model per task and city).
To create submissions that use them, run the script according to the task (note that the scripts assume that you have a CUDA device available): 
```bash
cd t4c22/gnn_code

# Core competition
python create_cc_submission.py

# Extended competition
python create_eta_submission.py
```
The code is setup to load the pretrained models, to use your own, specify the path to your models in the `gnn_cities` list in the scripts and make sure that the model configuration is correct.
Also, the `basedir` can be set to another path to load other test data.

## Train models
We provide scripts to train the cc models on the full dataset and the eta models on an 80/20 split. The cc training uses Approach (2) as explained in the paper and the eta training a combination of Approach (1) and (2)
To run the trainings:
```bash
cd t4c22/gnn_code

# Start training for the core competition
python train_cc.py

# Start training for the extended competition
python train_eta.py
```

Models are stored in `models` after every epoch and can be used for the submission creation.


## Cite
To cite our work:

```
@InProceedings{to be announced,
}

```

