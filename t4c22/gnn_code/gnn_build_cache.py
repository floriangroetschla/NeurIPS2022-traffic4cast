import os
import sys
import numpy as np


sys.path.insert(0, os.path.abspath("../"))  # noqa:E402


def task(arg):
    sys.path.insert(0, os.path.abspath("../"))  # noqa:E402


    from pathlib import Path


    import t4c22
    from t4c22.t4c22_config import load_basedir
    from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
    
    BASEDIR = load_basedir(fn="t4c22_config.json", pkg=t4c22)


    path = "cache"
    cities = ['madrid', 'melbourne', 'london']
    edge_features = ['parsed_maxspeed', 'importance', 'length_meters']
    compress = True

    datasets = [T4c22GeometricDataset(root=BASEDIR, city=city, split="train", cachedir=Path(path), edge_attributes=edge_features, compress = compress) for city in cities]
    ld = [len(x) for x in datasets]


    for data_i, data_len in enumerate(ld):
        for i in arg:
                if i >= data_len:
                    continue
                data = datasets[data_i].get(i)
    return arg


from multiprocessing import Pool


with Pool(32) as pool:
    lists = np.array_split(range(8000), 100)
    pool.map(task, lists)

    
