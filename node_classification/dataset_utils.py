import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikipediaNetwork, HeterophilousGraphDataset, WebKB, Actor, CitationFull, DBLP, NELL
import torch
from torch_geometric.data import Data
import numpy as np

def DataLoader(name):
    name = name.lower()
    
    # Standard PyTorch Geometric datasets
    if name in ['cora', 'citeseer', 'pubmed']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Planetoid(path, name, split="geom-gcn", transform=None)
    elif name in ['cora1', 'citeseer1', 'pubmed', 'cora_ml', 'dblp']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = CitationFull(path, name, transform=None)
    elif name in ['cs', 'physics']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Coauthor(path, name, transform=None)
    elif name in ['computers', 'photo', 'physics']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, transform=None)
    elif name in ['chameleon', 'crocodile', 'squirrel']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = WikipediaNetwork(path, name, transform=None)
    elif name in ['cornell', 'texas', 'wisconsin']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = WebKB(path, name)
    elif name in ['film']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Actor(path)
    elif name in ['nell']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = NELL(path)

    # New heterophilous datasets
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = HeterophilousGraphDataset(path, name, transform=None)
        
    elif name in ["cora_full"]:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = CitationFull(path, name="Cora")
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    
    return dataset

def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(42)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data

