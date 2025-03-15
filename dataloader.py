import os.path as osp
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree
from torch_geometric.io import read_txt_array
from torch_sparse import coalesce
import scipy.sparse as sp
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data.dataset import ConcatDataset
from torch_geometric.transforms import ToUndirected
import torch
import pandas as pd


def read_file(folder, name, dtype=None):
    path = osp.join(folder, '{}.txt'.format(name))
    return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
  

   
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice]) 
    row, _ = data.edge_index    
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

   
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()   

    slices = {'edge_index': edge_slice}  
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def read_graph_data(folder, feature):
    """
	PyG util code to create PyG data instance from raw graph data
	"""

    node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')    
    edge_index = read_file(folder, 'A', torch.long).t()    
  
    node_graph_id = np.load(folder + 'node_graph_id.npy')
    graph_labels = np.load(folder + 'graph_labels.npy')
    
    edge_attr = None
    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)       

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)    
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr)  
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)  

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, node_graph_id)   
    return data, slices


class FNNDataset(InMemoryDataset):

  
    def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.feature = feature
        super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        names = ['node_graph_id', 'graph_labels']
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

  
    def process(self):

        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)


        # Get labels
        labels = self.data.y.numpy()

        # Split data based on label
        train_indices = np.where(labels == 1)[0]  
        test_indices = np.where(labels >= 0)[0]  

        # Convert to PyTorch tensors
        self.train_idx = torch.from_numpy(train_indices).to(torch.long)
        self.test_idx = torch.from_numpy(test_indices).to(torch.long)
        self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)

        torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast


class CombinedDataset(InMemoryDataset):
    def __init__(self, root, name, feature='spacy', texts=None, labels=None, empty=False, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.feature = feature
        self.texts = texts
        self.labels = labels
        super(CombinedDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])


    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        names = ['node_graph_id', 'graph_labels']
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):
        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # Get labels
        labels = self.data.y.numpy()

        # Split data based on label
        train_indices = np.where(labels == 1)[0]  # Assuming label 1 for training (real news)
        test_indices = np.where(labels >= 0)[0]   # All news

        # Convert to PyTorch tensors
        self.train_idx = torch.from_numpy(train_indices).to(torch.long)
        self.test_idx = torch.from_numpy(test_indices).to(torch.long)
        # self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)

        # Save graph data
        torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

    def __getitem__(self, idx):
        graph_data = super().__getitem__(idx)
        text = self.texts[idx]
        label = self.labels[idx]
        return graph_data, text, label
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))







def collate_fn1(batch):

    texts, labels = zip(*batch)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=128)
    print(inputs)

    labels = torch.tensor(labels)
    print("Collate function output:", {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']})

    return inputs['input_ids'],  inputs['attention_mask'], labels

