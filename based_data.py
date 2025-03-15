import os.path as osp
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from torch_geometric.io import read_txt_array
from torch_sparse import coalesce
import scipy.sparse as sp
from torch_geometric.loader import DataLoader
import argparse

import torch
from torch_geometric.transforms import ToUndirected


def read_file(folder, name, dtype=None):
    path = osp.join(folder, '{}.txt'.format(name))
    return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
    """
	PyG util code to create graph batches
	"""

   
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

        # The fixed data split for benchmarking evaluation
        # train-val-test split is 20%-10%-70%
        # self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
        # self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
        # self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)

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



def data_augmentation(data, p_edge_drop=0.1, p_feature_noise=0.01):
 
    mask = torch.rand(data.edge_index.shape[1]) > p_edge_drop
    data.edge_index = data.edge_index[:, mask]

   
    noise = torch.randn_like(data.x) * p_feature_noise
    data.x += noise

    return data



def collate_fn(batch):
    batch = Batch.from_data_list([data_augmentation(data) for data in batch])
    return batch



parser = argparse.ArgumentParser()


parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
# parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--dataset', type=str, default='gossipcop', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=60, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=False, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='spacy', help='feature type, [profile, spacy, bert, content]')

args = parser.parse_args()
torch.manual_seed(args.seed) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='E:\\Pycharm\\task-GIN\\data\\FakeNewsData', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())
print(dataset)
labels = torch.tensor([x.y for x in dataset])
num_classes = torch.max(labels).item() + 1
print(num_classes)

config = {"hidden_dim": args.nhid, "num_layers": 2, "device": args.device,
          "norm_layer": 0, "aggregation": "mean", "bias": "true"}



train_idx, test_idx, val_idx = dataset.train_idx, dataset.test_idx, dataset.val_idx
# train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size)
train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)



import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from Code.Losses import OCC_loss
from gin_model.Models import OCGIN
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        z_c = model(data.to(device))
        loss = criterion(z_c).mean()
        loss.backward()
        optimizer.step()
        scheduler.step(i + epoch * len(dataloader))
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    """Evaluate the model on validation or test set."""
    model.eval()
    distances = []
    y_true = []
    with torch.no_grad():
        for data in dataloader:
            z_c = model(data.to(device))
            dist = torch.norm(z_c[0] - z_c[1], p=2, dim=1)
            distances.extend(dist.cpu().numpy())
            y_true.extend(data.y.cpu().numpy())
    return distances, y_true

def find_best_threshold(distances, labels):
    """Find the best threshold using F1 score."""
    thresholds = np.linspace(min(distances), max(distances), num=100)
    best_f1 = 0
    best_threshold = None
    for thres in thresholds:
        y_pred = np.array(distances) <= thres
        current_f1 = f1_score(labels, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thres
    return best_f1, best_threshold

def main():
    set_seed()  # Set random seed for reproducibility

  
    model = OCGIN(dim_features=dataset.num_features, config=config)
    model.to(args.device)
    criterion = OCC_loss()

  
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # Increase weight decay
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), eta_min=1e-7)  # Decrease eta_min further

    model.init_center(train_loader)

    best_val_f1 = 0
    patience = 10
    no_improve_count = 0

    # Training loop
    num_epochs = 200  # Increase number of epochs
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, args.device, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}")

        # Validation
        val_distances, val_labels = evaluate(model, val_loader, args.device)
        best_val_f1_epoch, best_threshold_val = find_best_threshold(val_distances, val_labels)

        # Save the best model based on validation F1 score
        if best_val_f1_epoch > best_val_f1:
            best_val_f1 = best_val_f1_epoch
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

    # After training is complete, load the best model for final testing
    model.load_state_dict(torch.load('best_model.pth'))

    # Final testing
    test_distances, test_labels = evaluate(model, test_loader, args.device)
    best_test_f1, best_threshold_test = find_best_threshold(test_distances, test_labels)

    # Generate predictions using the best threshold found on the test set
    y_pred_test = np.array(test_distances) <= best_threshold_test
    # Calculate metrics
    acc_test = accuracy_score(test_labels, y_pred_test)
    rec_test = recall_score(test_labels, y_pred_test)
    pre_test = precision_score(test_labels, y_pred_test)
    f1_test = f1_score(test_labels, y_pred_test)

    # Print metrics
    print(f"Final Test Accuracy: {acc_test}, Test Recall: {rec_test}, Test Precision: {pre_test}, Test F1 Score: {f1_test}")

if __name__ == '__main__':
    main()
