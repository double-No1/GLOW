# Raising the Bar in Graph-level Anomaly Detection (GLAD)
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch.nn as nn
import torch
from Code.gin_model.GraphNets import GIN,GraphNorm,GIN_classifier,GINConv
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool


class OCGIN(nn.Module):
    def __init__(self, dim_features, config):
        super(OCGIN, self).__init__()
        self.dropout = nn.Dropout(p=0.5)  # 添加dropout层
        self.dim_targets = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.net = GIN(dim_features, self.dim_targets, config)
        self.center = torch.zeros(1, self.dim_targets * self.num_layers, requires_grad=False).to('cuda')
        self.reset_parameters()
    def forward(self, data):
        data = data.to(self.device)
        # print(data)
        # print(data.edge_index)
        z = self.net(data)
        # print("z", z)
        return z, self.center

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cuda')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()

