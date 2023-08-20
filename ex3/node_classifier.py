import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import PairNorm
from torch_geometric.utils import dropout_edge
import torch.optim as optim
from torch_geometric.datasets import Planetoid


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        

    
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.linear(x)
        sp = torch.sparse.FloatTensor(edge_index, norm)
        x = torch.matmul(sp, x)
        return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_pair_norm=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.use_pair_norm = use_pair_norm

    def PairNorm(self,x_feature):
        mode = 'PN-SI'
        scale = 1
        col_mean = x_feature.mean(dim=0)
        if mode == 'PN':
            x_feature = x_feature - col_mean
            row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
            x_feature = scale * x_feature / row_norm_mean

        if mode == 'PN-SI':
            x_feature = x_feature - col_mean
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual

        if mode == 'PN-SCS':
            row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x_feature = scale * x_feature / row_norm_individual - col_mean

        return x_feature
    
    def forward(self, x, edge_index):
      # edge_index,_=dropout_edge(edge_index,p=0.4)
        x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x = self.conv2(x, edge_index)
        if self.use_pair_norm:
            x = self.PairNorm(x)
        x = F.log_softmax(x,dim=1)
        
        return x
    
#数据加载及划分
dataset = Planetoid(root='cora', name='Cora')
data = dataset[0]
#self - loop
# from torch_geometric.utils import add_self_loops
# new_edge,_=add_self_loops(data.edge_index)
# data.edge_index = new_edge
# print(data.has_self_loops())

#预设值
model = GCN(data.num_features, 32, 7, use_pair_norm=False)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = F.nll_loss

def train_node_classification(model = model, optimizer = optimizer, criterion = criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def val_node_classification(model = model, optimizer = optimizer, criterion = criterion):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss.item()

#训练
epoch_num = 100
train_losses = []
val_losses = []
for epoch in range(epoch_num):
    train_loss = train_node_classification()
    out = model(data.x, data.edge_index)

    val_loss = val_node_classification()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}: Train loss = {train_loss:.4f}: Val loss = {val_loss:.4f}")

#训练过程可视化
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(epoch_num), train_losses)
plt.legend("train loss")
plt.plot(np.arange(epoch_num), val_losses)
plt.legend("val loss")
plt.show()

#预测
model.eval()
pred = model(data.x,data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
acc = int(correct) / int(data.test_mask.sum())
print(acc)
