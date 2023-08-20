import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PairNorm
from torch_geometric.utils import dropout_edge
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


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
    

class Net2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_pair_norm = False):
        super(Net2, self).__init__()
        self.conv1 = GCNLayer(in_channels, 64)
        # self.conv3 = GCNLayer(64,64)
        self.conv2 = GCNLayer(64, out_channels)
        self.use_pair_norm = use_pair_norm

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
  
        # x = F.sigmoid(x)
        # x = F.relu(self.conv3(x, edge_index))
        
        x = self.conv2(x, edge_index)
        if self.use_pair_norm:
          self.PairNorm(x)
        return x
    #节点编码成out_channels向量

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

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # [2,E]
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # *：element-wise乘法

    def decode_all(self, z):
        prob_adj = z @ z.t()  # @：矩阵乘法，自动执行适合的矩阵乘法函数
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, pos_edge_index, neg_edge_index):
        return self.decode(self.encode(x, pos_edge_index), pos_edge_index, neg_edge_index)
    
#edge split
dataset = Planetoid(root='cora', name='Cora', pre_transform=NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)

def get_link_labels(pos_edge_index,neg_edge_index):
    num_links=pos_edge_index.size(1)+neg_edge_index.size(1)
    link_labels=torch.zeros(num_links,dtype=torch.float)
    link_labels[:pos_edge_index.size(1)]=1
    return link_labels

def train(data,model,optimizer,criterion):
    model.train()
    
    neg_edge_index=negative_sampling( #训练集负采样
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    
    optimizer.zero_grad()
    z=model.encode(data.x,data.train_pos_edge_index)
    link_logits=model.decode(z,data.train_pos_edge_index,neg_edge_index)
    link_labels=get_link_labels(data.train_pos_edge_index,neg_edge_index).to(data.x.device)#训练集中正样本标签
    loss=criterion(link_logits,link_labels)
    loss.backward()
    optimizer.step()
    
    return loss

@torch.no_grad()
def mytest(data,model, plot = False):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()#计算链路存在的概率
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
        if plot == True:
          fpr, tpr, thresholds = roc_curve(link_labels, link_probs)
          plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(link_labels.cpu(), link_probs.cpu()))
          plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title('Receiver operating characteristic')
          plt.legend(loc="lower right")
          plt.show()

    return results

#训练过程
device = torch.device('cpu')
model = Net2(dataset.num_features, 64, True).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0007)
criterion = F.binary_cross_entropy_with_logits
best_val_auc = test_auc = 0
for epoch in range(1,500):
  loss=train(data,model,optimizer,criterion)
  if epoch == 499:
    val_auc,tmp_test_auc=mytest(data,model,plot=True)
  val_auc,tmp_test_auc=mytest(data,model)
  if val_auc>best_val_auc:
      best_val_auc=val_auc
      test_auc=tmp_test_auc
  print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')
#预测
z=model.encode(data.x,data.train_pos_edge_index)
final_edge_index=model.decode_all(z)

