import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, TopKPooling,SAGEConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
class GCN(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(GCN, self).__init__()
        """
        在这个例子中，input_dim 代表的是每个节点的输入特征的维度。
        在图卷积网络（GCN）中，每个节点都有一个特征向量表示其属性或特征。
        input_dim 就是这个特征向量的维度大小，它反映了每个节点的输入特征的信息量或维度。
        在这个模型中，GCNConv 层的输入维度就是 input_dim。
        """
        self.input_dim = input_dim
        self.conv1 = GCNConv(input_dim,input_dim)
        self.pool1 = TopKPooling(input_dim, ratio=0.8)

        self.conv2 = SAGEConv(input_dim,input_dim)
        self.pool2 = TopKPooling(input_dim, ratio=0.8)

        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)
        self.linear = nn.Sequential(
            nn.Linear(input_dim,int(input_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim/2),output_dim)
        )
        # self.
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        # item_embedding = torch.nn.Embedding(x.shape[0], embedding_dim=self.input_dim)
        # x = item_embedding(x)# n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        # x = x.squeeze(1) # n*128

        self.f1 = x  # 原始特征(时域输入时为时域特征，频域输入时为频域特征)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        self.f2 = x  # 第一层特征
        x1 = gap(x, batch)



        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        self.f3 = x  # 第二层特征
        x2 = gap(x, batch)
        # x = x1+x2

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear(x)
        x = F.dropout(x, p=0.5, training=self.training)
        self.f4 = x  # 全连接层特征


        out = F.log_softmax(x,dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]