import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGEConv,GCNConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim,num_heads=4):
        self.input_dim = input_dim
        self.num_heads = num_heads
        super(GAT, self).__init__()

        self.Conv1 = SAGEConv(input_dim,input_dim)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)

        self.Conv2 = GCNConv(input_dim, input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)

        self.Conv3 = GATConv(input_dim,input_dim)
        self.bn3 = torch.nn.BatchNorm1d(input_dim)

        self.linear = nn.Sequential(
            nn.Linear(input_dim,int(input_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim/2),output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        self.f1 = x  # 原始特征(时域输入时为时域特征，频域输入时为频域特征)

        x = self.Conv1(x, edge_index)
        x = x.view(-1, self.input_dim) # 可能需要根据你的具体情况调整形状
        x = self.bn1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        self.f2 = x  # 第一层特征

        x = self.Conv2(x, edge_index)
        x = x.view(-1, self.input_dim)  # 可能需要根据你的具体情况调整形状
        x = self.bn2(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        self.f3 = x  # 第二层特征

        x = self.Conv3(x, edge_index)
        x = x.view(-1, self.input_dim)  # 可能需要根据你的具体情况调整形状
        x = self.bn3(x)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        self.f4 = x  # 第三层特征

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        self.f5 = x  # 全连接层特征

        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4,self.f5]

