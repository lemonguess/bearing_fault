import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,SAGEConv,GATConv
import torch.nn.functional as F
class LSTM_GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        super(LSTM_GCN, self).__init__()

        self.Conv1 = SAGEConv(input_dim,input_dim)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.Conv2 = GCNConv(input_dim, input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)
        # Define LSTM layer
        self.lstm = nn.LSTM(input_dim, input_dim, num_layers=3, batch_first=True)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)

        self.linear = nn.Sequential(
            nn.Linear(input_dim,int(input_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim/2),output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.f1 = x
        # GCN layer
        x = self.Conv2(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # x = self.Conv2(x, edge_index)
        # # x = x.view(-1, self.input_dim)  # 可能需要根据你的具体情况调整形状
        # x = self.bn2(x)
        # x = F.relu(x)


        self.f2 = x
        # 假设批处理大小是整个数据集的大小，这里需要根据你的数据调整
        batch_size = x.size(0)

        # 假设每个节点的特征代表一个时间步
        seq_length = 1  # 由于数据不是自然的序列数据，这里的序列长度设为1

        # 重新塑形以匹配LSTM的期望输入 [batch_size, seq_length, input_dim]
        x = x.view(batch_size, seq_length, -1)

        # LSTM layer
        x, _ = self.lstm(x)

        # 调整x的尺寸以匹配后续层的期望输入
        x = x.view(batch_size * seq_length, -1)
        x = self.bn2(x)
        x = F.relu(x)
        self.f3 = x
        x = self.linear(x)
        x = F.dropout(x, p=0.5, training=self.training)
        self.f4 = x

        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]
