#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-04-12 23:30
@Author  : lxc
@File    : DCRNN.py
@Desc    :

"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, SAGEConv,GATConv
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from einops import rearrange
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class GraphDiffusionConv(MessagePassing):
    def __init__(self, in_channels):
        super(GraphDiffusionConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, in_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Transform node feature matrix.
        x = self.lin(x)

        # Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # If desired, skip the `update` step and simply return `aggr_out`.
        return aggr_out

class LSTM_GDC(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTM_GDC, self).__init__()
        self.input_dim = input_dim

        # GDC layer
        self.Conv1 = GraphDiffusionConv(input_dim)

        # SAGEConv layer
        # self.Conv2 = GraphDiffusionConv(input_dim)
        self.Conv2 = GATConv(input_dim,input_dim)

        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, input_dim, num_layers=4, batch_first=True)

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim / 2), output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.f1 = x
        print("x.shape",x.shape)
        print("edge_index", edge_index.shape)
        # GDC layer
        x = self.Conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # GDC layer
        x = self.Conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        self.f2 = x
        # LSTM layer
        batch_size = x.size(0)
        seq_length = 1
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = x.view(batch_size * seq_length, -1)
        self.f3 = x
        # Linear layers
        x = self.linear(x)
        x = F.dropout(x, p=0.5, training=self.training)
        self.f4 = x
        out = F.log_softmax(x, dim=1)

        return out
    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]




class RFCA_graph(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, reduction=32):
        super(RFCA_graph, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)

class RFCA(torch.nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super(RFCA, self).__init__()
        self.inp = inp
        self.oup = oup
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction = reduction

        self.rfca_conv = RFCA_graph(inp, oup, kernel_size, stride, reduction)

    def forward(self, x, edge_index):
        # 增加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        num_nodes = x.size(0)
        x = x.view(num_nodes, self.inp, 1, 1)
        x = self.rfca_conv(x)
        x = x.view(num_nodes, self.oup)
        return x

class LSTM_RFCA(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTM_RFCA, self).__init__()
        self.input_dim = input_dim
        # GDC layer
        self.Conv1 = GraphDiffusionConv(input_dim)

        # GAT layer
        # self.Conv2 = GraphDiffusionConv(input_dim)
        self.Conv2 = GATConv(input_dim,input_dim)

        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)
        # 使用 RFCA 层

        self.Conv3 = RFCA(inp=input_dim, oup=input_dim, kernel_size=3, stride=1)
        # self.Conv2 = RFCA(inp=input_dim, oup=input_dim, kernel_size=3, stride=1)

        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.bn2 = torch.nn.BatchNorm1d(input_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, input_dim, num_layers=4, batch_first=True)

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim / 2), output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.f1=x
        # GCN layer
        x = self.Conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.Conv3(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        self.f2 = x
        # GAT layer
        x = self.Conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        self.f2 = x
        # RFA
        x = self.Conv3(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        self.f2 = x
        # LSTM layer
        batch_size = x.size(0)
        seq_length = 1
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = x.view(batch_size * seq_length, -1)
        self.f3 = x

        # Linear layers
        x = self.linear(x)
        x = F.dropout(x, p=0.5, training=self.training)
        self.f4 = x
        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1, self.f2, self.f3, self.f4]  # 返回模型中间层的特征，便于分析和调试
