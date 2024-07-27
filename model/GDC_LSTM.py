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
