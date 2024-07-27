#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-24 0:26
@Author  : lxc
@File    : 边的构建.py
@Desc    :
对于均匀全连接的无向图，每个节点都与其他所有节点相连，而且边是无序的，因此每条边只需定义一次。对于 10 个节点的情况，共有 \( \frac{{10 \times (10 - 1)}}{2} = 45 \) 条边。
10*(10-1)/2=45
这里使用列表解析生成所有节点对，并且只生成节点值较小的那一半边，然后将其转换为 PyTorch 张量，并且使用 `.t()` 进行转置操作以符合 torch_geometric 的要求。
"""
import torch

num_nodes = 10
edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(i + 1, num_nodes)], dtype=torch.long).t()

print(edge_index)