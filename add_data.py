#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-17 21:25
@Author  : lxc
@File    : test.py
@Desc    :

"""
from apps.database_model import *

# UsersInfo.create(user_name="admin",
#                  password="admin")

DataSetInfo.create(**{"dataset_name" : "凯斯西储大学数据", "dataset_path" : ""})
