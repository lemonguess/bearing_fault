#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-17 1:36
@Author  : lxc
@File    : database_model.py
@Desc    :
1.用户信息表——users_info
2.数据集信息表——dataset_info
3.模型信息表——models_info
4.任务信息表——task_info
5.结果信息表——result_info
"""
from peewee import *
from datetime import datetime
from peewee import SqliteDatabase
database = SqliteDatabase('apps/db.db')
class UnknownField(object):
    def __init__(self, *_, **__): pass

class BaseModel(Model):
    class Meta:
        database = database

class UsersInfo(BaseModel):
    """用户信息表"""
    user_id = IntegerField(primary_key=True, verbose_name="用户id")
    user_name = TextField(null=True, verbose_name='用户名称')
    password = TextField(null=True, verbose_name='用户密码')
    created_time = DateTimeField(default=datetime.now, verbose_name="添加时间")
    update_time = DateTimeField(verbose_name="更新时间", default=datetime.now)

    class Meta:
        table_name = 'UsersInfo'

class DataSetInfo(BaseModel):
    """数据集信息表"""
    dataset_id = IntegerField(primary_key=True, verbose_name="数据集id")
    dataset_name = TextField(null=True, verbose_name='数据集名称')
    dataset_intro = TextField(null=True, verbose_name='数据集简介')
    dataset_path = TextField(null=True, verbose_name='数据集路径')
    created_time = DateTimeField(default=datetime.now, verbose_name="添加时间")
    update_time = DateTimeField(verbose_name="更新时间", default=datetime.now)

    class Meta:
        table_name = 'DataSetInfo'

class ModelInfo(BaseModel):
    """模型信息表"""
    model_id = IntegerField(primary_key=True, verbose_name="模型id")
    model_name = TextField(null=True, verbose_name='模型名称')
    model_intro = TextField(null=True, verbose_name='模型简介')
    created_time = DateTimeField(default=datetime.now, verbose_name="添加时间")
    update_time = DateTimeField(verbose_name="更新时间", default=datetime.now)

    class Meta:
        table_name = 'ModelInfo'

class OptimizerInfo(BaseModel):
    """优化器信息表"""
    optimizer_id = IntegerField(primary_key=True, verbose_name="优化器id")
    optimizer_name = TextField(null=True, verbose_name='优化器名称')
    optimizer_intro = TextField(null=True, verbose_name='优化器简介')
    created_time = DateTimeField(default=datetime.now, verbose_name="添加时间")
    update_time = DateTimeField(verbose_name="更新时间", default=datetime.now)

    class Meta:
        table_name = 'OptimizerInfo'

class TaskInfo(BaseModel):
    """任务信息表"""
    task_id = IntegerField(primary_key=True, verbose_name="任务id")
    task_name = TextField(null=True, verbose_name='任务名称')
    user_id = IntegerField(null=True, verbose_name='用户id')
    dataset_id = IntegerField(null=True, verbose_name='数据集id')
    model_id = IntegerField(null=True, verbose_name='模型id')
    optimizer_id = IntegerField(null=True, verbose_name='优化器id')
    config_info = TextField(null=True, verbose_name='配置信息')
    result_info = TextField(null=True, verbose_name='结果信息')
    created_time = DateTimeField(default=datetime.now, verbose_name="添加时间")
    update_time = DateTimeField(verbose_name="更新时间", default=datetime.now)

    class Meta:
        table_name = 'TaskInfo'

class ResultInfo(BaseModel):
    """结果信息表"""
    result_id = IntegerField(primary_key=True, verbose_name="结果id")
    task_id = IntegerField(null=True, verbose_name='任务id')
    is_succeed = IntegerField(null=True, verbose_name='是否成功')
    result_info = TextField(null=True, verbose_name='结果信息')
    created_time = DateTimeField(default=datetime.now, verbose_name="添加时间")
    update_time = DateTimeField(verbose_name="更新时间", default=datetime.now)

    class Meta:
        table_name = 'ResultInfo'

if __name__ == '__main__':
    # 删除表
    # FabaoLawsFile.drop_table()
    # # 创建表
    # FabaoLawsFile.create_table()
    # 插入数据
    # FabaoLawsFile.create(major_audit_items=level1_mulu,
    #                         audit_item=level2_mulu,
    #                         audit_issues=level3_mulu,
    #                         dxyj=dxyj,
    #                         cfyj=cfyj,
    #                         xgyj=xgyj)
    UsersInfo.create_table()
    DataSetInfo.create_table()
    ModelInfo.create_table()
    OptimizerInfo.create_table()
    TaskInfo.create_table()
    ResultInfo.create_table()