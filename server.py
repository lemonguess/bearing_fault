#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-17 1:23
@Author  : lxc
@File    : server.py
@Desc    :

"""
import json

from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
import argparse
import os
import uuid
import uvicorn
import traceback
import logging
import zipfile
from multiprocessing import Process
from utils.log_util import log_config
from apps.database_model import *
from utils.train_utils import train_utils

logger = logging.getLogger()
# 初始化 FastAPI 实例
app = FastAPI()

# 解决跨域问题
origins = ["*"]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["PUT", "GET", "POST", "DELETE"],
#     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "Referer", "User-Agent"],
# )

# 表单提交相关校验
class FileForm(BaseModel):
    file: bytes


# 接口鉴权
@app.middleware("http")
async def add_custom_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get(
        "Origin") or "http://127.0.0.1:5000" or "http://localhost:8080/"
    response.headers["Access-Control-Allow-Methods"] = "PUT, GET, POST, DELETE"
    response.headers[
        "Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, Referer, User-Agent"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# 测试服务器连通
@app.get("/")
def hello_world():
    print(111)
    return "Hello World!"


# 测试前后端可通信
@app.get("/getMsg")
@app.post("/getMsg")
def home():
    response = {"msg": "Hello, Python !"}
    return response


# 获取注册请求及处理
@app.post("/register")
def get_register_request(data: dict):
    try:
        # 连接数据库
        db = pymysql.connect(
            user="root",
            password="root",
            port=3306,
            host="localhost",
            database="geerwheel",
        )

        print("连接成功")
        cursor = db.cursor()

        username = data.get("username")
        password = data.get("password")
        password2 = data.get("password2")
        truename = data.get("truename")
        idcardnum = data.get("idcardnum")

        if password == password2:
            sql_0 = "INSERT INTO users(username, password,truename,idcardnum) VALUES (%s,%s,%s,%s)"
            sql = sql_0 % (
                repr(username),
                repr(password),
                repr(truename),
                repr(idcardnum),
            )

            cursor.execute(sql)
            db.commit()
            return "注册成功"
        else:
            return "注册失败"
    except Exception as e:
        traceback.print_exc()
        db.rollback()
        return "注册失败"
    finally:
        db.close()


# 获取登录参数及处理
@app.post("/login")
def get_login_request(data: dict):
    try:
        username = data.get("username")
        password = data.get("password")
        user_info = UsersInfo.select().where(UsersInfo.user_name == username).get()
        if user_info.password == password:
            return {'errcode': '200', 'errmsg': '密码正确', 'data': ''}
        else:
            return {'errcode': '400', 'errmsg': '用户名或密码不正确', 'data': ''}
    except IndexError:
        return {'errcode': '400', 'errmsg': '用户名错误，无当前用户信息', 'data': ''}
    except Exception as e:
        logger.exception(e)
        return {'errcode': '400', 'errmsg': e, 'data': ''}


# 展示数据集
@app.get("/showDataset")
def show_dataset():
    try:
        rows = DataSetInfo.select().dicts()
        data_set_info = [row for row in rows]
        return {'errcode': '200', 'errmsg': '成功', 'data': data_set_info}
    except Exception as e:
        logger.exception(e)
        return {'errcode': '400', 'errmsg': e, 'data': ''}


@app.get("/showModel")
def show_model():
    try:
        rows = ModelInfo.select().dicts()
        model_info = [row for row in rows]
        return {'errcode': '200', 'errmsg': '成功', 'data': model_info}
    except Exception as e:
        logger.exception(e)
        return {'errcode': '400', 'errmsg': e, 'data': ''}


@app.get("/showOptimizer")
def show_optimizer():
    try:
        rows = OptimizerInfo.select().dicts()
        optimizer_info = [row for row in rows]
        return {'errcode': '200', 'errmsg': '成功', 'data': optimizer_info}
    except Exception as e:
        logger.exception(e)
        return {'errcode': '400', 'errmsg': e, 'data': ''}


@app.get("/showTask")
def show_task():
    try:
        task_info = []
        rows = OptimizerInfo.select().dicts()
        for row in rows:
            item = {}
            item["task_name"] = row.get("task_name")
            item["user_name"] = UsersInfo.select().where(UsersInfo.user_id == row.get("user_id")).first().get(
                "user_name")
            item["dataset_name"] = DataSetInfo.select().where(
                DataSetInfo.dataset_id == row.get("dataset_id")).first().get("dataset_name")
            item["model_name"] = ModelInfo.select().where(ModelInfo.model_id == row.get("model_id")).first().get(
                "model_name")
            item["optimizer_name"] = OptimizerInfo.select().where(
                OptimizerInfo.optimizer_id == row.get("optimizer_id")).first().get("optimizer_name")
            item["config_info"] = json.dumps(json.loads(row.get("task_name")), ensure_ascii=False, indent=4)
            item["result_info"] = row.get("result_info")
            task_info.append(item)
        return {'errcode': '200', 'errmsg': '成功', 'data': task_info}
    except Exception as e:
        logger.exception(e)
        return {'errcode': '400', 'errmsg': e, 'data': ''}


# 添加数据集
@app.post("/uploadDataset")
def upload_dataset(data: dict, file: UploadFile = File(...)):
    try:
        dataset_path = os.path.join(os.getcwd(), "static", "datasets", uuid.uuid4().hex)
        os.makedirs(dataset_path, exist_ok=True)
        # 检查文件后缀是否为.zip
        if file.filename.lower().endswith('.zip'):
            # 指定解压的目标路径
            target_dir = dataset_path
            # 确保目标路径存在，如果不存在则创建
            os.makedirs(target_dir, exist_ok=True)
            try:
                # 使用with语句打开文件流，并解压到指定目录
                with zipfile.ZipFile(file.file, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
            except Exception as e:
                # 如果发生其他异常，抛出异常
                return {'errcode': '400', 'errmsg': "解压文件出错：\n%s " % e, 'data': ''}
        else:
            # 如果文件不是ZIP后缀，抛出异常
            return {'errcode': '400', 'errmsg': "请上传zip格式的数据集！", 'data': ''}
        # 数据表插入
        dataset_name = data.get("dataset_name")
        dataset_intro = data.get("dataset_intro")
        DataSetInfo.create(dataset_name=dataset_name, dataset_intro=dataset_intro, dataset_path=dataset_path)
        return {'errcode': '200', 'errmsg': "上传成功", 'data': ''}
    except Exception as e:
        return {'errcode': '400', 'errmsg': "上传数据集出错！\n%s" % e, 'data': ''}


# 调用训练函数
@app.post("/train")
def train(data: dict):
    try:
        # 数据集选择
        dataset_id = int(data.get("dataset_id"))
        dataset = DataSetInfo.select().where(DataSetInfo.dataset_id == dataset_id).get()
        dataset_name = dataset.dataset_name
        dataset_path = dataset.dataset_path
        # 模型选择
        model_id = data.get("model_id")
        model = ModelInfo.select().where(ModelInfo.model_id == model_id).get()
        model_type = model.model_name
        # 参数选择
        ...
        # 准备训练c
        parser = argparse.ArgumentParser()
        # basic parameters
        # ===================================================dataset parameters=============================================================================
        parser.add_argument('--dataset_name', type=str, default='CWRU',
                            help='the name of the dataset：CWRU、SEU、XJTU、JNU、MFPT、UoC、DC')
        parser.add_argument('--dataset_path', type=str, default=r"E:\故障诊断数据集\凯斯西储大学数据",
                            help='the file path of the dataset')
        parser.add_argument('--dir_path', type=str, default='12DE',
                            help='the sample frequency of CWRU：12DE、48DE represent 12kHZ and 48kHZ respectively')
        parser.add_argument('--SEU_channel', type=int, default=1, help='the channel number of SEU：0-7')
        parser.add_argument('--minute_value', type=int, default=10,
                            help='the last (minute_value) csv file of XJTU datasets each fault class')
        parser.add_argument('--XJTU_channel', type=str, default='X', help='XJTU channel signal:X 、Y 、XY')

        # ===================================================data preprocessing parameters=============================================================================
        parser.add_argument('--sample_num', type=int, default=50, help='the number of samples')
        parser.add_argument('--train_size', type=float, default=0.6, help='train size')
        parser.add_argument('--sample_length', type=int, default=1024, help='the length of each samples')
        parser.add_argument('--overlap', type=int, default=1024, help='the sampling shift of neibor two samples')
        parser.add_argument('--norm_type', type=str, default='unnormalization', help='the normlized method')
        parser.add_argument('--noise', type=int, default=0, help='whether add noise')
        parser.add_argument('--snr', type=int, default=0, help='the snr of noise')

        parser.add_argument('--input_type', type=str, default='FD',
                            help='TD——time domain signal，FD——frequency domain signal')
        parser.add_argument('--graph_type', type=str, default='path_graph', help='the type of graph')
        parser.add_argument('--knn_K', type=int, default=5, help='the K value of knn-graph')
        parser.add_argument('--ER_p', type=float, default=0.5, help='the p value of ER-graph')
        parser.add_argument('--node_num', type=int, default=10, help='the number of node in a graph')
        parser.add_argument('--direction', type=str, default='undirected', help='directed、undirected')
        parser.add_argument('--edge_type', type=str, default='0-1', help='the edge weight method of graph')
        parser.add_argument('--edge_norm', type=bool, default=False, help='whether normalize edge weight')
        parser.add_argument('--batch_size', type=int, default=64)

        # ===================================================model parameters=============================================================================
        parser.add_argument('--model_type', type=str, default='GCN', help='the model of training and testing')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--momentum', type=float, default=1e-5)
        parser.add_argument('--optimizer', type=str, default='Adam')

        # ===================================================visualization parameters=============================================================================
        parser.add_argument('--visualization', type=bool, default=True, help='whether visualize')
        # 输出路径
        output_path = os.path.join(os.getcwd(), "output", uuid.uuid4().hex)
        os.makedirs(output_path, exist_ok=True)
        parser.add_argument('--output_path', type=bool, default=output_path, help='output_path')
        # 执行任务
        args = parser.parse_args()
        main_p = Process(target=train_utils(args))
        main_p.start()

        return {'errcode': 200, 'errmsg': "创建任务成功", 'data': ''}
    except Exception as e:
        traceback.print_exc()
        return {'errcode': '400', 'errmsg': "创建任务失败！" % e, 'data': ''}


# 查看训练日志
@app.post("/result")
def read_file(model: str):
    try:
        file_path = "E:/DiagnosisSystem/BackEnd/Algorithm/BackEnd/Algorithm/save_logs/{}.log".format(model)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return {
            "content": content,
            "status": "success"
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "error"
        }


if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=5001,
                log_config=log_config, log_level='debug')
