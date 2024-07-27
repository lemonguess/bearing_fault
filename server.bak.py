#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-17 1:23
@Author  : lxc
@File    : server.py
@Desc    :

"""
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware

from utils.log_util import log_config
import os
import uuid
import uvicorn
import traceback
import logging
from apps.database_model import *
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
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin") or "http://127.0.0.1:5000" or "http://localhost:8080/"
    response.headers["Access-Control-Allow-Methods"] = "PUT, GET, POST, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, Origin, Referer, User-Agent"
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
        db = pymysql.connect(
            host="localhost", user="root", password="root", database="geerwheel", charset="utf8"
        )
        cursor = db.cursor()

        username = data.get("username")
        password = data.get("password")
        sql_0 = "select * from users where username= %s and password= %s"
        sql = sql_0 % (repr(username), repr(password))

        cursor.execute(sql)
        results = cursor.fetchall()

        if len(results) == 1:
            return "登录成功"
        else:
            return "用户名或密码不正确"
        db.commit()
    except Exception as e:
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

# 展示数据集
@app.post("/showDataset")
def show_dataset():
    try:
        db = pymysql.connect(
            host="localhost", user="root", password="root", database="geerwheel", charset="utf8"
        )
        cursor = db.cursor()
        sql = "select * from dataset"

        cursor.execute(sql)
        results = cursor.fetchall()

        dataset_list = []
        for row in results:
            dic = {
                "id": row[0],
                "name": row[1],
                "region": row[2],
                "contact": row[3],
                "description": row[4],
                "ischoosed": row[5],
            }
            dataset_list.append(dic)

        return dataset_list
    except Exception as e:
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

# 更改数据集选中状态
@app.post("/updataDataset")
def update_dataset(status: int):
    try:
        db = pymysql.connect(
            host="localhost", user="root", password="root", database="geerwheel", charset="utf8"
        )
        cursor = db.cursor()

        if status == 1:
            sql = "UPDATE dataset SET ischoosed='否' WHERE id=2"
        else:
            sql = "UPDATE dataset SET ischoosed='是' WHERE id=2"

        cursor.execute(sql)
        db.commit()
        return "更改成功"
    except Exception as e:
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

# 添加数据集
@app.post("/uploadDataset")
def upload_dataset(data: dict):
    try:
        db = pymysql.connect(
            host="localhost", user="root", password="root", database="geerwheel", charset="utf8"
        )
        cursor = db.cursor()

        id = data.get("id")
        name = data.get("name")
        region = data.get("region")
        contact = data.get("contact")
        description = data.get("description")
        ischoosed = data.get("ischoosed")

        sql_0 = "INSERT INTO dataset(id,name,region,contact,description,ischoosed) VALUES (%s,%s,%s,%s,%s,%s)"
        sql = sql_0 % (
            repr(id),
            repr(name),
            repr(region),
            repr(contact),
            repr(description),
            repr(ischoosed),
        )

        cursor.execute(sql)
        db.commit()
        return "添加成功"
    except Exception as e:
        traceback.print_exc()
        db.rollback()
        return "添加失败"
    finally:
        db.close()

# 上传文件
@app.post("/uploadDatafile")
def upload_datafile(file: UploadFile = File(...)):
    try:
        file_contents = file.file.read()

        # 保存文件文件中
        with open(os.path.join(save_path, file.filename), "wb") as f:
            f.write(file_contents)

        return {
            "code": 200,
            "message": "上传请求成功",
            "fileName": file.filename
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "code": 400,
            "message": "上传失败"
        }

# 创建数据集文件夹
@app.post("/makeDatadir")
def make_datadir(name: str):
    try:
        dir = os.path.dirname(__file__)
        if not os.path.exists(f"{dir}/static/data/{name}"):
            os.mkdir(f"{dir}/static/data/{name}")
            global save_path
            save_path = f"{dir}/static/data/{name}"
            return {
                "code": 200,
                "message": save_path
            }
        else:
            return {
                "code": 400,
                "message": "路径已存在"
            }
    except Exception as e:
        traceback.print_exc()
        return {
            "code": 400,
            "message": "创建失败"
        }

# 调用训练函数
@app.post("/train")
def train(data: dict):
    try:
        model = data.get("model")
        if model == "CNN":
            from Algorithm.BackEnd.Algorithm.sign_cnn import run_Algorithm
        elif model == "INCEPTION":
            from Algorithm.BackEnd.Algorithm.sign_inception import run_Algorithm

        num_classes = data.get("num_classes")
        epochs = data.get("epochs")
        train = data.get("train")
        valid = data.get("valid")
        test = data.get("test")

        run_Algorithm(num_classes=num_classes, epochs=epochs, train=train, valid=valid, test=test)

        return {
            "code": 200,
            "message": "成功"
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "code": 400,
            "message": "失败"
        }

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