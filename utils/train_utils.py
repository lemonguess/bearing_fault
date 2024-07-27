# -*- coding: utf-8 -*-
import os
import uuid

import torch
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import CWRU_data,SEU_data,XJTU_data,JNU_data,MFPT_data,UoC_data,DC_data
from model.ChebyNet import ChebyNet
from model.GAT import GAT
from model.GIN import GIN
from model.GCN import GCN
from model.GraphSAGE import GraphSAGE
from model.MLP import MLP
from model.SGCN import SGCN
from model.CNN import CNN_1D
from model.LeNet import LeNet
from model.LSTM_GCN import LSTM_GCN
from model.GDC_LSTM import LSTM_GDC
from model.RFAconv import LSTM_RFCA
from utils.visualization_confusion import visualization_confusion
from utils.visualization_tsne import visualization_tsne
import logging
logger = logging.getLogger()
import matplotlib
from matplotlib import font_manager as fm, rcParams
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
plt.rcParams['font.family']=['Microsoft YaHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
def train_utils(args):

    #==============================================================1、训练集、测试集===================================================
    if args.dataset_name == 'CWRU':
        loader_train, loader_test, loader_val = CWRU_data.data_preprocessing(
            dataset_path=args.dataset_path,
            sample_number=args.sample_num,
            dir_path=args.dir_path,
            window_size=args.sample_length,
            overlap=args.overlap,
            normalization=args.norm_type,
            noise=args.noise,
            snr=args.snr,
            input_type=args.input_type,
            graph_type=args.graph_type,
            K=args.knn_K,
            p=args.ER_p,
            node_num=args.node_num,
            direction=args.direction,
            edge_type=args.edge_type,
            edge_norm=args.edge_norm,
            batch_size=args.batch_size,
            train_size=args.train_size
        )
        output_dim = 10 #10分类

    elif args.dataset_name == 'SEU':
        loader_train, loader_test, loader_val = SEU_data.data_preprocessing(dataset_path=args.dataset_path,channel=args.SEU_channel,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,
                                                                   normalization=args.norm_type, noise=args.noise,snr=args.snr,input_type=args.input_type,graph_type=args.graph_type, K=args.knn_K,
                                                                   p=args.ER_p,node_num=args.node_num, direction=args.direction,edge_type=args.edge_type,
                                                                   edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 10  # 10分类

    elif args.dataset_name == 'XJTU':
        loader_train, loader_test, loader_val = XJTU_data.data_preprocessing(dataset_path=args.dataset_path,channel=args.XJTU_channel,minute_value=args.minute_value, sample_number=args.sample_num,
                                                                window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type, noise=args.noise,snr=args.snr, input_type=args.input_type,
                                                                graph_type=args.graph_type, K=args.knn_K,p=args.ER_p,node_num=args.node_num, direction=args.direction,
                                                                edge_type=args.edge_type,edge_norm=args.edge_norm, batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 15  # 15分类

    elif args.dataset_name == 'JNU':
        loader_train, loader_test, loader_val = JNU_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length,overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise,snr=args.snr, input_type=args.input_type,graph_type=args.graph_type, K=args.knn_K,p=args.ER_p,
                                                                node_num=args.node_num,direction=args.direction,edge_type=args.edge_type,edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 12  # 12分类

    elif args.dataset_name == 'MFPT':
        loader_train, loader_test, loader_val = MFPT_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                                node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 15  # 15分类

    elif args.dataset_name == 'UoC':
        loader_train, loader_test, loader_val = UoC_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                                node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 9  # 9分类

    elif args.dataset_name == 'DC':
        loader_train, loader_test, loader_val = DC_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                               noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                               node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 10  # 10分类

    else:
        print('this dataset is not existed!!!')

    #==============================================================2、网络模型===================================================
    input_dim = loader_train.dataset[0].x.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'GCN':
        model = GCN(input_dim, output_dim).to(device)
    elif args.model_type == 'ChebyNet':
        model = ChebyNet(input_dim, output_dim).to(device)
    elif args.model_type == 'GAT':
        model = GAT(input_dim, output_dim).to(device)
    elif args.model_type == 'GIN':
        model = GIN(input_dim, output_dim).to(device)
    elif args.model_type == 'GraphSAGE':
        model = GraphSAGE(input_dim, output_dim).to(device)
    elif args.model_type == 'MLP':
        model = MLP(input_dim, output_dim).to(device)
    elif args.model_type == 'SGCN':
        model = SGCN(input_dim, output_dim).to(device)
    elif args.model_type == 'LeNet':
        model = LeNet(in_channel=1,out_channel=output_dim).to(device)
    elif args.model_type == '1D-CNN':
        model = CNN_1D(in_channel=1,out_channel=output_dim).to(device)
    elif args.model_type == 'LSTM_GCN':
        model = LSTM_GCN(input_dim, output_dim).to(device)
    elif args.model_type == 'GDC_LSTM':
        model = LSTM_GDC(input_dim, output_dim).to(device)
    elif args.model_type == 'RFAConv':
        model = LSTM_RFCA(input_dim, output_dim).to(device)
    else:
        print('this model is not existed!!!')

    # ==============================================================3、超参数===================================================
    epochs = args.epochs
    lr = args.learning_rate
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=args.momentum)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    else:
        print('this optimizer is not existed!!!')

    # ==============================================================4、训练===================================================
    all_train_loss = []
    all_train_accuracy = []
    train_time = []

    for epoch in range(epochs):

        start = time.perf_counter()

        model.train()
        correct_train = 0
        train_loss = 0
        for step, train_data in enumerate(loader_train):
            train_data = train_data.to(device)
            train_out = model(train_data)
            loss = F.nll_loss(train_out, train_data.y)  # 负对数似然损失,如用交叉熵损失：F.cross_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            pre_train = torch.max(train_out.cpu(), dim=1)[1].data.numpy()
            correct_train = correct_train + (pre_train == train_data.y.cpu().data.numpy()).astype(int).sum()

        end = time.perf_counter()

        train_time.append(end-start)  #记录训练时间

        train_accuracy = correct_train / (len(loader_train.dataset) * loader_train.dataset[0].num_nodes)
        all_train_loss.append(train_loss)
        all_train_accuracy.append(train_accuracy)

        logger.info('epoch：{} '
              '| train loss：{:.4f} '
              '| train accuracy：{}/{}({:.4f}) '
              '| train time：{}(s/epoch)'.format(
            epoch,train_loss,correct_train,len(loader_train.dataset) * loader_train.dataset[0].num_nodes,100*train_accuracy,end-start))
        print('epoch：{} '
              '| train loss：{:.4f} '
              '| train accuracy：{}/{}({:.4f}) '
              '| train time：{}(s/epoch)'.format(
            epoch,train_loss,correct_train,len(loader_train.dataset) * loader_train.dataset[0].num_nodes,100*train_accuracy,end-start))
        # 验证步骤
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 禁用梯度计算
            for data in loader_val:
                data = data.to(device)
                output = model(data)
                loss = F.cross_entropy(output, data.y)
                val_loss += loss.item() * data.size(0)

        val_loss = val_loss / len(loader_val.dataset)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}/{len(loader_val.dataset)}, Validation Loss: {val_loss:.4f}')
    # ==============================================================5、测试===================================================
    y_fea = []
    list(map(lambda x:y_fea.append([]),range(len(model.get_fea()))))  # y_fea = [] 根据可视化的层数来创建相应数量的空列表存放特征

    prediction = np.empty(0,)  #存放预测标签绘制混淆矩阵
    # 初始化变量来存储所有测试标签和预测结果
    true_labels = []
    predicted_labels = []
    model.eval()
    correct_test = 0
    for test_data in loader_test:
        test_data = test_data.to(device)
        test_out = model(test_data)
        pre_test = torch.max(test_out.cpu(),dim=1)[1].data.numpy()
        correct_test = correct_test + (pre_test == test_data.y.cpu().data.numpy()).astype(int).sum()
        prediction = np.append(prediction,pre_test) #保存预测结果---混淆矩阵
        list(map(lambda j: y_fea[j].extend(model.get_fea()[j].cpu().detach().numpy()),range(len(y_fea)))) #保存每一层特征---tsne
        # 测试标签和预测结果
        true_labels.extend(test_data.y.cpu().data.numpy())
        predicted_labels.extend(pre_test)

    test_accuracy = correct_test / (len(loader_test.dataset)*loader_test.dataset[0].num_nodes)

    # 计算测试集中所有图的节点总数
    total_num_nodes = sum([data.num_nodes for data in loader_test.dataset])

    # 计算测试集的准确率
    test_accuracy = correct_test / total_num_nodes

    # 打印和记录测试集的准确率
    print('test accuracy：{}/{}({:.4f}%)'.format(correct_test, total_num_nodes, 100 * test_accuracy))
    logger.info('test accuracy：{}/{}({:.4f}%)'.format(correct_test, total_num_nodes, 100 * test_accuracy))

    # 打印和记录训练时间
    print('all train time：{}(s/100epoch)'.format(np.array(train_time).sum()))
    logger.info('all train time：{}(s/100epoch)'.format(np.array(train_time).sum()))

    # 确定 num_classes
    num_classes = len(set(true_labels))

    # 添加计算每个类别的准确率的部分
    class_correct = np.zeros(num_classes)  # 用于存储每个类别预测正确的样本数量
    class_total = np.zeros(num_classes)  # 用于存储每个类别的总样本数量

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == predicted_label:
            class_correct[true_label] += 1
        class_total[true_label] += 1

    # 计算每个类别的准确率
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
        else:
            accuracy = 0  # 避免除以零错误
        print('Class {}: Accuracy: {:.4f}'.format(i, accuracy))
        logger.info('Class {}: Accuracy: {:.4f}'.format(i, accuracy))

    # 1.计算指标
    # 计算各类别的指标
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)
    accracy = f1_score(true_labels, predicted_labels, average=None)

    # 输出指标
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print('Class {}: Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(i, p, r, f))
        logger.info('Class {}: Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}'.format(i, p, r, f))

    # 2.计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("ROC曲线{}.png".format(uuid.uuid4().hex))

    # 4.绘制训练损失和训练准确率随epoch变化的曲线图
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 绘制训练损失曲线
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('训练损失', color=color)
    ax1.plot(range(1, epochs + 1), all_train_loss, label='训练损失', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建次坐标系
    ax2 = ax1.twinx()

    # 绘制训练准确率曲线
    color = 'tab:red'
    ax2.set_ylabel('训练准确率', color=color)
    ax2.plot(range(1, epochs + 1), all_train_accuracy, label='训练准确率', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    fig.tight_layout()
    fig.legend(loc='upper right')

    # plt.show()
    plt.savefig("损失图{}.png".format(uuid.uuid4().hex))

    if args.visualization == True:
        visualization_confusion(loader_test=loader_test,prediction=prediction)  #混淆矩阵
        for num,fea in enumerate(y_fea):
            visualization_tsne(loader_test=loader_test,y_feature=np.array(fea),classes=output_dim)  #t-SNE可视化
        # plt.show()
        # plt.savefig(os.path.join(args.output_path, '{}.png'.format(uuid.uuid4().hex)))



