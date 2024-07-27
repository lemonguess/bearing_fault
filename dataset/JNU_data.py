import math
import random
from collections import Counter
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from dataset.__construct_graph import generate_graph, generate_sftf_graph
from scipy.signal import stft
from imblearn.over_sampling import SMOTE
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import DataLoader as GeoDataLoader
def data_preprocessing(dataset_path,sample_number,window_size,overlap,normalization,noise,snr,
                         input_type,graph_type,K,p,node_num,direction,edge_type,edge_norm,train_size,batch_size):

    root = dataset_path

    health = ['n600_3_2.csv', 'n800_3_2.csv', 'n1000_3_2.csv']  # 600 800 1000转速下的正常信号
    inner = ['ib600_2.csv', 'ib800_2.csv', 'ib1000_2.csv']  # 600 800 1000转速下的内圈故障信号
    outer = ['ob600_2.csv', 'ob800_2.csv', 'ob1000_2.csv']  # 600 800 1000转速下的外圈故障信号
    ball = ['tb600_2.csv', 'tb800_2.csv', 'tb1000_2.csv']  # 600 800 1000转速下的滚动体故障信号

    file_name = []  # 存放三种转速下、四种故障状态的文件名，一共12种类型
    file_name.extend(health)
    file_name.extend(inner)
    file_name.extend(outer)
    file_name.extend(ball)

    data1 = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据(每一类数据不平衡)
    for num, each_name in enumerate(file_name):
        dir = os.path.join(root, each_name)
        with open(dir, "r", encoding='gb18030', errors='ignore') as f:
            for line in f:
                line = float(line.strip('\n'))  # 删除每一行后的换行符号，并将字符型转化为数字
                data1[num].append(line)  # 将取出来的数据逐个存放到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据（每一类数据平衡）shape：(12,500500)
    for data1_i in range(len(data1)):
        data[data1_i].append(data1[data1_i][:500500])  # 将所有类型数据总长度截取为500500

    data = np.array(data).squeeze(axis=1)  # shape：(12,500500)

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], math.floor((noise_data.shape[1] - window_size) / overlap + 1), window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,
                                                          overlap=overlap)

    # sample_data = sample_data[:, :sample_number, :]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    if input_type == 'TD':  #时域信号
        data = norm_data
    elif input_type == 'FD':  #频域信号
        data = np.zeros((norm_data.shape[0],norm_data.shape[1],norm_data.shape[2]))
        for label_index in range(norm_data.shape[0]):
            fft_data = FFT(norm_data[label_index,:,:])
            data[label_index,:,:] = fft_data
    elif input_type == 'TFD':  # 时频域
        # overlap_samples = window_size - overlap  # 计算重叠的样本数
        _, _, magnitude = stft(norm_data[0, :, :], nperseg=1024, noverlap=512)
        data = np.zeros((norm_data.shape[0], norm_data.shape[1], magnitude.shape[1], magnitude.shape[2]))
        # data = np.zeros((norm_data.shape[0], norm_data.shape[1], norm_data.shape[2]))
        for label_index in range(norm_data.shape[0]):
            _, _, magnitude = stft(norm_data[label_index,:,:], nperseg=1024, noverlap=512)
            # 提取频谱特征
            spectral_features = np.abs(magnitude)  # 这里简单地取频谱的幅度
            # 对频谱特征进行归一化处理
            normalized_features = (spectral_features - np.mean(spectral_features)) / np.std(spectral_features)
            # 先对后两维进行转置
            # nadarry_transposed = normalized_features.transpose(0, 2, 1)  # 转置第二维和第三维
            data[label_index, :, :] = normalized_features
        # 时频域信号处理
        # arr_transposed = data.transpose(0, 1, 3, 2)
        # data = arr_transposed.reshape(data.shape[0], data.shape[1] * data.shape[3], data.shape[2])
        # graph_dataset = generate_sftf_graph(feature=data,graph_type=graph_type,node_num=node_num,direction=direction,
    data = data[:, :(node_num*sample_number), :]
    data = data[:, :sample_number, :]
    graph_dataset = generate_graph(feature=data,graph_type=graph_type,node_num=node_num,direction=direction,
                                   edge_type=edge_type,edge_norm=edge_norm,K=K,p=p)

    # 获取标签列表
    str_y = [data.y[0].item() if data.y.dim() > 0 else data.y.item() for data in graph_dataset]

    # 划分训练集和剩余数据集
    train_data, remaining_data = train_test_split(graph_dataset, train_size=train_size, shuffle=True, stratify=str_y)
    # train_data, remaining_data = balanced_split(graph_dataset, str_y, test_size=0.7)

    # 获取剩余数据集的标签列表
    remaining_labels = [data.y[0].item() if data.y.dim() > 0 else data.y.item() for data in remaining_data]

    # 使用均衡划分策略划分测试集和验证集
    test_data, val_data = balanced_split(remaining_data, remaining_labels, test_size=0.5)

    # 创建数据加载器
    loader_train = GeoDataLoader(train_data, batch_size=batch_size)
    loader_test = GeoDataLoader(test_data, batch_size=batch_size)
    loader_val = GeoDataLoader(val_data, batch_size=batch_size)


    return loader_train, loader_test, loader_val

def resampled(data):
    adjusted_data = []
    # 统计每个数字出现的次数
    counter = Counter(data)
    # 找出出现次数最少的数字的频次
    _min = min(counter.values())

    for i in list(counter.keys()):
        for _ in range(_min):
            adjusted_data.append(i)
    rate = len(adjusted_data)/len(data)
    random.shuffle(adjusted_data)
    # 遍历原始数据
    for k, v in counter.items():
        # 如果该数字的剩余保留次数大于0，则添加到调整后的列表中，并减少剩余次数
        if v > _min:
            for i in range(v-_min):
                adjusted_data.append(k)
    return rate, adjusted_data


def balanced_split(dataset, labels, test_size=0.5):
    # 根据标签创建索引映射
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    test_indices = []
    val_indices = []

    # 对每个类别进行均衡划分
    for label, indices in label_to_indices.items():
        # 如果类别样本太少，可能无法均匀划分
        if len(indices) < 2:
            raise ValueError(f"Not enough samples for label {label} to split.")

        # 随机划分当前类别的样本
        test_idx, val_idx = train_test_split(indices, test_size=test_size, shuffle=True)
        test_indices.extend(test_idx)
        val_indices.extend(val_idx)

    # 创建数据子集
    test_subset = Subset(dataset, test_indices)
    val_subset = Subset(dataset, val_indices)

    return test_subset, val_subset