# -*- coding: utf-8 -*-
"""
@Time    : 2024-03-23 4:24
@Author  : lxc
@File    : 数据集的读取.py
@Desc    :

"""
import os
import numpy as np
from scipy.io import loadmat
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
data_path = r"E:\故障诊断数据集\凯斯西储大学数据\12k Drive End Bearing Fault Data"
# 读取MAT文件
data1 = loadmat(os.path.join(data_path, '100.mat'))  # 正常信号 转速1730r/min
data2 = loadmat(os.path.join(data_path, '169.mat'))  # 0.14mm 内圈 转速1730r/min
data3 = loadmat(os.path.join(data_path, '185.mat'))  # 0.14mm 滚珠 转速1730r/min
data4 = loadmat(os.path.join(data_path, '197.mat'))  # 0.14mm 外圈 转速1730r/min
# 注意，读取出来的data是字典格式，可以通过函数type(data)查看。
# 第二步，数据集中统一读取 驱动端加速度数据，取一个长度为1024的信号进行后续观察和实验
# DE - drive end accelerometer data 驱动端加速度数据
data_list1 = data1['X100_DE_time'].reshape(-1)
data_list2 = data2['X169_DE_time'].reshape(-1)
data_list3 = data3['X185_DE_time'].reshape(-1)
data_list4 = data4['X197_DE_time'].reshape(-1)
# 划窗取值（大多数窗口大小为1024）
data_list1 = data_list1[0:256]
data_list2 = data_list2[0:256]
data_list3 = data_list3[0:256]
data_list4 = data_list4[0:256]
# 第三步，进行数据可视化
print("#"*50)
print("进行数据可视化")
print("#"*50)
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
plt.plot(data_list1)
plt.title('正常')
plt.subplot(2,2,2)
plt.plot(data_list2)
plt.title('内圈')
plt.subplot(2,2,3)
plt.plot(data_list3)
plt.title('滚珠')
plt.subplot(2,2,4)
plt.plot(data_list4)
plt.title('外圈')
plt.savefig("原数据展示.jpg")
plt.show()
# STFT与参数选择

from scipy.signal import stft
print("#"*50)
print("STFT与参数选择")
print("#"*50)

# 设置STFT参数
window_size = 256  # 窗口大小
overlap = 0.5  # 重叠比例
# 计算重叠的样本数
overlap_samples = int(window_size * overlap)
frequencies1, times1, magnitude1 = stft(data_list1, nperseg=window_size, noverlap=overlap_samples)
print(frequencies1)
print(times1)
print(magnitude1)
# 设置STFT参数
# window_size = 64  # 窗口大小
# overlap = 0.5  # 重叠比例
# 计算重叠的样本数
overlap_samples = int(window_size * overlap)
frequencies2, times2, magnitude2 = stft(data_list2, nperseg=window_size, noverlap=overlap_samples)

# 设置STFT参数
# window_size = 128  # 窗口大小
# overlap = 0.5  # 重叠比例
# 计算重叠的样本数
overlap_samples = int(window_size * overlap)
frequencies3, times3, magnitude3 = stft(data_list3, nperseg=window_size, noverlap=overlap_samples)


# 设置STFT参数
# window_size = 128  # 窗口大小
# overlap = 0.5  # 重叠比例
# 计算重叠的样本数
overlap_samples = int(window_size * overlap)
frequencies4, times4, magnitude4 = stft(data_list4, nperseg=window_size, noverlap=overlap_samples)

# 数据可视化
print("#"*50)
print("SFTF数据可视化")
print("#"*50)


plt.figure(figsize=(20,10), dpi=100)
plt.subplot(2,2,1)
plt.pcolormesh(times1, frequencies1, np.abs(magnitude1), shading='gouraud')
plt.title('正常')
plt.subplot(2,2,2)
plt.pcolormesh(times2, frequencies2, np.abs(magnitude2), shading='gouraud')
plt.title('内圈')
plt.subplot(2,2,3)
plt.pcolormesh(times3, frequencies3, np.abs(magnitude3), shading='gouraud')
plt.title('滚珠')
plt.subplot(2,2,4)
plt.pcolormesh(times4, frequencies4, np.abs(magnitude4), shading='gouraud')
plt.title('外圈')
plt.savefig("样本时频图展示.jpg")
plt.show()